from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
import torch
from .data import PITCH_X, PITCH_Y


VALID_VARIANTS = {'full', 'masked_shot', 'pre_shot', 'shot_only'}


@dataclass(frozen=True)
class Vocabularies:
    player: dict[int, int]
    event_type: dict[str, int]
    outcome: dict[str, int]

    @property
    def n_players(self) -> int:
        return max(self.player.values()) + 1

    @property
    def n_event_types(self) -> int:
        return max(self.event_type.values()) + 1

    @property
    def n_outcomes(self) -> int:
        return max(self.outcome.values()) + 1


def _categorical_vocab(values: Sequence[str]) -> dict[str, int]:
    vocab = {'<PAD>': 0, '<UNK>': 1, '<MASK>': 2}
    for value in sorted(set(map(str, values))):
        if value not in vocab:
            vocab[value] = len(vocab)
    return vocab


def fit_vocabularies(train_sequences: pd.DataFrame) -> Vocabularies:
    players = {
        int(player_id)
        for sequence in train_sequences['player_id']
        for player_id in sequence
        if int(player_id) != -1
    }
    player_vocab = {-1: 1}
    player_vocab.update(
        {player_id: index + 2 for index, player_id in enumerate(sorted(players))}
    )
    event_types = [
        str(value)
        for sequence in train_sequences['type_name']
        for value in sequence
    ]
    outcomes = [
        str(value)
        for sequence in train_sequences['outcome_name']
        for value in sequence
    ]
    return Vocabularies(
        player=player_vocab,
        event_type=_categorical_vocab(event_types),
        outcome=_categorical_vocab(outcomes),
    )


def _variant_indices(is_shot: list[bool], variant: str) -> list[int]:
    if variant not in VALID_VARIANTS:
        raise ValueError(f'Unknown sequence variant: {variant}')
    if variant in {'full', 'masked_shot'}:
        return list(range(len(is_shot)))
    if variant == 'pre_shot':
        return [index for index, shot in enumerate(is_shot) if not shot]
    return [index for index, shot in enumerate(is_shot) if shot]


def transform_sequence(
    row: pd.Series,
    variant: str,
) -> dict[str, list[object]]:
    is_shot = [bool(value) for value in row['is_shot']]
    indices = _variant_indices(is_shot, variant)
    players = [int(row['player_id'][index]) for index in indices]
    event_types = [str(row['type_name'][index]) for index in indices]
    outcomes = [str(row['outcome_name'][index]) for index in indices]
    coordinates = [list(row['coordinates'][index]) for index in indices]
    deltas = [float(row['delta_seconds'][index]) for index in indices]
    shots = [is_shot[index] for index in indices]

    if variant == 'masked_shot' and any(shots):
        shot_index = max(index for index, shot in enumerate(shots) if shot)
        players[shot_index] = -1
        outcomes[shot_index] = '<MASK>'
        coordinates[shot_index] = [0.0, 0.0, 0.0, 0.0]

    continuous = [
        [
            float(values[0]) / PITCH_X,
            float(values[1]) / PITCH_Y,
            float(values[2]) / PITCH_X,
            float(values[3]) / PITCH_Y,
            float(np.log1p(max(0.0, delta)) / np.log1p(10.0)),
        ]
        for values, delta in zip(coordinates, deltas)
    ]
    return {
        'player_id': players,
        'type_name': event_types,
        'outcome_name': outcomes,
        'continuous': continuous,
        'is_shot': shots,
    }


class ShotSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sequences: pd.DataFrame,
        *,
        vocabularies: Vocabularies,
        variant: str,
        sequence_length: int,
        shuffle_events: bool = False,
        shuffle_seed: int = 2026,
    ) -> None:
        if variant not in VALID_VARIANTS:
            raise ValueError(f'Unknown sequence variant: {variant}')
        self.sequences = sequences.reset_index(drop=True)
        self.vocabularies = vocabularies
        self.variant = variant
        self.sequence_length = sequence_length
        self.shuffle_events = shuffle_events
        self.shuffle_seed = shuffle_seed

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        row = self.sequences.iloc[index]
        sequence = transform_sequence(row, self.variant)
        n_tokens = min(len(sequence['player_id']), self.sequence_length)
        start = len(sequence['player_id']) - n_tokens
        pad = self.sequence_length - n_tokens
        token_slice = slice(start, None)

        original_players = np.full(self.sequence_length, -1, dtype=np.int64)
        player_ids = np.zeros(self.sequence_length, dtype=np.int64)
        type_ids = np.zeros(self.sequence_length, dtype=np.int64)
        outcome_ids = np.zeros(self.sequence_length, dtype=np.int64)
        continuous = np.zeros((self.sequence_length, 5), dtype=np.float32)
        is_shot = np.zeros(self.sequence_length, dtype=bool)
        valid = np.zeros(self.sequence_length, dtype=bool)

        if n_tokens:
            players = sequence['player_id'][token_slice]
            event_types = sequence['type_name'][token_slice]
            outcomes = sequence['outcome_name'][token_slice]
            original_players[pad:] = np.asarray(players, dtype=np.int64)
            player_ids[pad:] = np.asarray(
                [self.vocabularies.player.get(value, 1) for value in players],
                dtype=np.int64,
            )
            type_ids[pad:] = np.asarray(
                [self.vocabularies.event_type.get(value, 1) for value in event_types],
                dtype=np.int64,
            )
            outcome_ids[pad:] = np.asarray(
                [self.vocabularies.outcome.get(value, 1) for value in outcomes],
                dtype=np.int64,
            )
            continuous[pad:] = np.asarray(
                sequence['continuous'][token_slice],
                dtype=np.float32,
            )
            is_shot[pad:] = np.asarray(sequence['is_shot'][token_slice], dtype=bool)
            valid[pad:] = True

        if self.shuffle_events and n_tokens > 1:
            positions = np.flatnonzero(valid)
            generator = np.random.default_rng(
                self.shuffle_seed + int(row['shot_seq_id'])
            )
            permutation = generator.permutation(len(positions))
            for values in [
                original_players,
                player_ids,
                type_ids,
                outcome_ids,
                continuous,
                is_shot,
            ]:
                original = values[positions].copy()
                values[positions] = original[permutation]

        item = {
            'player': torch.from_numpy(player_ids),
            'event_type': torch.from_numpy(type_ids),
            'outcome': torch.from_numpy(outcome_ids),
            'continuous': torch.from_numpy(continuous),
            'valid_mask': torch.from_numpy(valid),
            'original_player_id': torch.from_numpy(original_players),
            'is_shot': torch.from_numpy(is_shot),
            'shot_seq_id': torch.tensor(int(row['shot_seq_id']), dtype=torch.int64),
            'match_id': torch.tensor(int(row['match_id']), dtype=torch.int64),
            'shooter_id': torch.tensor(int(row['shooter_id']), dtype=torch.int64),
        }
        return item, torch.tensor(float(row['xg']), dtype=torch.float32)


def collate_sequences(
    batch: list[tuple[dict[str, torch.Tensor], torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    keys = batch[0][0].keys()
    items = {
        key: torch.stack([sample[0][key] for sample in batch])
        for key in keys
    }
    targets = torch.stack([sample[1] for sample in batch])
    return items, targets
