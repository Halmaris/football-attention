import pandas as pd
import torch

from football_attention.variants import (
    ShotSequenceDataset,
    fit_vocabularies,
    transform_sequence,
)


def example_row() -> pd.Series:
    return pd.Series(
        {
            'player_id': [10, 11, 12],
            'type_name': ['Pass', 'Carry', 'Shot'],
            'outcome_name': ['Success', 'Success', 'Goal'],
            'coordinates': [
                [10.0, 20.0, 30.0, 40.0],
                [30.0, 40.0, 90.0, 40.0],
                [102.0, 40.0, 120.0, 40.0],
            ],
            'delta_seconds': [0.0, 2.0, 1.0],
            'is_shot': [False, False, True],
        }
    )


def test_masked_shot_removes_direct_shot_information() -> None:
    sequence = transform_sequence(example_row(), 'masked_shot')
    assert sequence['player_id'][-1] == -1
    assert sequence['outcome_name'][-1] == '<MASK>'
    assert sequence['continuous'][-1][:4] == [0.0, 0.0, 0.0, 0.0]
    assert sequence['type_name'][-1] == 'Shot'


def test_pre_shot_and_shot_only_are_disjoint() -> None:
    pre_shot = transform_sequence(example_row(), 'pre_shot')
    shot_only = transform_sequence(example_row(), 'shot_only')
    assert not any(pre_shot['is_shot'])
    assert shot_only['is_shot'] == [True]
    assert len(pre_shot['player_id']) == 2
    assert len(shot_only['player_id']) == 1


def test_shuffled_dataset_is_deterministic_and_preserves_events() -> None:
    row = example_row()
    row['shot_seq_id'] = 1
    row['match_id'] = 100
    row['shooter_id'] = 12
    row['xg'] = 0.2
    df = pd.DataFrame([row])
    vocabularies = fit_vocabularies(df)
    ordered = ShotSequenceDataset(
        df,
        vocabularies=vocabularies,
        variant='pre_shot',
        sequence_length=5,
    )[0][0]
    shuffled_dataset = ShotSequenceDataset(
        df,
        vocabularies=vocabularies,
        variant='pre_shot',
        sequence_length=5,
        shuffle_events=True,
    )
    shuffled = shuffled_dataset[0][0]
    repeated = shuffled_dataset[0][0]
    valid = ordered['valid_mask']
    assert torch.equal(shuffled['event_type'], repeated['event_type'])
    assert not torch.equal(
        ordered['event_type'][valid],
        shuffled['event_type'][valid],
    )
    assert torch.equal(
        ordered['event_type'][valid].sort().values,
        shuffled['event_type'][valid].sort().values,
    )
