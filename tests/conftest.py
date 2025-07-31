import pandas as pd
import pytest


def assert_raises_with_message(func, expected_msg, *args, **kwargs):
    with pytest.raises((AssertionError, ValueError, Exception)) as exc_info:
        func(*args, **kwargs)
    assert expected_msg in str(exc_info.value)

@pytest.fixture(scope="module")
def tourism_df():
    df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/tourism.csv')
    df = df.rename({'Trips': 'y', 'Quarter': 'ds'}, axis=1)
    df.insert(0, 'Country', 'Australia')
    return df