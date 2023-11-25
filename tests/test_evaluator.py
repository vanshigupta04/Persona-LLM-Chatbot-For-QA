import pandas as pd
import pytest

@pytest.fixture
def df():
    return pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6]
    })

def test_evaluator(df):
    assert df['a'].sum() == 6

if __name__ == "__main__":
    pytest.main()
