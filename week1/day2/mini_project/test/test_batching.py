from pipeline.batching import batch
import pytest

@pytest.mark.parametrize(
    "items, batch_size, expected",
    [
        # batch_size=2
        ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]),
        ([], 2, []),
        # batch_size=1
        ([1, 2, 3], 1, [[1], [2], [3]])
    ]
)
def test_batch(items, batch_size, expected):
    result = list(batch(items, batch_size))
    assert result == expected

@pytest.mark.parametrize("invalid_size", [0, -5])
def test_invalid_size(invalid_size):
    with pytest.raises(ValueError):
        list(batch([1, 2, 3], invalid_size))