from pipeline.pipeline import run_pipeline
def test_skip_unvalid():
    records = [
        {"id": 1, "text": "I love this"},
        {"id": -1, "text": "bad id"},
        {"id": 3, "text": "good product"}
    ]
    result = run_pipeline(records, 1)
    assert len(result) == 2
    assert result[0]["id"] == 1
    assert result[1]["id"] == 3

def test_empty_records():
    result = run_pipeline([], 1)
    assert result == []

def test_different_size():
    records = [
        {"id": 1, "text": "I love this"},
        {"id": 2, "text": "bad id"},
        {"id": 3, "text": "good product"},
        {"id": 4, "text": "This is what I want"}
    ]
    result1 = run_pipeline(records, 1)
    result2 = run_pipeline(records, 2)
    result3 = run_pipeline(records, 5)
    assert result1 == result2
    assert result2 == result3