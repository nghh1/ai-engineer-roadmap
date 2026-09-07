# 1. Generator-based batching 
items = [1, 2, 3, 4, 5, 6, 7]

def batch(items: list, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]

for mini_batch in batch(items, 3):
    print(mini_batch)

# 2. Build a generator
records = [
    {"id": 1, "text": "hello"},
    {"id": -1, "text": "bad id"},
    {"id": 2, "text": "world"},
    {"id": 3, "text": "   "}
]

def read_valid_records(records: list[dict]):
    from mini_project.pipeline.validation import validate_record
    for record in records:
        if validate_record(record):
            (yield record)

for record in read_valid_records(records):
    print(record)

# 3. TypedDict exercise
from typing import TypedDict
class InputRecord(TypedDict):
    id: int
    text: str

class PredictionRecord(TypedDict):
    id: int
    text: str
    prediction: str

def validate_record(record: InputRecord) -> bool:
    pass

def preprocess_text(text: str) -> str:
    pass

def predict(text: str) -> str:
    pass

def run_pipeline(records: list[InputRecord]) -> list[PredictionRecord]:
    pass

# 4. Functions as arguments
from collections.abc import Callable
def lowercase(text: str) -> str:
    return text.lower()
def remove_spaces(text: str) -> str:
    return text.replace(" ", "")
def process_text(text: str, processor: Callable[[str], str]):
    return processor(text)
lower = process_text("HELLO WORLD", lowercase)
print(lower)
removed = process_text("HELLO WORLD", remove_spaces)
print(removed)
