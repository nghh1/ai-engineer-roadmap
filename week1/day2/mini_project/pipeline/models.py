from typing import TypedDict
class InputRecord(TypedDict):
    id: int
    text: str

class PredictionRecord(TypedDict):
    id: int
    text: str
    prediction: str