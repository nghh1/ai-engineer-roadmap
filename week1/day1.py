# 1. Given a list of numbers, return mean, min, max, and median
def firstQuestion(numbers: list[int]) -> tuple[float, int, int, float]:
    mean = sum(numbers) / len(numbers)
    min = min(numbers)
    max = max(numbers)
    ordered = sorted(numbers)
    if len(ordered) % 2 == 1:
        median = ordered[len(ordered)//2]
    else:
        median = (ordered[(len(ordered)//2)-1] + ordered[len(ordered)//2]) / 2
    return mean, min, max, median

# 2. Given a list of words, produce a frequency dictionary
def secondQuestion(words: list[str]) -> dict[str, int]:
    frequency_dict = {}
    for word in words:
        frequency_dict[word] = frequency_dict.get(word, 0) + 1
    return frequency_dict

# 3. Remove duplicates from a list while preserving order
def thirdQuestion(list_duplicates: list[str]) -> list[str]:
    record = set()
    deduplicate = list()
    for i in list_duplicates:
        if i not in record:
            deduplicate.append(i)
        record.add(i)
    return deduplicate

# 4. Write functions to filter models with accuracy >= 0.88, sort them by latency, 
# and return the name of the best-accuracy model
def fourthQuestion(models: list[dict[str, str|float|int]]) -> str:
    filtered = list(model for model in models if model["accuracy"]>=0.88)
    sorted_by_latency = sorted(filtered, key=lambda model: model["latency"])
    most_accurate = max(filtered, key=lambda model: model["accuracy"])
    return most_accurate["name"]

# 5. Write a batch(items: list, batch_size: int)
def batch(items: list, batch_size: int):
    result = []
    for i in range(0, len(items), batch_size):
        result.append(items[i:i+batch_size])

# Mini-project: tiny inference pipeline
import json
records = [
    {"id": 1, "text": "I absolutely love this product"},
    {"id": 2, "text": "This is terrible"},
    {"id": 3, "text": "Pretty good overall"}
]

def validate_record(record: dict) -> bool:
    try:
        # Record type check
        if not isinstance(record, dict):
            raise TypeError("Record must be a dictionary")
        # Key existence check
        missing = set(["id", "text"]) - record.keys()
        if missing:
            raise KeyError(f"Missing required keys: {', '.join(missing)} in the record")
        # Type check
        if not isinstance(record["id"], int):
            raise TypeError(f"'id' must be an int, got {type(record['id'])}")
        if not isinstance(record["text"], str):
            raise TypeError(f"'text' must be a str, got {type(record['text'])}")
        # Value check
        if record["id"] <= 0:
            raise ValueError("'id' must be a postive integer")
        if not record["text"].strip():
            raise ValueError("'text' must be a non-empty string")
    except (TypeError, KeyError, ValueError):
        return False
    else:
        return True

def preprocess_text(text: str) -> str:
    text = text.strip().lower()
    return text

def predict(text: str) -> str:
    positive_words = {"love", "good", "great", "excellent"}
    negative_words = {"bad", "hate", "terrible", "awful"}
    check = set(text.split())
    if positive_words & check:
        return "positive"
    elif negative_words & check:
        return "negative"
    else:
        return "neutral"

def run_pipeline(records: list[dict]) -> list[dict]:
    result = []
    for record in records:
        if not validate_record(record):
            continue
        text = preprocess_text(record["text"])
        sentiment = predict(text)
        result.append({"id": record["id"],
                       "text": record["text"],
                       "prediction": sentiment})
    return result

results = run_pipeline(records)
with open("predictions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

