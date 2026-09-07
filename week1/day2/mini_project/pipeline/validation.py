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