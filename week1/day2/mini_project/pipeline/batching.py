def batch(items: list, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]