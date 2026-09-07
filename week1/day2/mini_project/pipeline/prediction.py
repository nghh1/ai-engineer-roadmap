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