from .validation import validate_record
from .preprocessing import preprocess_text
from .prediction import predict
from .models import InputRecord, PredictionRecord
from .batching import batch
def run_pipeline(records: list[InputRecord], 
                 batch_size: int) -> list[PredictionRecord]:
    valid = [record for record in records if validate_record(record)]
    all_results: list[PredictionRecord] = []
    for b in batch(valid, batch_size):
        for r in b:
            text = preprocess_text(r['text'])
            sentiment = predict(text)
            all_results.append(PredictionRecord(id=r["id"], text=r["text"], prediction=sentiment))
    return all_results