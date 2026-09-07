# main script for mini project
from pipeline.pipeline import run_pipeline
import json

def main():
    records = [
        {"id": 1, "text": "I absolutely love this product"},
        {"id": 2, "text": "This is terrible"},
        {"id": 3, "text": "Pretty good overall"}
        ]
    # or records = load_records(file)
    predictions = run_pipeline(records, batch_size=1)
    with open("predictions.json", 'w', encoding='utf-8') as f:
        json.dump(predictions, f)
        
if __name__ == "__main__":
    main()
    
