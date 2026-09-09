from src import loading, cleaning, features, summary

def main():
    raw = loading.load_data("data/raw.csv")
    cleaned = cleaning.clean_data(raw)
    processed = features.add_features(cleaned)
    processed.to_csv("data/processed.csv", index=False)
    summarised = summary.model_summary(processed)
    print(summarised)

if __name__ == "__main__":
    main()