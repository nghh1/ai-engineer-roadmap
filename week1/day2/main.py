# 5. Modules and imports
from text_utils.statistics import word_count
from text_utils.cleaning import normalise

if __name__ == "__main__":
    text = " I am Ivan Ng "
    cleaned = normalise(text)
    count = word_count(cleaned)
    print(f"{cleaned} {count}")