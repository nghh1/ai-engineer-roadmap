from pipeline.validation import validate_record
class TestRecord:
    def test_valid_record(self):
        assert validate_record({"id": 1, "text": "hello"}) == True
    def test_missing_id(self):
        assert validate_record({"text": "hello"}) == False
    def test_missing_text(self):
        assert validate_record({"id": 1}) == False
    def test_invalid_id(self):
        assert validate_record({"id": 0, "text": "hello"}) == False
        assert validate_record({"id": -1, "text": "hello"}) == False
    def test_whitespace_text(self):
        assert validate_record({"id": 1, "text": " "}) == False
    def test_incorrect_types(self):
        assert validate_record({"id": "1", "text": "hello"}) == False
        assert validate_record({"id": 1, "text": 1}) == False