from pipeline.preprocessing import preprocess_text
class TestPreprocessText:
    def test_lowercase(self):
        assert preprocess_text("I am Ivan") == "i am ivan"
    def test_strip(self):
        assert preprocess_text(" i am ivan ") == "i am ivan"