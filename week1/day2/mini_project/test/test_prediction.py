from pipeline.prediction import predict
class TestPredict:
    def test_positive(self):
        assert predict("I love you") == "positive"
    def test_negative(self):
        assert predict("I hate you") == "negative"
    def test_neutral(self):
        assert predict("I am kind") == "neutral"