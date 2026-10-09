from text.pipeline_components.normalization.english_normalization import normalize_english

class TestEnglishNormalization:

    def test_whitespace_collapse(self):
        output = normalize_english(" \t \n   normalized   \n \t ")
        assert output == " normalized "

    def test_lowercase(self):
        output = normalize_english("This is a test of Normalization.")
        assert output == "this is a test of normalization."

    def test_numbers(self):
        output = normalize_english("1,000")
        assert output == "one thousand"

    def test_abbreviations(self):
        output = normalize_english("mr. ed")
        assert output == "mister ed"