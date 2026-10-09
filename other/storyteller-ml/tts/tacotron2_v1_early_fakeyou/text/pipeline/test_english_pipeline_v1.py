from text.pipeline.english_pipeline_v1 import EnglishPipelineV1
from text.pipeline_components.phonetization.cmudict import DEFAULT_CMUDICT_PATH, CMUDict
from text.pipeline_components.phonetization.english_phonetization import EnglishPhonetization


class TestEnglishPipelineV1:

    def build_pipeline_instance(self):
        # NB: CMUDict used in production may differ
        cmudict = CMUDict(DEFAULT_CMUDICT_PATH)
        phonetization = EnglishPhonetization(cmudict)
        return EnglishPipelineV1(phonetization)

    def test_no_input(self):
        pipeline = self.build_pipeline_instance()
        output = pipeline.user_input_to_sequence("")
        assert output == []

    def test_whitespace(self):
        pipeline = self.build_pipeline_instance()
        output = pipeline.user_input_to_sequence("   \n    \t ")
        assert output == []

    def test_single_word(self):
        pipeline = self.build_pipeline_instance()
        output = pipeline.user_input_to_sequence("hello")
        assert output == [106, 73, 117, 123] # NB: Arpabet encoded

    def test_two_words(self):
        pipeline = self.build_pipeline_instance()
        output = pipeline.user_input_to_sequence("hello hello")
        assert output == [
            106, 73, 117, 123, # NB: Arpabet encoded
            11, # Space
            106, 73, 117, 123, # Arpabet encoded
        ]

    def test_two_words_with_arpabet_segment(self):
        pipeline = self.build_pipeline_instance()
        output = pipeline.user_input_to_sequence("hello {HH AH0 L OW1}")
        assert output == [
            106, 73, 117, 123, # NB: Arpabet encoded
            11, # Space
            106, 73, 117, 123, # Manually arpabet encoded
        ]

    def test_words_and_punctuation(self):
        pipeline = self.build_pipeline_instance()
        output = pipeline.user_input_to_sequence("hello friends, welcome!")
        assert output == [
            106, 73, 117, 123, # NB: Arpabet encoded
            11, # Space
            104, 130, 94, 119, 90, 146,
            11, # Space (TODO: Get rid of extraneous spaces with richer parsing)
            6,  # Comma
            11, # Space
            144, 94, 117, 116, 73, 118, 
            11, # Space
            2, # Exclamation mark
        ]

    def test_apostrophe_word(self):
        pipeline = self.build_pipeline_instance()
        output = pipeline.user_input_to_sequence("you're") 
        assert output == [145, 137, 130] # YOU'RE  Y UH1 R   ~   145  137  130
