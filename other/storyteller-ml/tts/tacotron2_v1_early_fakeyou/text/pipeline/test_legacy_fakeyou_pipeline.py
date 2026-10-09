
from text.pipeline.legacy_fakeyou_pipeline import LegacyFakeYouPipeline


class TestLegacyFakeYouPipeline():

    def test_no_input(self):
        pipeline = LegacyFakeYouPipeline()
        output = pipeline.user_input_to_sequence("")
        assert output == []

    def test_single_word(self):
        pipeline = LegacyFakeYouPipeline()
        output = pipeline.user_input_to_sequence("hello")
        # NB: Graphemes remains ASCII! No CMUDict lookup or phoneme prediction.
        # Also note the ending ";".
        assert output == [
            45, 42, 49, 49, 52, # "h e l l o"
            9, # Semicolon line ender
        ] 

    def test_two_words(self):
        pipeline = LegacyFakeYouPipeline()
        output = pipeline.user_input_to_sequence("hello hello")
        # NB: Graphemes remains ASCII! No CMUDict lookup or phoneme prediction.
        assert output == [
            45, 42, 49, 49, 52, # "h e l l o"
            11, # Space
            45, 42, 49, 49, 52, # "h e l l o"
            9, # Semicolon line ender
        ]

    def test_two_words_with_arpabet_segment(self):
        pipeline = LegacyFakeYouPipeline()
        output = pipeline.user_input_to_sequence("hello {HH AH0 L OW1}")
        # NB: Graphemes remains ASCII! No CMUDict lookup or phoneme prediction.
        assert output == [
            45, 42, 49, 49, 52, # "h e l l o"
            11, # Space
            106, 73, 117, 123, # Legacy FakeYou supports manual arpabet inclusion
            9, # Semicolon line ender
        ]
