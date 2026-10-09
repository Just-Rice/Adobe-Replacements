from text.pipeline_components.phonetization.cmudict import DEFAULT_CMUDICT_PATH, CMUDict
from text.pipeline_components.phonetization.english_phonetization import EnglishPhonetization
from text.pipeline_components.input_parsing.parser import ArpabetSequence
from text.pipeline_components.input_parsing.parser import GraphemeSequence

class TestEnglishPhonetization:

    def test_no_input(self):
        phoneme_dictionary = CMUDict(DEFAULT_CMUDICT_PATH)
        phonetization = EnglishPhonetization(phoneme_dictionary)
        output = phonetization.try_convert_to_phonemes("")
        assert output == []

    def test_all_words_in_dictionary(self):
        phoneme_dictionary = CMUDict(DEFAULT_CMUDICT_PATH)
        phonetization = EnglishPhonetization(phoneme_dictionary)
        output = phonetization.try_convert_to_phonemes("this is a test")
        assert output == [
            ArpabetSequence("DH IH1 S"), 
            ArpabetSequence("IH1 Z"),
            ArpabetSequence("AH0"), 
            ArpabetSequence("T EH1 S T"),
        ]

    def test_most_words_in_dictionary(self):
        phoneme_dictionary = CMUDict(DEFAULT_CMUDICT_PATH)
        phonetization = EnglishPhonetization(phoneme_dictionary)
        output = phonetization.try_convert_to_phonemes("nonword is a test")
        assert output == [
            GraphemeSequence("nonword"), 
            ArpabetSequence("IH1 Z"), 
            ArpabetSequence("AH0"), 
            ArpabetSequence("T EH1 S T"),
        ]

    def test_words_with_apostrophe(self):
        phoneme_dictionary = CMUDict(DEFAULT_CMUDICT_PATH)
        phonetization = EnglishPhonetization(phoneme_dictionary)
        output = phonetization.try_convert_to_phonemes("I'm can't you're we're ain't")
        assert output == [
            ArpabetSequence("AY1 M"), 
            ArpabetSequence("K AE1 N T"), 
            ArpabetSequence("Y UH1 R"),
            ArpabetSequence("W IY1 R"),
            ArpabetSequence("EY1 N T"),
        ]

    def test_words_with_hyphenation(self):
        phoneme_dictionary = CMUDict(DEFAULT_CMUDICT_PATH)
        phonetization = EnglishPhonetization(phoneme_dictionary)
        output = phonetization.try_convert_to_phonemes("avant-garde bake-off")
        assert output == [
            ArpabetSequence("AH0 V AA1 N T G AA1 R D"), 
            ArpabetSequence("B EY1 K AO1 F"), 
        ]

    def test_punctuation(self):
        phoneme_dictionary = CMUDict(DEFAULT_CMUDICT_PATH)
        phonetization = EnglishPhonetization(phoneme_dictionary)
        output = phonetization.try_convert_to_phonemes("this...")
        assert output == [
            ArpabetSequence("DH IH1 S"), 
            GraphemeSequence("..."),
        ]
        output = phonetization.try_convert_to_phonemes("this!")
        assert output == [
            ArpabetSequence("DH IH1 S"), 
            GraphemeSequence("!"),
        ]
        output = phonetization.try_convert_to_phonemes("this?")
        assert output == [
            ArpabetSequence("DH IH1 S"), 
            GraphemeSequence("?"),
        ]

    def test_extra_spaces(self):
        phoneme_dictionary = CMUDict(DEFAULT_CMUDICT_PATH)
        phonetization = EnglishPhonetization(phoneme_dictionary)
        output = phonetization.try_convert_to_phonemes("    this    is    a     test   ")
        assert output == [
            ArpabetSequence("DH IH1 S"), 
            ArpabetSequence("IH1 Z"),
            ArpabetSequence("AH0"), 
            ArpabetSequence("T EH1 S T"),
        ]

    def test_multi_non_word_extra_spaces(self):
        phoneme_dictionary = CMUDict(DEFAULT_CMUDICT_PATH)
        phonetization = EnglishPhonetization(phoneme_dictionary)
        output = phonetization.try_convert_to_phonemes("    nonword    anothernonword   ")
        assert output == [
            GraphemeSequence("nonword"), 
            GraphemeSequence("anothernonword"), 
        ]