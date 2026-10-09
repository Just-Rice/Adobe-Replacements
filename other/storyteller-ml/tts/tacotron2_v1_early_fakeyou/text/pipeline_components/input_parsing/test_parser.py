from text.pipeline_components.input_parsing.parser import ArpabetSequence, GraphemeSequence, parse_grapheme_and_arpabet_sequence

class TestParser:

    def test_only_text(self):
        parsed = parse_grapheme_and_arpabet_sequence("This is a test")
        assert len(parsed) == 1
        assert isinstance(parsed[0], GraphemeSequence)

    def test_only_arpabet(self):
        parsed = parse_grapheme_and_arpabet_sequence("{T EH1 S T IH0 NG}")
        assert len(parsed) == 1
        assert isinstance(parsed[0], ArpabetSequence)

    def test_text_and_arpabet(self):
        parsed = parse_grapheme_and_arpabet_sequence("This is a {T EH1 S T}")
        assert len(parsed) == 2
        assert isinstance(parsed[0], GraphemeSequence)
        assert isinstance(parsed[1], ArpabetSequence)

    def test_text_and_arpabet_interleved_1(self):
        parsed = parse_grapheme_and_arpabet_sequence("{DH IH1 S} is a {T EH1 S T}")
        assert len(parsed) == 3
        assert isinstance(parsed[0], ArpabetSequence)
        assert isinstance(parsed[1], GraphemeSequence)
        assert isinstance(parsed[2], ArpabetSequence)

    def test_text_and_arpabet_interleved_2(self):
        parsed = parse_grapheme_and_arpabet_sequence("Check out {DH IH1 S} {T EH1 S T} of the {S IH1 S T AH0 M}")
        assert len(parsed) == 5
        assert isinstance(parsed[0], GraphemeSequence)
        assert isinstance(parsed[1], ArpabetSequence)
        assert isinstance(parsed[2], ArpabetSequence)
        assert isinstance(parsed[3], GraphemeSequence)
        assert isinstance(parsed[4], ArpabetSequence)

    def test_text_and_arpabet_interleved_2(self):
        parsed = parse_grapheme_and_arpabet_sequence("Check out {DH IH1 S} {T EH1 S T} of the {S IH1 S T AH0 M}.")
        assert len(parsed) == 6
        assert isinstance(parsed[0], GraphemeSequence)
        assert isinstance(parsed[1], ArpabetSequence)
        assert isinstance(parsed[2], ArpabetSequence)
        assert isinstance(parsed[3], GraphemeSequence)
        assert isinstance(parsed[4], ArpabetSequence)
        assert isinstance(parsed[5], GraphemeSequence)