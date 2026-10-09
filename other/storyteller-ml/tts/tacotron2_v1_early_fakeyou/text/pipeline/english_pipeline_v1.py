from typing import List
from text.pipeline.base_pipeline import BasePipeline
from text.pipeline_components.input_parsing.parser import ArpabetSequence, GraphemeSequence, ParsedSequence, parse_grapheme_and_arpabet_sequence
from text.pipeline_components.normalization.english_normalization import normalize_english
from text.pipeline_components.normalization.strip_and_add_ending_period import strip_and_add_ending_period
from text.pipeline_components.phonetization.base_phonetization import BasePhonetization
from text.pipeline_components.phonetization.cmudict import DEFAULT_CMUDICT_PATH, CMUDict
from text.pipeline_components.phonetization.english_phonetization import EnglishPhonetization
from text.pipeline_components.sequencing.english_symbols_v1 import SPACE_INTEGER_ENCODING, arpabet_to_sequence, symbols_to_sequence

class EnglishPipelineV1(BasePipeline):
    """
    This is a new text pipeline for FakeYou that will be introduced in July 2022.
    It introduces Arpabet and individually upgradable components.
    The code is not 1:1 with our notebooks, but we'll go to efforts to make sure that 
    the model input expectations on both sides match one another.

    NB: Updated 2023-01-12 to force addition of ending periods to prevent "strokes".
    """

    def __init__(self, phonetization: BasePhonetization):
        self.phonetization = phonetization

    def user_input_to_sequence(self, input_text: str) -> List[int]:
        """Convert raw user input into a numerical sequence for ML inference."""

        # Force addition of an ending period to prevent "strokes"
        input_text = strip_and_add_ending_period(input_text)


        parsed_sequence = parse_grapheme_and_arpabet_sequence(input_text)
        self._normalize_graphemes(parsed_sequence)

        parsed_sequence = self._convert_phonemes(parsed_sequence)
        
        return self._convert_to_integer_sequence(parsed_sequence)

    def _normalize_graphemes(self, parsed_sequence: ParsedSequence):
        for i in range(len(parsed_sequence)):
            subsequence = parsed_sequence[i]
            if isinstance(subsequence, GraphemeSequence):
                normalized = normalize_english(subsequence.sequence)
                parsed_sequence[i] = GraphemeSequence(normalized)


    def _convert_phonemes(self, parsed_sequence: ParsedSequence) -> ParsedSequence:
        # NB: We will likely break grapheme runs into multiple arpabet sequences, extending the 
        # output list length.
        updated_sequence = []

        for subsequence in parsed_sequence:
            if isinstance(subsequence, ArpabetSequence):
                updated_sequence.append(subsequence)

            elif isinstance(subsequence, GraphemeSequence):
                converted_sequence = self.phonetization.try_convert_to_phonemes(subsequence.sequence)
                updated_sequence.extend(converted_sequence)

        return updated_sequence

    def _convert_to_integer_sequence(self, parsed_sequence: ParsedSequence) -> List[int]:
        integer_sequence = []
        last_index = len(parsed_sequence) - 1

        for i, subsequence in enumerate(parsed_sequence):
            integer_subsequence = []

            if isinstance(subsequence, ArpabetSequence):
                integer_subsequence = arpabet_to_sequence(subsequence.sequence)
            elif isinstance(subsequence, GraphemeSequence):
                integer_subsequence = symbols_to_sequence(subsequence.sequence)
                
            if integer_subsequence:
                integer_sequence.extend(integer_subsequence)

                if i < last_index and integer_sequence[-1] != SPACE_INTEGER_ENCODING:
                    # Separate words with single spaces if not already present.
                    integer_sequence.append(SPACE_INTEGER_ENCODING)

        return integer_sequence
            