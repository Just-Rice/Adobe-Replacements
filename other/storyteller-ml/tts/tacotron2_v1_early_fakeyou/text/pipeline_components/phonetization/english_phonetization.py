from typing import List
from text.pipeline_components.input_parsing.parser import ArpabetSequence, GraphemeSequence, ParsedSequence
from text.pipeline_components.phonetization.base_phonetization import BasePhonetization
from text.pipeline_components.phonetization.cmudict import CMUDict

class EnglishPhonetization(BasePhonetization):

    def __init__(self, phoneme_dictionary: CMUDict):
        super().__init__(phoneme_dictionary)

    def try_convert_to_phonemes(self, grapheme_sequence: str) -> ParsedSequence:
        """
        Attempt to enrich a grapheme sequence with phonemes where word lookup succeeds.
        Grapheme symbols will be left in place.
        The output list will need to be space-separated before inference.
        """
        words = self._split_grapheme_sentence_to_words(grapheme_sequence)
        converted_words = []

        for word in words:
            maybe_phonemes = self.phoneme_dictionary.lookup(word)
            if maybe_phonemes:
                # NB: Dictionary returns a list of possible matches. 
                # We'll just use the first.
                converted_words.append(ArpabetSequence(maybe_phonemes[0]))
            else:
                # TODO: Use g2p prediction to predict the phonemes.
                converted_words.append(GraphemeSequence(word))

        return converted_words
