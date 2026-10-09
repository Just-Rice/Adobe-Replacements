import re
from typing import List
from text.pipeline.base_pipeline import BasePipeline
from text.pipeline_components.normalization.common_cleaners import collapse_whitespace, convert_to_ascii, lowercase
from text.pipeline_components.normalization.english_cleaners import expand_abbreviations, expand_numbers
from text.pipeline_components.normalization.strip_and_add_ending_period import strip_and_add_ending_period
from text.pipeline_components.sequencing.english_symbols_v1 import arpabet_to_sequence, symbols_to_sequence

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

class LegacyFakeYouPipeline(BasePipeline):
    """
    This pipeline was in use from June 2021 to July 2022.
    In contrast to legacy "vo.codes" (2020 - 2021), it does not 
    leverage CMUDict to look up arpabet even when models were known to
    be trained with arpabet support. (It did support curly braces for 
    manual arpabet inclusion.)

    NB: Updated 2023-01-12 to force addition of ending periods to prevent "strokes".
    """

    def user_input_to_sequence(self, input_text: str) -> List[int]:
        """Convert raw user input into a numerical sequence for ML inference."""
        pre_sequence = self._preprocess_text(input_text)
        return self._text_to_sequence(pre_sequence, ['english_cleaners'])

    def _preprocess_text(self, input_text: str, is_raw_input: bool = True) -> str:
        # Force addition of an ending period to prevent "strokes"
        input_text = strip_and_add_ending_period(input_text)

        pre_sequence = ""
        for line in input_text.split("\n"):
            if len(line) < 1:
                continue
            if is_raw_input:
                if line[-1] != ";":
                    line = line + ";"
            else:
                pass
            pre_sequence += line
        return pre_sequence


    def _text_to_sequence(self, text, cleaner_names):
        '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

            The text can optionally have ARPAbet sequences enclosed in curly braces embedded
            in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

            Args:
            text: string to convert to a sequence
            cleaner_names: names of the cleaner functions to run the text through

            Returns:
            List of integers corresponding to the symbols in the text
        '''
        sequence = []

        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += symbols_to_sequence(english_cleaners(text))
                break
            sequence += symbols_to_sequence(english_cleaners(m.group(1)))
            sequence += arpabet_to_sequence(m.group(2))
            text = m.group(3)

        return sequence


def english_cleaners(text):
  '''Pipeline for English text, including number and abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)
  return text
