
import re
from typing import List
from text.pipeline_components.phonetization.cmudict import CMUDict

# Regex taken from open source tacotron2 extensions
# NB: This was broken and couldn't handle apostrophes
#_arpabet_split_re = re.compile(r'[^\s]*{.*?}[^\s]*|[\w]+|[^\w\s]+')

# Modified regex to not split apostrophe (eg. "you're", "I'm", etc.) or hyphenation (eg "avant-garde")
_arpabet_split_re = re.compile(r'[^\s]*{.*?}[^\s]*|[\w]+\'[\w]+|[\w]+\-[\w]+|[\w]+|[^\w\s]+')

class BasePhonetization:
    def __init__(self, phoneme_dictionary: CMUDict):
        self.phoneme_dictionary = phoneme_dictionary

    def _split_grapheme_sentence_to_words(self, text_sequence: str) -> List[str]:
        """Split a grapheme word sequence to individual grapheme words"""
        return [word for word in _arpabet_split_re.findall(text_sequence)]
