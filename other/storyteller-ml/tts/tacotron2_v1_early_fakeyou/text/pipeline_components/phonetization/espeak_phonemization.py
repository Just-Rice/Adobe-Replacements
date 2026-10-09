import re

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

from typing import List
from text.pipeline_components.input_parsing.parser import ArpabetSequence, GraphemeSequence, ParsedSequence
from text.pipeline_components.phonetization.base_phonetization import BasePhonetization
from text.pipeline_components.input_parsing.parser import parse_grapheme_and_arpabet_sequence
# IPA to ARPABET

basic_replacements = {
  'aɪ': 'AA0 Y',
  'aʊ': 'AA0 W',
  'b': 'B',
  'd': 'D',
  'dʒ': 'JH',
  'eɪ': 'EY0',
  'f': 'F',
  'h': 'HH',
  'i': 'IY0',
  'j': 'Y',
  'k': 'K',
  'l': 'L',
  'l̩': 'EL',
  'm': 'M',
  'm̩': 'EM',
  'n': 'N',
  'n̩': 'EN',
  'oʊ': 'OW',
  'p': 'P',
  's': 'S',
  't': 'T',
  'tʃ': 'CH',
  'u': 'UW0',
  'v': 'V',
  'w': 'W',
  'z': 'Z',
  'æ': 'AE0',
  'ð': 'DH',
  'ŋ': 'NG',
  'ɑ': 'AA0',
  'ɔ': 'AO0',
  'ɔɪ': 'OY0',
  'ə': 'AX',
  'ɚ': 'AXR',
  'ɛ': 'EH0',
  'ɝ': 'ER',
  'ɡ': 'G',
  'ɨ': 'IX',
  'ɪ': 'IH0',
  'ɹ': 'R',
  'ɾ': 'R',
  'ɾ̃': 'NX',
  'ʃ': 'SH',
  'ʉ': 'UX',
  'ʊ': 'UH0',
  'ʌ': 'AH0',
  'ʍ': 'WH',
  'ʒ': 'ZH',
  'ʔ': 'Q',
  'θ': 'TH'
}

class EspeakPhonetization(BasePhonetization):

    def __init__(self, lang="en"):
        super().__init__(None)
        if lang == 'en':
            from text.pipeline_components.phonetization.lang.es import valid_symbols, extra_replacements
        elif lang == 'es':
            from text.pipeline_components.phonetization.lang.es import valid_symbols, extra_replacements
        else:
            raise Exception("Not supported language")


        self.replacements = {**basic_replacements, **extra_replacements}
        self.separator = Separator(phone='|', word=None)

        lang2espeaklang = {"en": "en-us", "es": "es", "fr": "fr-fr"}
        self.backend = EspeakBackend(lang2espeaklang[lang],with_stress=True, preserve_punctuation=False)

    def grapheme2ipa(self, word):
        return self.backend.phonemize([word],separator=self.separator)[0].strip().strip("|")

    def ipa2arpabet(self, ipa_word):
        # print(ipa_word)
        ipa_word = ipa_word.replace("ɾ|ɾ", "r")
        if ipa_word.endswith("j|j"):
            ipa_word = ipa_word.replace("j|j", "i")
        else:
            ipa_word = ipa_word.replace("j|j", "ʎ")
        # Remove secondary accents, they are not important in spanish
        ipa_word = ipa_word.replace("ˌ", "")
        # Remove long consonant indication
        ipa_word = ipa_word.replace("ː", "")
        arpabet_word = ""
        for c in ipa_word.split("|"):
            accent = False
            if c.startswith("ˈ"):
                c = c[1:]
                accent = True
            if c in self.replacements:
                replacement = self.replacements[c]
                if accent:
                    replacement = replacement.replace("0", "1")
                arpabet_word += replacement + " "
            else:
                print(ipa_word)
                print(f"'{c}' is not in the replacements dictionary")
                raise Exception()
        return "{" + arpabet_word.strip() + "}"

    def word2phonemes(self, word):
        return self.ipa2arpabet(self.grapheme2ipa(word))

    def __len__(self):
        return len(self._entries)

    def try_convert_to_phonemes(self, grapheme_sequence: str) -> ParsedSequence:
        '''Converts word to spanish adapted ARPAbet'''

        # Apply it only on words, keep punctuation unchanged
        for match in re.findall(r"[A-Za-zÀ-ÿ]+", grapheme_sequence):
            grapheme_sequence = grapheme_sequence.replace(match, self.word2phonemes(match), 1)

        return parse_grapheme_and_arpabet_sequence(grapheme_sequence)