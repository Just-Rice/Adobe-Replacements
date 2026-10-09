"""
Originally adapted from https://github.com/keithito/tacotron

This defines the set of symbols used in text input to the model.

This is the set of character encodings that FakeYou and UberDuck models have
been trained on. We freeze it here as "V1", because these may change in future
notebooks.

Original documentation: 
    The default is a set of ASCII characters that works well for English or 
    text that has been run through Unidecode. For other data, you can modify 
    _characters. See TRAINING_DATA.md for details.
"""

from typing import List

class Symbols:
    def __init__(self, lang='en'):
        if lang == 'en':
            from text.pipeline_components.phonetization.lang.es import pad, special, punctuation, letters, valid_symbols
        elif lang == 'es':
            from text.pipeline_components.phonetization.lang.es import pad, special, punctuation, letters, valid_symbols
        else:
            raise Exception("Not supported language")

        # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
        arpabet = ['@' + s for s in valid_symbols]

        # Export all symbols:
        symbols = [pad] + list(special) + list(punctuation) + list(letters) + arpabet

        # Mappings from symbol to numeric ID and vice versa:
        # The ordering **CANNOT** change.
        self._symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(symbols)}

        # Spaces can be injected when joining words.
        self.SPACE_INTEGER_ENCODING = self._symbol_to_id[' ']

    def _should_keep_symbol(self, s):
        return s in self._symbol_to_id and s != '_' and s != '~'

    def symbols_to_sequence(self, symbols: str) -> List[int]:
        return [self._symbol_to_id[s] for s in symbols if self._should_keep_symbol(s)]

    def arpabet_to_sequence(self, arpabet_string: str) -> List[int]:
        return self.symbols_to_sequence(['@' + s for s in arpabet_string.split()])

    def sequence_to_text(self, sequence):
        '''Converts a sequence of IDs back to a string'''
        result = ''
        for symbol_id in sequence:
            if symbol_id in self._id_to_symbol:
                s = self._id_to_symbol[symbol_id]
                # Enclose ARPAbet back in curly braces:
                if len(s) > 1 and s[0] == '@':
                    s = '{%s}' % s[1:]
                result += s
        return result.replace('}{', ' ')
