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

from text.pipeline_components.phonetization import cmudict
from typing import List

# NB: DO NOT CHANGE THE ORDER OF SYMBOLS! 
# Models are trained on and depend upon the order.
_pad        = '_'
_punctuation = '!\'(),.:;? ' #NB: Includes space
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet

# Mappings from symbol to numeric ID and vice versa:
# The ordering **CANNOT** change.
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Spaces can be injected when joining words.
SPACE_INTEGER_ENCODING = _symbol_to_id[' ']


def symbols_to_sequence(symbols: str) -> List[int]:
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def arpabet_to_sequence(arpabet_string: str) -> List[int]:
    return symbols_to_sequence(['@' + s for s in arpabet_string.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != '_' and s != '~'
