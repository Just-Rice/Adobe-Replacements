"""
Copied or adapted from Keith Ito's Tacotron2 code.
https://github.com/keithito/tacotron
"""

import re
from unidecode import unidecode

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

def lowercase(text: str) -> str:
  return text.lower()

def collapse_whitespace(text) -> str:
  return re.sub(_whitespace_re, ' ', text)

def convert_to_ascii(text: str) -> str:
  return unidecode(text)

def basic_cleaners(text: str) -> str:
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text

def transliteration_cleaners(text: str) -> str:
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text
