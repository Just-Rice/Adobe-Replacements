""" from https://github.com/keithito/tacotron """

import re
from unidecode import unidecode

from num2words import num2words

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

class Cleaner:
  def __init__(self, lang='en'):
    self.lang=lang
    if lang == 'en':
      from text.pipeline_components.normalization.lang.en import ZERO_DOLLARS, CENT_SINGULAR, CENT_PLURAL, DOLLAR_SINGULAR, DOLLAR_PLURAL, FLOAT_COMMA,_abbreviations_list, EUROS_PLURAL, _ordinal_re
    elif lang == 'es':
      from text.pipeline_components.normalization.lang.es import ZERO_DOLLARS, CENT_SINGULAR, CENT_PLURAL, DOLLAR_SINGULAR, DOLLAR_PLURAL, FLOAT_COMMA,_abbreviations_list, EUROS_PLURAL, _ordinal_re
    else:
      raise Exception("Language not supported")
    
    # List of (regular expression, replacement) pairs for abbreviations:
    self._abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in _abbreviations_list]
    self._comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
    self._decimal_number_re = re.compile(r'([0-9]+\,[0-9]+)')
    self._euros_re = re.compile(r'€([0-9\,]*[0-9]+)')
    self._dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
    self._ordinal_re = _ordinal_re
    self._number_re = re.compile(r'[0-9]+')
    self.ZERO_DOLLARS = ZERO_DOLLARS
    self.CENT_SINGULAR = CENT_SINGULAR
    self.CENT_PLURAL = CENT_PLURAL
    self.DOLLAR_SINGULAR = DOLLAR_SINGULAR
    self.DOLLAR_PLURAL = DOLLAR_PLURAL
    self.FLOAT_COMMA = FLOAT_COMMA
    self.EUROS_PLURAL = EUROS_PLURAL

  def lowercase(self, text: str) -> str:
    return text.lower()

  def collapse_whitespace(self, text) -> str:
    return re.sub(_whitespace_re, ' ', text)

  def convert_to_ascii(self, text: str) -> str:
    return unidecode(text)

  def basic_cleaners(self, text: str) -> str:
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = Cleaner.lowercase(text)
    text = Cleaner.collapse_whitespace(text)
    return text

  def transliteration_cleaners(self, text: str) -> str:
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = Cleaner.convert_to_ascii(text)
    text = Cleaner.lowercase(text)
    text = Cleaner.collapse_whitespace(text)
    return text

  def expand_abbreviations(self, text):
    for regex, replacement in self._abbreviations:
      text = re.sub(regex, replacement, text)
    return text

  def _remove_commas(self, m):
    return m.group(1).replace('.', '')

  def _expand_decimal_point(self, m):
    return m.group(1).replace(',', f' {self.FLOAT_COMMA} ')

  def _expand_dollars(self, m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
      return match + f' {self.DOLLAR_PLURAL}'  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
      dollar_unit = self.DOLLAR_SINGULAR if dollars == 1 else self.DOLLAR_PLURAL
      cent_unit = self.CENT_SINGULAR if cents == 1 else self.CENT_PLURAL
      return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
      dollar_unit = self.DOLLAR_SINGULAR if dollars == 1 else self.DOLLAR_PLURAL
      return '%s %s' % (dollars, dollar_unit)
    elif cents:
      cent_unit = self.CENT_SINGULAR if cents == 1 else self.CENT_PLURAL
      return '%s %s' % (cents, cent_unit)
    else:
      return self.ZERO_DOLLARS

  def _expand_ordinal(self, m):
    return num2words(m.group(0), to='ordinal',lang=self.lang)

  def _expand_number(self, m):
    num = int(m.group(0))
    return num2words(num,lang=self.lang)

  def normalize_numbers(self, text):
    text = re.sub(self._comma_number_re, self._remove_commas, text)
    text = re.sub(self._euros_re, r'\1 EUROS_PLURAL', text).replace("EUROS_PLURAL", self.EUROS_PLURAL)
    text = re.sub(self._dollars_re, self._expand_dollars, text)
    text = re.sub(self._decimal_number_re, self._expand_decimal_point, text)
    text = re.sub(self._ordinal_re, self._expand_ordinal, text)
    text = re.sub(self._number_re, self._expand_number, text)
    return text

  def expand_numbers(self, text):
    return self.normalize_numbers(text)