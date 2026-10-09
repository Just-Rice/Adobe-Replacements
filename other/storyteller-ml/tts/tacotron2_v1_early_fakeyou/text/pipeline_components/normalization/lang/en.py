import re

ZERO_DOLLARS = "zero dollars"
CENT_SINGULAR = "cent"
CENT_PLURAL = "cents"
DOLLAR_SINGULAR = "dollar"
DOLLAR_PLURAL = "dollars"
FLOAT_COMMA = "point"
EUROS_PLURAL = "euros"

_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')

_abbreviations_list = [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]
