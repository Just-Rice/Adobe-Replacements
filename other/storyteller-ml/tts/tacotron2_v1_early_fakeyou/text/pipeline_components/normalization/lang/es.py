import re

ZERO_DOLLARS = "cero dólares"
CENT_SINGULAR = "centavo"
CENT_PLURAL = "centavos"
DOLLAR_SINGULAR = "dólar"
DOLLAR_PLURAL = "dólares"
FLOAT_COMMA = "coma"
EUROS_PLURAL = "euros"

_ordinal_re = re.compile(r'[0-9]+(\º|\ª|er|do|os|as)')

_abbreviations_list = [
  ('srs', 'señores'),
  ('sr', 'señor'),
  ('dr', 'doctor'),
  ('drs', 'doctores'),
  ('jr', 'yunior'),
]

