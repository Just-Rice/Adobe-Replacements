
# If a sentence doesn't end in expected ending punctuation, we'll want to auto-punctuate.
EXPECTED_ENDING_PUNCTUATIONS = [
    '!',
    '.', 
    '?',
    '…',
    '\'',
    '\"',
]

PERIOD = '.'

# Prevent model from having a "stroke" by adding missing ending punctuation.
def strip_and_add_ending_period(input_text: str) -> str:
    # TODO: This isn't  great for all languages.
    input_text = input_text.strip()
    ends_with_period = any([input_text.endswith(e) for e in EXPECTED_ENDING_PUNCTUATIONS])
    if not ends_with_period: 
        input_text += PERIOD
    return input_text
