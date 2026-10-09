from text.pipeline_components.normalization.espeak_cleaners import Cleaner

def normalize(raw_input_text: str, lang='en') -> str:
    cleaner = Cleaner(lang)
    text = cleaner.expand_numbers(raw_input_text)
    text = cleaner.expand_abbreviations(text)
    text = cleaner.collapse_whitespace(text)
    return text
