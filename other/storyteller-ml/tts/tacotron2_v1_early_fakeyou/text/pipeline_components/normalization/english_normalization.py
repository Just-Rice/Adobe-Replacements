from text.pipeline_components.normalization.common_cleaners import collapse_whitespace
from text.pipeline_components.normalization.common_cleaners import convert_to_ascii
from text.pipeline_components.normalization.common_cleaners import lowercase
from text.pipeline_components.normalization.english_cleaners import expand_abbreviations, expand_numbers

def normalize_english(raw_input_text: str) -> str:
    text = convert_to_ascii(raw_input_text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)

    return text
