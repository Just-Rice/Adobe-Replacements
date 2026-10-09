from phonemizer import phonemize
from phonemizer.separator import Separator
import os

def phonemize_files(file_paths, language='en-us', backend='espeak', output_prefix='phonemized_'):
    """
    Phonemizes the text in each of the specified files and writes the results to new files.

    Parameters:
    - file_paths: A list of strings, where each string is a path to a file to be processed.
    - language: The language code to use for phonemization.
    - backend: The phonemization backend to use.
    - output_prefix: Prefix for the output file names.

    Returns:
    A list of paths to the output files containing the phonemized text.
    """
    output_files = []

    def phonemize_text(text):
        return phonemize(
            text,
            language=language,
            backend=backend,
            preserve_punctuation=True,
            strip=True)

    for input_file in file_paths:
        base_name = os.path.basename(input_file)
        output_file = f'{output_prefix}{base_name}'
        output_files.append(output_file)

        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                path_to_wav, text, speaker_id = line.strip().split('|')
                phonemized_text = phonemize_text(text)
                outfile.write(f'{path_to_wav}|{phonemized_text}|{speaker_id}\n')

    return output_files


def main():
    file_paths = ['Data/OOD_harness.txt', 'Data/train_harness.txt','Data/val_harness.txt']  # Replace these with your actual file paths
    output_files = phonemize_files(file_paths)
    print("Output files:", output_files)

if __name__ == "__main__":
    main()