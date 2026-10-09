# The model dict is used for webUI only
from typing import Dict
import unittest
import pathlib


def path_to_dict(path: pathlib.Path) -> Dict[str, str]:
    path = pathlib.Path(path)
    result = {}
    for file in path.iterdir():
        if file.is_file() and file.suffix == '.safetensors':
            # Convert file name to camel case with underscores between words
            name = '_'.join(word.capitalize() for word in file.stem.split('_'))
            # Add entry to dictionary
            result[name] = str(file.relative_to(path.parent))
    result['Stable Diffusion 1.5'] = ''
    print("result: ", result)
    return result


# model_dict = {
#     'Stable Diffusion 1.5': '',
#     'revAnimated_v11': 'models/revAnimated_v11.safetensors',
#     'realisticVisionV20_v20': 'models/realisticVisionV20_v20.safetensors',
#     'CounterfeitV30_25': 'models/CounterfeitV30_25.safetensors',
#     'ghibli_style_offset': 'models/ghibli_style_offset.safetensors'
# }
model_dict = path_to_dict("./models")

loRA_dict = {
    'None': '',
    'Dark Sushi': 'loRAs/dark_sushi'
}

class TestPathToDict(unittest.TestCase):
    def test_path_to_dict(self):
        # Create a temporary directory
        result = path_to_dict(pathlib.Path("/home/tensor/code/storyteller/storyteller-ml/animation/Rerender_A_Video/models"))
        print(result)

if __name__ == '__main__':
    unittest.main()