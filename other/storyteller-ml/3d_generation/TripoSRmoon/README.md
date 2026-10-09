## Introduction
This is an image-to-3d render commandline inference point based on the official TripoSR repo at https://github.com/VAST-AI-Research/TripoSR

### Installation
- Python >= 3.8
- Install CUDA if available
- Create environment with `conda create -n triposr python=3.11`
- Install PyTorch with `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` OTHERWISE do so according to your platform: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) **[Please make sure that the locally-installed CUDA major version matches the PyTorch-shipped CUDA major version. For example if you have CUDA 11.x installed, make sure to install PyTorch compiled with CUDA 11.x.]**
- Update setuptools by `pip install --upgrade setuptools`
- Install other dependencies by `pip install -r requirements.txt`

### Commandline Inference
```sh
python run.py examples/chair.png --output-dir output/ --bake-texture
```
This will save the reconstructed 3D model to `output/`. You can also specify more than one image path separated by spaces. The default options takes about **6GB VRAM** for a single image input.

For detailed usage of this script, use `python run.py --help`.

### Commandline Arguments
`python run.py` commandline options:
  `-h, --help` show this help message and exit
  
  `--device` DEVICE
  Device to use. If no CUDA-compatible device is found, will fallback to 'cpu'. Default: 'cuda:0'
  
  `--pretrained-model-name-or-path` PRETRAINED_MODEL_NAME_OR_PATH
Path to the pretrained model. Could be either a huggingface model id is or a local path. Default: 'stabilityai/TripoSR'

  `--chunk-size` CHUNK_SIZE
Evaluation chunk size for surface extraction and rendering. Smaller chunk size reduces VRAM usage but   
increases computation time. 0 for no chunking. Default: 8192

  `--mc-resolution` MC_RESOLUTION
Marching cubes grid resolution. Default: 256
 If specified, the background will NOT be automatically removed from the input image, and the input image should be an RGB image with gray background and properly-sized foreground. Default: false
 
  `--foreground-ratio` FOREGROUND_RATIO 
Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified.
Default: 0.85

  `--output-dir` OUTPUT_DIR
 Output directory to save the results. Default: 'output/'
 
  `--model-save-format` {obj,glb}
 Format to save the extracted mesh. Default: 'obj'
 
  `--bake-texture`
Bake a texture atlas for the extracted mesh, instead of vertex colors

  `--texture-resolution` TEXTURE_RESOLUTION
Texture atlas resolution, only useful with --bake-texture. Default: 2048

  `--render`
If specified, save a NeRF-rendered video. Default: false

## Troubleshooting
> AttributeError: module 'torchmcubes_module' has no attribute 'mcubes_cuda'

or

> torchmcubes was not compiled with CUDA support, use CPU version instead.

This is because `torchmcubes` is compiled without CUDA support. Please make sure that 

- The locally-installed CUDA major version matches the PyTorch-shipped CUDA major version. For example if you have CUDA 11.x installed, make sure to install PyTorch compiled with CUDA 11.x.
- `setuptools>=49.6.0`. If not, upgrade by `pip install --upgrade setuptools`.

Then re-install `torchmcubes` by:
```sh
pip uninstall torchmcubes
pip install git+https://github.com/tatsy/torchmcubes.git
```
