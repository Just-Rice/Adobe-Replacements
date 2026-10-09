import sys
import torch

# TODO: De-duplicate this
def print_gpu_info():
    print('\n========================================')
    print('Python interpreter', sys.executable)
    print('PyTorch version', torch.__version__)
    print('CUDA Available?', torch.cuda.is_available())
    print('CUDA Device count', torch.cuda.device_count())
    print('CUDA architectures library was compiled for', torch.cuda.get_arch_list())

    #try:
    #    from tensorflow.python.client import device_lib
    #    print('local devices', str(device_lib.list_local_devices()).replace("\n", "\n  "))
    #except ImportError:
    #    print('no tensorflow - cannot list devices')
    #    pass
    print('========================================\n', flush=True)