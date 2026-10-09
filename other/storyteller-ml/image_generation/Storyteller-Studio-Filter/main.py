import argparse
import os
import pathlib
import torch
import sys

# Note please use type hints.
from stablediffusionfe import StableDiffusionFE
import random
import string

def print_gpu_info():
    print('========================================')
    print('Python Interpreter', sys.executable)
    print('Python Version', sys.version)
    print('Python Version Info', sys.version_info)
    print('PyTorch Version', torch.__version__)
    print('CUDA Available?', torch.cuda.is_available())
    print('CUDA Version', torch.version.cuda)
    print('CUDA Device Count', torch.cuda.device_count())
    print('CUDA architectures library was compiled for', torch.cuda.get_arch_list())
    print('========================================', flush=True)

print_gpu_info()

def generate_random_string(length=6):
    # Choose from uppercase, lowercase letters and digits
    characters = string.ascii_letters + string.digits
    # Use random.choices to generate a list of random characters, then join them into a string
    random_string = ''.join(random.choices(characters, k=length))
    return random_string

class ModelInfo:
    def __init__(self,path:pathlib.Path):
        self.version = 1.5
        self.path = path
        
class StableDiffusion(ModelInfo):
    def __init__(self, path: pathlib.Path):
        super().__init__(path)
        
class loRA(ModelInfo):
    def __init__(self, path: pathlib.Path):
        super().__init__(path)

class VAE(ModelInfo):
    def __init__(self, path: pathlib.Path):
        super().__init__(path)


_au111_to_diffusers_samplers = {'DPM++ 2M Karras': 'DPMSolverMultistepScheduler',
     'DPM++ 2M SDE Exponential': 'DPMSolverSDEScheduler',
     'DPM++ 2M SDE Karras': 'DPMSolverSDEScheduler',
     'Euler a': 'EulerAncestralDiscreteScheduler',
     'Euler': 'EulerDiscreteScheduler',
     'LMS': 'LMSDiscreteScheduler',
     'Heun': 'HeunDiscreteScheduler',
     'DPM2': 'KDPM2DiscreteScheduler',
     'DPM2 a': 'KDPM2AncestralDiscreteScheduler',
     'DPM++ 2S a': 'DDPMScheduler',
     'DPM++ 2M': 'DPMSolverMultistepScheduler',
     'DPM++ 2M SDE': 'DPMSolverSDEScheduler',
     'DPM++ 2M SDE Heun': 'DPMSolverSDEScheduler',
     'DPM++ 2M SDE Heun Karras': 'DPMSolverSDEScheduler',
     'DPM++ 2M SDE Heun Exponential': 'DPMSolverSDEScheduler',
     'DPM++ 3M SDE': 'UniPCMultistepScheduler',
     'DPM++ 3M SDE Karras': 'UniPCMultistepScheduler',
     'DPM++ 3M SDE Exponential': 'UniPCMultistepScheduler',
     'DPM fast': 'DEISMultistepScheduler',
     'DPM adaptive': 'DDIMScheduler',
     'LMS Karras': 'LMSDiscreteScheduler',
     'DPM2 Karras': 'KDPM2DiscreteScheduler',
     'DPM2 a Karras': 'KDPM2AncestralDiscreteScheduler',
     'DPM++ 2S a Karras': 'DDPMScheduler'}


_available_samplers = ['DDIMScheduler', 'DPMSolverSDEScheduler', 'DDPMScheduler',
       'EulerDiscreteScheduler', 'DPMSolverMultistepScheduler',
       'DEISMultistepScheduler', 'LMSDiscreteScheduler', 'PNDMScheduler',
       'HeunDiscreteScheduler', 'KDPM2AncestralDiscreteScheduler', 'DPMSolverSinglestepScheduler',
       'KDPM2DiscreteScheduler', 'UniPCMultistepScheduler', 'EulerAncestralDiscreteScheduler']

class Sampler:
    samplers_k_diffusion_dict = {
    'DPM++ 2M Karras': ('DPM++ 2M Karras', 'sample_dpmpp_2m', ['k_dpmpp_2m_ka'], {'scheduler': 'karras'}),
    'DPM++ SDE Karras': ('DPM++ SDE Karras', 'sample_dpmpp_sde', ['k_dpmpp_sde_ka'], {'scheduler': 'karras', "second_order": True, "brownian_noise": True}),
    'DPM++ 2M SDE Exponential': ('DPM++ 2M SDE Exponential', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_exp'], {'scheduler': 'exponential', "brownian_noise": True}),
    'DPM++ 2M SDE Karras': ('DPM++ 2M SDE Karras', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_ka'], {'scheduler': 'karras', "brownian_noise": True}),
    'Euler a': ('Euler a', 'sample_euler_ancestral', ['k_euler_a', 'k_euler_ancestral'], {"uses_ensd": True}),
    'Euler': ('Euler', 'sample_euler', ['k_euler'], {}),
    'LMS': ('LMS', 'sample_lms', ['k_lms'], {}),
    'Heun': ('Heun', 'sample_heun', ['k_heun'], {"second_order": True}),
    'DPM2': ('DPM2', 'sample_dpm_2', ['k_dpm_2'], {'discard_next_to_last_sigma': True, "second_order": True}),
    'DPM2 a': ('DPM2 a', 'sample_dpm_2_ancestral', ['k_dpm_2_a'], {'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    'DPM++ 2S a': ('DPM++ 2S a', 'sample_dpmpp_2s_ancestral', ['k_dpmpp_2s_a'], {"uses_ensd": True, "second_order": True}),
    'DPM++ 2M': ('DPM++ 2M', 'sample_dpmpp_2m', ['k_dpmpp_2m'], {}),
    'DPM++ SDE': ('DPM++ SDE', 'sample_dpmpp_sde', ['k_dpmpp_sde'], {"second_order": True, "brownian_noise": True}),
    'DPM++ 2M SDE': ('DPM++ 2M SDE', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_ka'], {"brownian_noise": True}),
    'DPM++ 2M SDE Heun': ('DPM++ 2M SDE Heun', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_heun'], {"brownian_noise": True, "solver_type": "heun"}),
    'DPM++ 2M SDE Heun Karras': ('DPM++ 2M SDE Heun Karras', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_heun_ka'], {'scheduler': 'karras', "brownian_noise": True, "solver_type": "heun"}),
    'DPM++ 2M SDE Heun Exponential': ('DPM++ 2M SDE Heun Exponential', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_heun_exp'], {'scheduler': 'exponential', "brownian_noise": True, "solver_type": "heun"}),
    'DPM++ 3M SDE': ('DPM++ 3M SDE', 'sample_dpmpp_3m_sde', ['k_dpmpp_3m_sde'], {'discard_next_to_last_sigma': True, "brownian_noise": True}),
    'DPM++ 3M SDE Karras': ('DPM++ 3M SDE Karras', 'sample_dpmpp_3m_sde', ['k_dpmpp_3m_sde_ka'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "brownian_noise": True}),
    'DPM++ 3M SDE Exponential': ('DPM++ 3M SDE Exponential', 'sample_dpmpp_3m_sde', ['k_dpmpp_3m_sde_exp'], {'scheduler': 'exponential', 'discard_next_to_last_sigma': True, "brownian_noise": True}),
    'DPM fast': ('DPM fast', 'sample_dpm_fast', ['k_dpm_fast'], {"uses_ensd": True}),
    'DPM adaptive': ('DPM adaptive', 'sample_dpm_adaptive', ['k_dpm_ad'], {"uses_ensd": True}),
    'LMS Karras': ('LMS Karras', 'sample_lms', ['k_lms_ka'], {'scheduler': 'karras'}),
    'DPM2 Karras': ('DPM2 Karras', 'sample_dpm_2', ['k_dpm_2_ka'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    'DPM2 a Karras': ('DPM2 a Karras', 'sample_dpm_2_ancestral', ['k_dpm_2_a_ka'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    'DPM++ 2S a Karras': ('DPM++ 2S a Karras', 'sample_dpmpp_2s_ancestral', ['k_dpmpp_2s_a_ka'], {'scheduler': 'karras', "uses_ensd": True, "second_order": True})
    }
    def __init__(self):
        pass

class InputCheckers:
    @staticmethod
    def check_power_of_2(value:int):
        value = int(value)
        if value < 256 or value > 4096:
            raise argparse.ArgumentTypeError(f"Value {value} not in range [256, 4096]")
        if (value & (value - 1)) != 0:
            raise argparse.ArgumentTypeError(f"Value {value} is not a power of 2")
        return value
    
    @staticmethod
    def check_cfg_scale(value:int):
        value = int(value)
        if value < 1 or value > 30:
            raise argparse.ArgumentTypeError(f"Value {value} not in range [1, 30]")
        return value
    
    @staticmethod
    def check_sampler(value:str, dictionary:[str,tuple]):
        return value in dictionary
    
    @staticmethod
    def valid_path(path:pathlib.Path):
        if not os.path.exists(path):
            return path # TODO remove this later.
            raise argparse.ArgumentTypeError(f"The path '{path}' does not exist.")
        return path

def main():

    parser = argparse.ArgumentParser(description="CLI for various inputs.")
    parser.add_argument("--prompt", type=str, required=True, help="Text input called prompt")
    parser.add_argument("--negative-prompt", type=str, required=True, help="Text input called negative prompt")
    parser.add_argument("--number-of-samples",type=int, required=True)
    #parser.add_argument("--sampler", type=str, required=False, help="Text input called samplers", choices=_available_samplers)
    parser.add_argument("--samplers", type=str, required=False, help="Text input called samplers", choices=list(_au111_to_diffusers_samplers.keys()))
       
    # # ZDisket: We shoud really keep these static and instead use an upsampler afterwards instead
    parser.add_argument("--width", type=int, required=False, help="Width value (power of 2, between 256 and 4096)")
    parser.add_argument("--height", type=int,required=False, help="Height value (power of 2, between 256 and 4096)")
    
    parser.add_argument("--cfg-scale",type=float,required=True, help="Integer input called CFG scale (range from 1-30 can be .5)")
    parser.add_argument("--seed", type=int, required=False, help="Seed value (integer of any value)")
    parser.add_argument("--loRA-path", type=InputCheckers.valid_path, required=False, help="Path object to the loRA")
    parser.add_argument("--check-point", type=InputCheckers.valid_path, required=True, help="Path object to the Check Point")
    parser.add_argument("--vae", type=InputCheckers.valid_path, required=False, help="Path to the Variational auto encoder")
    parser.add_argument("--openpose", type=InputCheckers.valid_path, required=False,
                        help="Path to the OpenPose ControlNet")
    parser.add_argument("--openpose-aux", type=InputCheckers.valid_path, required=False,
                        help="Path to the OpenPose auxiliary to the ControlNet (image->pose)")
    parser.add_argument("--lcm", type=InputCheckers.valid_path, required=False,
                        help="Path to LCM LoRA")
    parser.add_argument("--image-input", type=InputCheckers.valid_path, required=False,
                        help="Input image when using ControlNet")
    parser.add_argument("--batch-size",type=int,required=True, help="pick 1 by default")
    parser.add_argument("--batch-count", type=int, required=True, help="pick 1 by default")

    parser.add_argument("--output-name", type=str, required=False, help="base name of output files. if not specified will be random string")
    parser.add_argument("--output-dir", type=str, required=False,
                        help="output directory. default is outputs in working dir",
                        default="outputs")

    args = parser.parse_args()

    # Logic here to process the arguments as needed
    print(f"prompt: {args.prompt}")
    print(f"negative_prompt: {args.negative_prompt}")
    print(f"number of samples: {args.number_of_samples}")
    print(f"samplers: {args.samplers}")
    print(f"width: {args.width}")
    print(f"height: {args.height}")
    print(f"cfg_scale: {args.cfg_scale}")
    print(f"seed: {args.seed}")
    print(f"loRA_path: {args.loRA_path}")
    print(f"check_point: {args.check_point}")
    print(f"vae: {args.vae}")
    print(f"batch_size: {args.batch_size}")
    print(f"batch_count: {args.batch_count}")

    ####### Load model ###########################
    # Set the device to CUDA if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the StableDiffusionFE with the chosen device
    sd_fe = StableDiffusionFE(device=device)

    # Load the model
    model_path = args.check_point
    sd_fe.load_model(model_path=model_path)

    # Set CFG scale
    sd_fe.cfg_scale = args.cfg_scale

    # Load LoRA
    if args.loRA_path is not None:
        lora_path = args.loRA_path
        sd_fe.load_lora_weights(lora_path=lora_path)

    # Replace sampler if specified
    if args.samplers is not None:
        sd_fe.replace_scheduler(_au111_to_diffusers_samplers[args.samplers])
        
    if args.seed is not None:
        sd_fe.manual_seed(args.seed)

    # Load VAE if specified (not tested)
    if args.vae is not None:
        sd_fe.load_vae(vae_path=args.vae)

    # Load LCM LoRA if specified
    if args.lcm is not None:
        sd_fe.load_lcm_lora(args.lcm)

    # Load OpenPose controlnet if specified
    if args.openpose is not None:
        openpose_auxnet_name = args.openpose_aux
        openpose_controlnet_name = args.openpose

        # Infuse the model with ControlNet abilities.
        sd_fe.make_controlnet(openpose_controlnet_name)
        # The make_controlnet function replaces the pipeline and reloads the last loaded LoRA

        # Add the OpenPose auxiliary net that automatically converts poses in regular images to OpenPose things. For other
        # ControlNets, you'll have to manually instantiate its auxnet and use set_controlnet_aux
        if openpose_auxnet_name is not None:
            sd_fe.make_openpose(openpose_auxnet_name)

    # Set width and height to dimensions. If any or both are None, it will result in the pipeline doing its default.
    dimensions = (args.width, args.height)


    ####### Load model ###########################

    ### Generate images############


    generated_images = sd_fe.generate_count(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.number_of_samples,
        batch_size=args.batch_size,
        batch_count=args.batch_count,
        dimensions=dimensions,
        image_input=args.image_input,
    )

    output_path = "outputs" if args.output_dir is None else args.output_dir
    print(f"Generated {len(generated_images)} images. Saving to {output_path}")
    #### Save ###########

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    filename_base = generate_random_string(6) if args.output_name is None else args.output_name
    for i, img in enumerate(generated_images):
        img_raw_filename = f"{filename_base}_{i}.png"
        img.save(
            os.path.join(output_path,
                         img_raw_filename)
        )

    print("Done!")


if __name__ == '__main__':
    main()
    
    
