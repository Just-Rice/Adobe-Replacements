from diffusers import StableDiffusionPipeline, AutoencoderKL
import torch
from typing import List, Tuple, Optional, Union
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
from diffusers import LCMScheduler


class StableDiffusionFE:
    """
    Wrapper around diffusers StableDiffusionPipeline, as if that wasn't easy enough.
    Supports loading LoRAs, and count-batch generation like automatic111's
    """

    def __init__(self, device: str, model_path: str = None, controlnet_path: str = None,
                 tensor_dtype: torch.dtype = torch.float32):
        """
        Initialize the StableDiffusionFE with the specified device and optionally load a model.

        Args:
        device (str): The device to run the model on (e.g., 'cuda', 'cpu').
        model_path (str, optional): Path to the .safetensors file to load the model from. Defaults to None.
        controlnet_path (str, optional): Path to ControlNet model. If not None and model_path is specified, will load
                                         the model in ControlNet mode.
        tensor_dtype (torch.dtype, optional): Only effective if model_path is passed, the type to load the model in.
                                            Defaults to torch.float32 (full precision)


        """
        self.use_auxnet = False
        self.device = device
        self.pipeline = None
        self.cfg_scale = 7.5
        self.generator = None
        self.controlnet_aux = None
        self.model_path = model_path
        self.tensor_dtype = tensor_dtype
        self.lora_path = None
        self.lcm_lora_path = None

        if model_path is not None:
            if controlnet_path is not None:
                self.make_controlnet(controlnet_path, model_path)
            else:
                self.load_model(model_path, tensor_dtype)

    def load_model(self, model_path: str, tensor_dtype: torch.dtype = torch.float32) -> None:
        """
        Load the Stable Diffusion model from a specified safetensors file path.

        Args:
        model_path (str): Path to the .safetensors file to load the model from.
        tensor_dtype (torch.dtype, optional): Only effective if model_path is passed, the type to load the model in.
                                            Defaults to torch.float32 (full precision)

        """
        # Loading the pipeline from a .safetensors file without a safety checker.
        self.pipeline = StableDiffusionPipeline.from_single_file(model_path, safety_checker=None,
                                                                 torch_dtype=tensor_dtype).to(self.device)
        # Turn off safety checker because I don't want black images. (safety_checker=None above sometimes isn't enough)
        self.pipeline.requires_safety_checker = False
        self.pipeline.safety_checker = None
        self.model_path = model_path
        self.tensor_dtype = tensor_dtype

    def manual_seed(self, seed: int) -> None:
        """
        Creates a generator with a fixed seed and stores it to be used for later inferences

        Args:
            seed (int): The seed value to use for random number generation.
        """
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def set_controlnet_aux(self, auxnet_class: object, use_auxnet: bool) -> None:
        """
        Sets the auxiliary network (e.g., OpenPose detector) for the class.

        Args:
            auxnet_class: The class object to be set as the auxiliary network.
            use_auxnet: Whether to use the aux net automatically. If not, just stores it as self.controlnet_aux
        """
        self.use_auxnet = use_auxnet
        self.controlnet_aux = auxnet_class

    def make_openpose(self, controlnet_model_name: str):
        """
        Creates an OpenPose auxiliary controlnet image pre-processor.

        The aux module is used to convert regular input pictures to OpenPose things automatically

        Args:
            controlnet_model_name (str): The name of the OpenPose model
        """
        openpose = OpenposeDetector.from_pretrained(controlnet_model_name).to(self.device)
        self.set_controlnet_aux(openpose, True)

    def make_controlnet(self, controlnet_model_name: str, model_path: str = None,
                        auto_reload_lora: bool = True) -> None:
        """
        Replaces the pipeline by loading the model with a specified ControlNet.

        Args:
            controlnet_model_name (str): The name of the ControlNet model to load.
            model_path (str, optional): Path to the .safetensors file to load the main Stable Diffusion model from.
                                        Defaults to using self.model_path (reloading the model) if None.
            auto_reload_lora (bool, optional): Automatically reload (last loaded) LoRA weights, since we are reloading the model
                                               Defaults to True, but only effective if model_path is None

        Note: If you're using OpenPose you can instantiate its aux-net to be used automatically with make_openpose
        """
        reloading = False
        if model_path is None:
            # If no model was loaded...
            if self.model_path is None:
                raise ValueError("Must have a model loaded to use make_controlnet without a new model path")

            reloading = True
            model_path = self.model_path

        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_name,
            torch_dtype=self.tensor_dtype
        ).to(self.device)

        self.pipeline = StableDiffusionControlNetPipeline.from_single_file(
            model_path,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=self.tensor_dtype
        ).to(self.device)

        # Turn off safety checker because I don't want black images. (safety_checker=None above sometimes isn't enough)
        self.pipeline.requires_safety_checker = False
        self.pipeline.safety_checker = None
        self.model_path = model_path

        if self.lora_path is not None and auto_reload_lora and reloading:
            self.load_lora_weights(self.lora_path)

        if reloading and self.lcm_lora_path is not None:
            self.load_lcm_lora(self.lcm_lora_path)


    def load_lora_weights(self, lora_path: str) -> None:
        """
        Load LoRA weights from a specified safetensors file path.

        Args:
        lora_path (str): Path to the .safetensors file to load the LoRA weights from.
        """
        if not self.pipeline:
            raise ValueError("Model must be loaded before loading LoRA weights.")

        self.lora_path = lora_path
    
        self.pipeline.load_lora_weights(lora_path,low_cpu_mem_usage=False, ignore_mismatched_sizes=True)

    def load_vae(self, vae_path: str) -> None:
        """
        Load a VAE model from the specified path and replace the existing VAE in the pipeline.

        Args:
            vae_path (str): The file path to the VAE model.
        """
        if not self.pipeline:
            raise ValueError("Pipeline must be initialized before loading a VAE.")

        vae_model = AutoencoderKL.from_single_file(vae_path)
        self.pipeline.vae = vae_model.to(self.device)  # Move the VAE to the correct device

    def generate_image(
            self,
            prompt: str,
            num_inference_steps: int,
            negative_prompt: str = None,
            dimensions: Optional[Tuple[int, int]] = None,
            image_input: Optional[Union[str, Image.Image]] = None
    ) -> Image.Image:
        """
        Generate an image based on the prompt and number of inference steps provided.
        Optionally uses an image input for ControlNet.

        Args:
            prompt (str): The prompt describing the desired output image.
            num_inference_steps (int): Number of inference steps to run the generation.
            negative_prompt (str, optional): Negative prompt.
            dimensions (tuple (int, int), optional): Width and height of output image for generation.
            image_input (Union[str, Image.Image], optional): The input image for ControlNet,
                can be a file path or a PIL.Image object.

        Returns:
            Image.Image: The generated image.
        """

        if not self.pipeline:
            raise ValueError("Model must be loaded before generating images.")

        width, height = dimensions if dimensions is not None else (None, None)

        # Process the image input if provided
        if image_input is not None:
            image_input = self.process_image_input(image_input)

            return self.pipeline(
                prompt,
                image_input,
                num_inference_steps=num_inference_steps,
                guidance_scale=self.cfg_scale,
                generator=self.generator,
                negative_prompt=negative_prompt,
                width=width,
                height=height
            ).images[0]
        else:
            # Normal image generation without ControlNet
            return self.pipeline(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=self.cfg_scale,
                generator=self.generator,
                width=width,
                height=height
            ).images[0]

    def generate_images(
            self,
            prompts: List[str],
            negative_prompts: List[str],
            num_inference_steps: int,
            dimensions: Optional[Tuple[int, int]] = None,
            image_inputs: Optional[List[Union[str, Image.Image]]] = None,
            force_no_process_img: bool = False
    ) -> List:
        """
        Generate a batch of images based on the given lists of prompts and negative prompts,
        with the specified number of inference steps for each generation.
        Optionally uses image inputs for ControlNet.

        Args:
            prompts (List[str]): The prompts describing the desired output images.
            negative_prompts (List[str]): The negative prompts to guide the generation away from undesired elements.
            num_inference_steps (int): Number of inference steps to run the generation.
            dimensions (tuple (int, int), optional): Width and height of output image for generation.
            image_inputs (List[Union[str, Image.Image]], optional): List of input images for ControlNet,
                can be file paths or PIL.Image objects.
            force_no_process_img (bool): If True, skips processing of image inputs with aux ControlNet

        Returns:
            List: A list of generated images.
        """

        if not self.pipeline:
            raise ValueError("Model must be loaded before generating images.")
        if len(prompts) != len(negative_prompts):
            raise ValueError("The number of prompts and negative prompts must be the same.")
        if image_inputs is not None and len(prompts) != len(image_inputs):
            raise ValueError("The number of prompts and image inputs must be the same.")

        width, height = dimensions if dimensions is not None else (None, None)

        # Process image inputs if provided and self.controlnet_aux is not None
        processed_images = None
        if image_inputs is not None:
            processed_images = []

            use_au_net = self.use_auxnet and self.controlnet_aux is not None
            use_controlnet_aux = use_au_net and not force_no_process_img

            for img_input in image_inputs:
                if isinstance(img_input, str):
                    img_input = load_image(img_input)

                processed_image = self.controlnet_aux(img_input) if use_controlnet_aux else img_input
                processed_images.append(processed_image)

        # Dynamic pipeline call based on whether image inputs are provided
        if processed_images is not None:
            images, _ = self.pipeline(
                prompts,
                image_input=processed_images,
                negative_prompt=negative_prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=self.cfg_scale,
                generator=self.generator,
                width=width,
                height=height,
                return_dict=False
            )
        else:
            images, _ = self.pipeline(
                prompts,
                negative_prompt=negative_prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=self.cfg_scale,
                generator=self.generator,
                width=width,
                height=height,
                return_dict=False
            )

        return images

    # Additional helper method for processing image inputs
    def process_image_input(self, img_input: Union[str, Image.Image]) -> Image.Image:
        if isinstance(img_input, str):
            img_input = load_image(img_input)

        # Just to be sure. It should be impossible for self.use_auxnet to be True while self.controlnet_aux is None,
        # but eh.
        use_au_net = self.use_auxnet and self.controlnet_aux is not None
        return self.controlnet_aux(img_input) if use_au_net else img_input

    def generate_count(
            self,
            prompt: str,
            negative_prompt: str,
            num_inference_steps: int,
            batch_size: int,
            batch_count: int,
            dimensions: Optional[Tuple[int, int]] = None,
            image_input: Optional[Union[str, Image.Image]] = None

    ) -> List:
        """
        Generate a specific number of images based on a single prompt and negative prompt,
        processing in batches of a given size.

        Args:
        prompt (str): The prompt describing the desired output image.
        negative_prompt (str): The negative prompt to guide the generation away from undesired elements.
        num_inference_steps (int): Number of inference steps to run the generation.
        batch_size (int): Number of images to generate in one batch.
        batch_count (int): Total number of images to generate.
        dimensions (tuple (int, int)): width and height of output image for generation
        image_input (Union[str, Image.Image], optional): A single input image for ControlNet,
            can be a file path or a PIL.Image object.

        Returns:
        List: A list of generated images.
        """

        if not self.pipeline:
            raise ValueError("Model must be loaded before generating images.")

        # Prepare lists to hold the full batch count of prompts and negative prompts
        full_prompts = [prompt] * batch_count
        full_negative_prompts = [negative_prompt] * batch_count

        # Process the single image input once if provided
        processed_image = self.process_image_input(image_input) if image_input is not None else None

        all_images = []

        # Process in batches
        for i in range(0, batch_count, batch_size):
            # Determine the size of the current batch
            current_batch_size = min(batch_size, batch_count - i)
            batch_prompts = full_prompts[i:i + current_batch_size]
            batch_negative_prompts = full_negative_prompts[i:i + current_batch_size]

            # Generate images for the current batch
            batch_images = self.generate_images(
                prompts=batch_prompts,
                negative_prompts=batch_negative_prompts,
                num_inference_steps=num_inference_steps,
                dimensions=dimensions,
                image_inputs=[processed_image] * current_batch_size if processed_image is not None else None,
                force_no_process_img=True,
            )
            all_images.extend(batch_images)

        return all_images

    def get_compatible_schedulers(self) -> List[str]:
        """
        Returns a list of compatible scheduler class names as strings,
        formatted without the 'diffusers.schedulers.scheduling_' prefix.

        Returns:
            List[str]: A list of scheduler names.
        """
        if not self.pipeline:
            raise ValueError("Pipeline must be loaded before getting schedulers.")

        # Extract class names and format them
        scheduler_names = [
            cls.__name__.replace('Scheduling', '').replace('_', '')
            for cls in self.pipeline.scheduler.compatibles
        ]
        return scheduler_names

    def replace_scheduler(self, scheduler_name: str) -> None:
        """
        Instantiates and replaces the scheduler in the pipeline with the given scheduler name.

        Args:
            scheduler_name (str): The name of the scheduler to instantiate.
        """
        if not self.pipeline:
            raise ValueError("Pipeline must be loaded before replacing scheduler.")

        # Mapping of scheduler name to the class in diffusers
        schedulers_mapping = {
            cls.__name__.replace('Scheduling', '').replace('_', ''): cls
            for cls in self.pipeline.scheduler.compatibles
        }

        # Find the scheduler class from the given scheduler name
        scheduler_class = schedulers_mapping.get(scheduler_name)
        if not scheduler_class:
            raise ValueError(f"Scheduler '{scheduler_name}' is not recognized as a compatible scheduler.")

        # Instantiate and replace the scheduler in the pipeline
        self.pipeline.scheduler = scheduler_class.from_config(self.pipeline.scheduler.config)

    def load_lcm_lora(self, adapter_id: str) -> None:
        """
        Loads a Latent Consistency Model (LCM) with LoRA, allowing generation in fewer steps.

        Args:
            adapter_id (str): The identifier of the LCM LoRA adapter.
        """
        if not self.pipeline:
            raise ValueError("Pipeline must be loaded before loading LCM LoRA.")

        # Update the scheduler to LCMScheduler
        self.pipeline.scheduler = LCMScheduler.from_config(self.pipeline.scheduler.config)

        # Load and fuse LCM LoRA
        self.pipeline.load_lora_weights(adapter_id)
        self.pipeline.fuse_lora()

        # https://huggingface.co/latent-consistency/lcm-lora-sdv1-5
        # >Please make sure to either disable guidance_scale or use values between 1.0 and 2.0.
        self.cfg_scale = 0
        self.lcm_lora_path = adapter_id