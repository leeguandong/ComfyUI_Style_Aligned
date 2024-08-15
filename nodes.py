import ast

import cv2
import torch
import numpy as np

from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
    LCMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    PNDMScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLPipeline
)
from transformers import DPTImageProcessor, DPTForDepthEstimation
from .pipeline_calls import get_depth_map, controlnet_call
from .sa_handler import StyleAlignedArgs, Handler
from .inversion import ddim_inversion, make_inversion_callback
import folder_paths
import comfy.model_management as mm


def process_input(input_string):
    # 第一种格式的处理
    if '\n' in input_string and '"' in input_string:
        # 删除多余的引号和换行符，并分割成列表
        cleaned_string = input_string.replace('"', '').replace(',', '')
        result_list = cleaned_string.split('\n')
    # 第二种格式的处理
    elif '"' in input_string:
        result_list = [input_string.replace('"', '')]
    # 第三种格式的处理
    elif '，\n' in input_string:
        result_list = input_string.split('，\n')
    # 当输入不符合任何已知模式时，引发异常
    else:
        return input_string

    # 除去列表中的空白项并去除每个字符串的前后空格
    result_list = [item.strip() for item in result_list if item.strip()]
    return result_list


def convert_preview_image(images):
    # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
    images_tensors = []
    for img in images:
        # 将 PIL.Image 转换为 numpy.ndarray
        img_array = np.array(img)
        # 转换 numpy.ndarray 为 torch.Tensor
        img_tensor = torch.from_numpy(img_array).float() / 255.
        # 转换图像格式为 CHW (如果需要)
        if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        # 添加批次维度并转换为 NHWC
        img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
        images_tensors.append(img_tensor)

    if len(images_tensors) > 1:
        output_image = torch.cat(images_tensors, dim=0)
    else:
        output_image = images_tensors[0]
    return output_image


class SchedulerLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scheduler": (
                    [
                        'DPMSolverMultistepScheduler',
                        'DPMSolverMultistepScheduler_SDE_karras',
                        'DDPMScheduler',
                        'DDIMScheduler',
                        'LCMScheduler',
                        'PNDMScheduler',
                        'DEISMultistepScheduler',
                        'EulerDiscreteScheduler',
                        'EulerAncestralDiscreteScheduler'
                    ], {
                        "default": 'DDIMScheduler'
                    }),
                "beta_start": (
                    "FLOAT", {"default": 0.00085, "min": 0.0, "max": 0.001, "step": 0.00005, "display": "slider"}),
                "beta_end": ("FLOAT", {"default": 0.012, "min": 0.0, "max": 0.1, "step": 0.01, "display": "slider"}),
                "beta_schedule": (["scaled_linear"], {"default": "scaled_linear"}),
                "clip_sample": ("BOOLEAN", {"default": False}),
                "set_alpha_to_one": ("BOOLEAN", {"default": False})
            },
        }

    RETURN_TYPES = ("SCHEDULER",)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "load_scheduler"
    CATEGORY = "StyleAligned"

    def load_scheduler(self,
                       scheduler,
                       beta_start,
                       beta_end,
                       beta_schedule,
                       clip_sample,
                       set_alpha_to_one):
        scheduler_config = {
            'beta_start': beta_start,
            'beta_end': beta_end,
            'beta_schedule': beta_schedule,
            'clip_sample': clip_sample,
            'set_alpha_to_one': set_alpha_to_one
        }
        if scheduler == 'DPMSolverMultistepScheduler':
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == 'DDIMScheduler':
            noise_scheduler = DDIMScheduler(**scheduler_config)
        elif scheduler == 'DPMSolverMultistepScheduler_SDE_karras':
            scheduler_config.update({"algorithm_type": "sde-dpmsolver++"})
            scheduler_config.update({"use_karras_sigmas": True})
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == 'DDPMScheduler':
            noise_scheduler = DDPMScheduler(**scheduler_config)
        elif scheduler == 'LCMScheduler':
            noise_scheduler = LCMScheduler(**scheduler_config)
        elif scheduler == 'PNDMScheduler':
            scheduler_config.update({"set_alpha_to_one": False})
            scheduler_config.update({"trained_betas": None})
            noise_scheduler = PNDMScheduler(**scheduler_config)
        elif scheduler == 'DEISMultistepScheduler':
            noise_scheduler = DEISMultistepScheduler(**scheduler_config)
        elif scheduler == 'EulerDiscreteScheduler':
            noise_scheduler = EulerDiscreteScheduler(**scheduler_config)
        elif scheduler == 'EulerAncestralDiscreteScheduler':
            noise_scheduler = EulerAncestralDiscreteScheduler(**scheduler_config)
        else:
            raise TypeError(f"not support {scheduler}!!!")

        return (noise_scheduler,)


class SASDXL_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    ["stabilityai/stable-diffusion-xl-base-1.0"],
                    {"default": "stabilityai/stable-diffusion-xl-base-1.0"},
                ),
                "scheduler": ("SCHEDULER",)
            },
            "optional": {
                "variant": ("STRING", {"default": "fp16"}),
                "use_safetensors": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "StyleAligned"

    def load_model(self, model, scheduler, variant, use_safetensors):
        device = mm.get_torch_device()
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16,
            variant=variant,
            use_safetensors=use_safetensors,
            scheduler=scheduler).to(device)
        return (pipeline,)


class SAControlnet_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ([
                              "lllyasviel/sd-controlnet-canny",
                              "diffusers/controlnet-depth-sdxl-1.0"
                          ],
                          {"default": "diffusers/controlnet-depth-sdxl-1.0"}),
                "vae_model": (["madebyollin/sdxl-vae-fp16-fix"], {"default": "madebyollin/sdxl-vae-fp16-fix"}),
                "variant": ("STRING", {"default": "fp16"}),
                "use_safetensors": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_controlnet"
    CATEGORY = "StyleAligned"

    def load_controlnet(self, model, vae_model, variant, use_safetensors):
        device = mm.get_torch_device()
        vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=torch.float16).to(device)
        controlnet = ControlNetModel.from_pretrained(
            model,
            vae=vae,
            variant=variant,
            use_safetensors=use_safetensors).to(device)
        controlnet.enable_model_cpu_offload()
        return (controlnet,)


class SADepth:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    ["Intel/dpt-hybrid-midas"],
                    {"default": "Intel/dpt-hybrid-midas"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "depth"
    CATEGORY = "StyleAligned"

    def depth(self, image, model):
        device = mm.get_torch_device()
        depth_estimator = DPTForDepthEstimation.from_pretrained(model).to(device)
        feature_processor = DPTImageProcessor.from_pretrained(model)
        depth_image = get_depth_map(image, feature_processor, depth_estimator)
        return (depth_image,)


class SAInversion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL"),
                "prompt": ("STRING", {"default": "a photo of an astronaut riding a horse on mars"}),
                "size": ("INT", {"default": 1024, "min": 512, "max": 1024}),
                "num_inference_steps": ("INT", {"default": 50, "min": 20, "max": 50}),
            }
        }

    RETURN_TYPES = ("LATENTS",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "inversion"
    CATEGORY = "StyleAligned"

    def inversion(self, image, model, prompt, size, num_inference_steps):
        x0 = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        zts = ddim_inversion(
            model,
            x0,
            prompt,
            num_inference_steps,
            2
        )
        return (zts,)


class SAHandler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "share_group_norm": ("BOOLEAN", {"default": False, }),
                "share_layer_norm": ("BOOLEAN", {"default": False, }),
                "share_attention": ("BOOLEAN", {"default": True, }),
                "adain_queries": ("BOOLEAN", {"default": True, }),
                "adain_keys": ("BOOLEAN", {"default": True, }),
                "adain_values": ("BOOLEAN", {"default": False, }),
                "model": ("MODEL",)
            },
            "optional": {
                "shared_score_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5}),
                "shared_score_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "handler"
    CATEGORY = "StyleAligned"

    def handler(self,
                share_group_norm,
                share_layer_norm,
                share_attention,
                adain_queries,
                adain_keys,
                adain_values,
                shared_score_scale,
                shared_score_shift,
                model):
        sa_args = StyleAlignedArgs(
            share_group_norm=share_group_norm,
            share_layer_norm=share_layer_norm,
            share_attention=share_attention,
            adain_queries=adain_queries,
            adain_keys=adain_keys,
            adain_values=adain_values,
            shared_score_scale=shared_score_scale,
            shared_score_shift=shared_score_shift,
        )
        handler = Handler(model)
        handler.register(sa_args, )
        return (model,)


class SASDXLKampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, }),
                "num_images_per_prompt": ("INT", {"default": 2, "min": 2, "max": 8, "step": 1, "display": "slider"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "kampler"
    CATEGORY = "StyleAligned"

    def kampler(self,
                model,
                prompt,
                num_images_per_prompt, ):
        prompt = process_input(prompt)

        images = model(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        output_image = convert_preview_image(images)
        return (output_image,)


class SASDXLTransferKsampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latents": ("LATENTS",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, }),
                "num_inference_steps": ("INT", {"default": 50, "min": 20, "max": 50, "disaply": "slider"}),
                "num_images_per_prompt": ("INT", {"default": 2, "min": 2, "max": 8, "step": 1, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "transfer_kampler"
    CATEGORY = "StyleAligned"

    def transfer_kampler(self, model, latents, prompt, num_inference_steps, num_images_per_prompt):
        zts = latents
        zT, inversion_callback = make_inversion_callback(zts, offset=5)
        g_cpu = torch.Generator(device='cpu')
        g_cpu.manual_seed(10)

        prompt = process_input(prompt)
        latents = torch.randn(len(prompt), 4, 128, 128, device='cpu', generator=g_cpu,
                              dtype=model.unet.dtype, ).to('cuda')
        latents[0] = zT

        images = model(prompt,
                       latents=latents,
                       callback_on_step_end=inversion_callback,
                       num_inference_steps=num_inference_steps,
                       guidance_scale=10.0).images
        output_image = convert_preview_image(images)
        return (output_image,)


class SASDXLControlnetKsampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, }),
                "reference_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, }),
                "num_inference_steps": ("INT", {"default": 50, "min": 20, "max": 50, "disaply": "slider"}),
                "num_images_per_prompt": ("INT", {"default": 2, "min": 2, "max": 8, "step": 1, "display": "slider"}),
                "controlnet_conditioning_scale": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 5.0, "step": 0.1})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "controlnet_kampler"
    CATEGORY = "StyleAligned"

    def controlnet_kampler(self,
                           model,
                           image,
                           prompt,
                           reference_prompt,
                           num_inference_steps,
                           num_images_per_prompt,
                           controlnet_conditioning_scale):
        prompt = process_input(prompt)
        latents = torch.randn(1 + num_images_per_prompt, 4, 128, 128).to(model.unet.dtype)
        images = controlnet_call(
            model,
            [reference_prompt, prompt],
            image=image,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_images_per_prompt=num_images_per_prompt,
            latents=latents
        )

        output_image = convert_preview_image(images)
        return (output_image,)


NODE_CLASS_MAPPINGS = {
    "SASDXL_ModelLoader": SASDXL_ModelLoader,
    "SchedulerLoader": SchedulerLoader,
    "SAControlnet_ModelLoader": SAControlnet_ModelLoader,
    "SADepth": SADepth,
    "SAInversion": SAInversion,
    "SAHandler": SAHandler,
    "SASDXLKampler": SASDXLKampler,
    "SASDXLTransferKsampler": SASDXLTransferKsampler,
    "SASDXLControlnetKsampler": SASDXLControlnetKsampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SASDXL_ModelLoader": "SA SDXL Model Loader",
    "SchedulerLoader": "SA Scheduler Loader",
    "SAControlnet_ModelLoader": "SA Controlnet ModelLoader",
    "SADepth": "SA Depth",
    "SAInversion": "SA Inversion",
    "SAHandler": "SA Handler",
    "SASDXLKampler": "SA SDXL Kampler",
    "SASDXLTransferKsampler": "SA SDXL Transfer Kampler",
    "SASDXLControlnetKsampler": "SA SDXL Controlnet Kampler"
}
