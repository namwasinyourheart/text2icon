import torch
from datetime import datetime
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers import StableDiffusionPipeline

def get_formatted_date():
    return datetime.now().strftime(r'%Y%m%d-%H%M%S')

def get_lora_config(lora_rank, lora_alpha, target_modules):
    return LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules # ["to_k", "to_q", "to_v", "to_out.0"]
    )

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def save_lora_weights(unet, output_dir, pretrained_model_name_or_path, lora_rank, max_train_steps, resolution, formatted_date, accelerator):
    unet = unet.to(torch.float32)
    unwrapped_unet = accelerator.unwrap_model(unet)
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
    weight_name = f"lora_{pretrained_model_name_or_path.split('/')[-1]}_rank{lora_rank}_s{max_train_steps}_r{resolution}_{formatted_date}.safetensors"
    StableDiffusionPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
        weight_name=weight_name
    )
