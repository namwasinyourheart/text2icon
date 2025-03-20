import os
import argparse
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from diffusers.utils import make_image_grid
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion with LoRA adaptation."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()

    # Load configuration from the specified YAML file.
    config = OmegaConf.load(args.config_path)
    model_cfg = config.model
    gen_cfg = config.generate

    # Extract model parameters.
    model_name_or_path = model_cfg.model_name_or_path
    # lora_name = model_cfg.lora_name
    # output_dir = model_cfg.output_dir
    # lora_model_path = os.path.join(output_dir, lora_name)
    lora_path = model_cfg.lora_path

    # Extract generation parameters.
    prompt = list(gen_cfg.prompt)
    negative_prompt = gen_cfg.negative_prompt
    num_images_per_prompt = gen_cfg.num_images_per_prompt
    generator_seed = gen_cfg.generator_seed
    width = gen_cfg.width
    height = gen_cfg.height
    guidance_scale = gen_cfg.guidance_scale
    scheduler_type = gen_cfg.scheduler

    # Set device.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the base Stable Diffusion pipeline.
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
    ).to(device)

    if lora_path:
        # lora_model_path = os.path.join(output_dir, lora_name)
        # Load LoRA weights.
        print("Loading LoRA Adapter...")
        pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict=lora_path,
            adapter_name="az_lora"
        )

        # Activate the LoRA adapter.
        pipe.set_adapters(["az_lora"], adapter_weights=[1.0])

    # Configure scheduler.
    if scheduler_type == "EulerDiscreteScheduler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    # Create a random generator for reproducibility.
    generator = torch.Generator(device).manual_seed(generator_seed)

    # Generate images.
    output = pipe(
        prompt=prompt,
        # negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        width=width,
        height=height,
        guidance_scale=guidance_scale
    )
    images = output.images

    # Create and save an image grid.
    rows = len(prompt)
    import math
    cols = num_images_per_prompt # math.ceil(rows * num_images_per_prompt / rows)
    grid = make_image_grid(images, rows=rows, cols=cols)
    grid.save("output.png")
    print("Output image saved as output.png")

if __name__ == "__main__":
    main()
