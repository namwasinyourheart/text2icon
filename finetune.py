#!/usr/bin/env python
"""
finetune.py

Fine-tuning script for Stable Diffusion with LoRA adaptation.
This script loads experiment configuration using Hydra, sets up the experiment
environment, loads the model and dataset, and runs the training loop.
"""

import os
import shutil
import argparse
import math

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from omegaconf import OmegaConf
from hydra import initialize, compose
from hydra.utils import instantiate

from accelerate import Accelerator, utils as accel_utils
from transformers import set_seed

from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler

# Local modules
from prepare_data import prepare_dataset, get_dataloader
from utils import get_formatted_date, get_lora_config, freeze_parameters, save_lora_weights
from src.utils.log_utils import setup_logger
from src.utils.exp_utils import create_exp_dir


def main():
    """Main function for fine-tuning Stable Diffusion with LoRA adaptation."""
    logger = setup_logger("ft_llm")
    logger.info("Setting up environment...")

    # Parse command-line arguments for configuration.
    parser = argparse.ArgumentParser(description="Process experiment configurations.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration file for the experiment.",
    )
    args, override_args = parser.parse_known_args()

    # Normalize and validate configuration path.
    config_path = os.path.normpath(args.config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    config_dir = os.path.dirname(config_path)
    config_fn = os.path.splitext(os.path.basename(config_path))[0]

    # Load configuration using Hydra.
    try:
        with initialize(version_base=None, config_path=config_dir):
            cfg = compose(config_name=config_fn, overrides=override_args)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")

    logger.info("Loaded configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Ensure experiment name consistency.
    expected_exp_name = os.path.basename(config_path).replace(".yaml", "")
    assert expected_exp_name == cfg.exp_manager.exp_name, (
        f"Experiment name mismatch: expected {expected_exp_name} but got {cfg.exp_manager.exp_name}"
    )

    # Create experiment directories and copy the config file.
    logger.info("Creating experiment directories...")
    exp_name = cfg.exp_manager.exp_name
    exp_dir, configs_dir, data_dir, checkpoints_dir, results_dir = create_exp_dir(exp_name)
    shutil.copy(config_path, configs_dir)

    # Extract configuration parameters.
    exp_args = cfg.exp_manager
    train_args = cfg.train
    data_args = cfg.prepare_data
    model_args = cfg.prepare_model

    # Set random seed for reproducibility.
    seed = exp_args.seed if "seed" in exp_args else 2025
    set_seed(seed)

    # Setup Accelerator.
    accel_utils.write_basic_config()
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=train_args.train_args.gradient_accumulation_steps, 
        mixed_precision="fp16"
    )
    device = accelerator.device
    logger.info(f"Device: {device}")

    # Load scheduler, tokenizer, and models.
    noise_scheduler = DDPMScheduler.from_pretrained(model_args.pretrained_model_name_or_path, subfolder="scheduler")
    weight_dtype = torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        model_args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    ).to(device)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet

    # Freeze parameters of VAE, text encoder, and unet (except LoRA adapters).
    freeze_parameters(unet)
    freeze_parameters(vae)
    freeze_parameters(text_encoder)

    # Configure and add LoRA adapter to unet.
    unet_lora_config = get_lora_config(model_args.lora.r, 
                                       model_args.lora.lora_alpha, 
                                       model_args.lora.target_modules)
    unet.add_adapter(unet_lora_config)
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)

    # Prepare dataset and dataloader.
    dataset = prepare_dataset(
        data_args.dataset.dataset_name,
        data_args.dataset.train_data_dir,
        data_args.dataset.train_n_samples,
        tokenizer,
    )
    train_dataloader, _, _ = get_dataloader(
        dataset,
        tokenizer,
        data_args.image.resolution,
        data_args.image.center_crop,
        data_args.image.random_flip,
        train_args.train_args.per_device_train_batch_size,
    )
    # logger.info("Data Size: %d", len(train_dataloader))

    # max_train_steps = train_args.train_args.num_train_epochs * len(train_dataloader)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_args.train_args.gradient_accumulation_steps)
    if train_args.train_args.max_train_steps is None:
        train_args.train_args.max_train_steps = train_args.train_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Initialize optimizer and learning rate scheduler.
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=train_args.optimizer.learning_rate,
        betas=(train_args.optimizer.adam_beta1, train_args.optimizer.adam_beta2),
        weight_decay=train_args.optimizer.adam_weight_decay,
        eps=train_args.optimizer.adam_epsilon,
    )


    lr_scheduler = get_scheduler(
        train_args.train_args.lr_scheduler_name,
        optimizer=optimizer,
        # num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        # num_training_steps=args.max_train_steps * accelerator.num_processes,
        # num_cycles=args.lr_num_cycles,
        # power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_args.train_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        train_args.train_args.max_train_steps = train_args.train_args.num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    train_args.train_args.num_train_epochs = math.ceil(train_args.train_args.max_train_steps / num_update_steps_per_epoch)


    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Initialise your wandb run, passing wandb parameters and any config information
    accelerator.init_trackers(
        project_name=cfg.exp_manager.wandb.project
        )

    # Train!
    total_batch_size = train_args.train_args.per_device_train_batch_size * accelerator.num_processes * train_args.train_args.gradient_accumulation_steps


    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {train_args.train_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_args.train_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {train_args.train_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {train_args.train_args.max_train_steps}")

    # import wandb
    # wandb.init(
    #     project=cfg.exp_manager.wandb.project,
    #     # name = cfg.exp_manager.exp_name
    # )
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(train_args.train_args.max_train_steps), 
        initial=initial_global_step,
        desc="Steps", 
        disable=not accelerator.is_local_main_process
    )


    # Training loop.
    for epoch in range(first_epoch, train_args.train_args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [unet]
            with accelerator.accumulate(models_to_accumulate):
                # Encode images into latent space.
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Add noise.
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    low=0,
                    high=noise_scheduler.config.num_train_timesteps,
                    size=(batch_size,),
                    device=latents.device,
                ).long()

                # Get text embeddings for conditioning.
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Determine target based on prediction type.
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Forward pass.
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Compute loss and perform backpropagation.
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_layers, train_args.train_args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            if accelerator.is_main_process:
                if train_args.train_args.checkpointing_steps:
                    if global_step % train_args.train_args.checkpointing_steps == 0:
                        save_path = os.path.join(checkpoints_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "epoch": epoch,
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)


    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora_weights(
            unet,
            results_dir,
            model_args.pretrained_model_name_or_path,
            model_args.lora.r,
            train_args.train_args.max_train_steps,
            data_args.image.resolution,
            get_formatted_date(),
            accelerator,
        )


    # Log exp artifact
    if exp_args.wandb.log_artifact == True:
        logger.info("LOGGING EXP ARTIFACTS...")
        # Create an artifact
        import wandb
        artifact = wandb.Artifact(
            name=exp_args.exp_name, 
            type="exp", 
        )

        # Add the directory to the artifact
        artifact.add_dir(exp_dir)

        # wandb_tracker = accelerator.get_tracker("wandb")
        # wandb_tracker.log_artifact(artifact)

        wandb.log_artifact(artifact)

    # # Finish the W&B run
    # wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    main()
