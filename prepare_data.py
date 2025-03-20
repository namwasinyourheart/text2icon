# prepare_data.py
from datasets import load_dataset
from torchvision import transforms
import torch

def tokenize_captions(examples, tokenizer, caption_column):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return inputs.input_ids

def get_train_transforms(resolution, center_crop=True, random_flip=True):
    transform_list = [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    ]
    if center_crop:
        transform_list.append(transforms.CenterCrop(resolution))
    else:
        transform_list.append(transforms.RandomCrop(resolution))
    if random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    else:
        transform_list.append(transforms.Lambda(lambda x: x))
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transforms.Compose(transform_list)

def preprocess_train(examples, tokenizer, image_column, caption_column, train_transforms):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples, tokenizer, caption_column)
    return examples

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

def prepare_dataset(dataset_name, train_data_dir, train_n_samples, tokenizer):
    if dataset_name:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset("imagefolder", data_dir=train_data_dir)
    train_data = dataset["train"]
    if train_n_samples>0:
        dataset["train"] = train_data.select(range(train_n_samples))
    return dataset

def get_dataloader(dataset, tokenizer, resolution, center_crop, random_flip, batch_size):
    dataset_columns = list(dataset["train"].features.keys())
    image_column, caption_column = dataset_columns[0], dataset_columns[1]
    train_transforms = get_train_transforms(resolution, center_crop, random_flip)
    def transform_fn(examples):
        return preprocess_train(examples, tokenizer, image_column, caption_column, train_transforms)
    train_dataset = dataset["train"].with_transform(transform_fn)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=0
    )
    return dataloader, image_column, caption_column
