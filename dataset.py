# dataset.py
import os
from PIL import Image,ImageOps
from torch.utils.data import Dataset
from transformers import TrOCRProcessor
import config

class ModiDevanagariDataset(Dataset):
    def __init__(self, image_dir, label_dir, processor):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.processor = processor
        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

        # Load image and label
        image = Image.open(img_path).convert("RGB")
        with open(label_path, "r", encoding="utf-8") as f:
            label = f.read().strip()

        # === Resize with vertical padding ===
        target_height = 384
        if image.height < target_height:
            pad_top = (target_height - image.height) // 2
            pad_bottom = target_height - image.height - pad_top
            image = ImageOps.expand(image, border=(0, pad_top, 0, pad_bottom), fill=(255, 255, 255))
        elif image.height > target_height:
            image = image.resize((int(image.width * target_height / image.height), target_height), Image.BICUBIC)

        # Let processor handle width scaling if needed
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()

        return {
            "pixel_values": pixel_values,
            "labels": self.processor.tokenizer(label, return_tensors="pt").input_ids.squeeze()
        }

def get_dataset():
    processor = TrOCRProcessor.from_pretrained(config.MODEL_NAME)
    train_dataset = ModiDevanagariDataset(
        image_dirs=[config.ORG_IMAGES_DIR, config.SYNTHETIC_IMAGES_DIR],
        labels_dir=config.IMAGE_LABELS_DIR,
        processor=processor
    )
    return train_dataset, processor
