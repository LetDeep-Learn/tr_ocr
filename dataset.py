# dataset.py
import os
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from transformers import TrOCRProcessor
import config

class ModiDevanagariDataset(Dataset):
    def __init__(self, image_dirs, label_dir, processor, target_height=384):
        """
        image_dirs: list[str] or str — one or more folders of images
        label_dir: str — folder containing .txt labels (same basename as image)
        processor: TrOCRProcessor
        """
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]
        self.image_dirs = image_dirs
        self.label_dir = label_dir
        self.processor = processor
        self.target_height = target_height

        # Collect all image paths (png/jpg/jpeg)
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
        self.images = []
        for d in self.image_dirs:
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(exts):
                    self.images.append((d, f))

        # Ensure tokenizer has a pad token (GPT2-like)
        tok = self.processor.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    def __len__(self):
        return len(self.images)

    def _pad_or_resize_height(self, image):
        """Keep aspect ratio; make height == target_height (pad if smaller, downscale if taller)."""
        th = self.target_height
        if image.height < th:
            pad_top = (th - image.height) // 2
            pad_bottom = th - image.height - pad_top
            image = ImageOps.expand(image, border=(0, pad_top, 0, pad_bottom), fill=(255, 255, 255))
        elif image.height > th:
            new_w = int(image.width * th / image.height)
            image = image.resize((new_w, th), Image.BICUBIC)
        return image

    def __getitem__(self, idx):
        img_dir, img_name = self.images[idx]
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

        # Load image + label
        image = Image.open(img_path).convert("RGB")
        with open(label_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Height normalization (no stretching)
        image = self._pad_or_resize_height(image)

        # Return raw PIL image + raw text; collate_fn will batch-process
        return {"image": image, "text": text}

def get_dataset():
    processor = TrOCRProcessor.from_pretrained(config.MODEL_NAME)
    # Ensure pad token globally
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    train_dataset = ModiDevanagariDataset(
        image_dirs=[config.ORG_IMAGES_DIR, config.SYNTHETIC_IMAGES_DIR],
        label_dir=config.IMAGE_LABELS_DIR,
        processor=processor
    )
    return train_dataset, processor
