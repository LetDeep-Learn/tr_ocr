# eval.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import config
from dataset import get_datasets
from model import load_model
from utils import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, dataloader, processor):
    model.eval()
    predictions, ground_truths = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]

            generated_ids = model.generate(pixel_values)
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            labels = processor.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(preds)
            ground_truths.extend(labels)

    return predictions, ground_truths

def main():
    # Load dataset (you can make a test split here if you want)
    dataset, processor = get_datasets(split="test")  # make sure your get_datasets supports split
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    model = load_model()
    model.to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(config.DRIVE_SAVE_DIR, "latest_checkpoint")
    if os.path.exists(checkpoint_path):
        model, processor = load_checkpoint(model, processor, checkpoint_path)
        print(f"[ckpt] Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"[warn] No checkpoint found at {checkpoint_path}, using fresh model.")

    # Evaluate
    preds, gts = evaluate(model, dataloader, processor)

    # Print first few results
    for i in range(min(10, len(preds))):
        print(f"GT: {gts[i]}")
        print(f"PR: {preds[i]}")
        print("-" * 40)

if __name__ == "__main__":
    main()
