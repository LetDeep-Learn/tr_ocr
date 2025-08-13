# train.py
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from tqdm import tqdm
import config
from dataset import get_dataset
from model import load_model, freeze_encoder, unfreeze_last_n_encoder_layers
from utils import save_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_collate_fn(processor, max_target_length=128):
    tok = processor.tokenizer
    pad_id = tok.pad_token_id

    def collate_fn(batch):
        # Batch of dicts: {"image": PIL.Image, "text": str}
        images = [b["image"] for b in batch]
        texts  = [b["text"] for b in batch]

        # Images -> pixel_values (batched)
        pixel_values = processor(images=images, return_tensors="pt").pixel_values

        # Texts -> token ids with dynamic padding to longest
        enc = tok(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt"
        )
        labels = enc.input_ids
        # Mask pads with -100 so they don't contribute to loss
        labels[labels == pad_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}
    return collate_fn

def train_phase(model, dataloader, optimizer, scheduler, num_epochs, processor, phase_name):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loop.set_description(f"{phase_name} Epoch {epoch+1}/{num_epochs}")
            loop.set_postfix(batch_loss=loss.item())

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"✅ {phase_name} | Epoch {epoch+1}/{num_epochs} | Average Loss: {avg_loss:.4f}")
        save_checkpoint(model, processor, epoch+1, name=phase_name)

def main():
    train_dataset, processor = get_dataset()
    collate_fn = make_collate_fn(processor, max_target_length=128)
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )

    model = load_model()
    # Ensure special tokens set
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id or processor.tokenizer.pad_token_id
    model.to(device)

    # Phase 1: Freeze encoder
    freeze_encoder(model)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(dataloader) * config.NUM_EPOCHS_PHASE1
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    train_phase(model, dataloader, optimizer, scheduler, config.NUM_EPOCHS_PHASE1, processor, phase_name="phase1")

    # Phase 2: Unfreeze last N encoder layers
    unfreeze_last_n_encoder_layers(model, n=4)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE/2, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(dataloader) * config.NUM_EPOCHS_PHASE2
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    train_phase(model, dataloader, optimizer, scheduler, config.NUM_EPOCHS_PHASE2, processor, phase_name="phase2")

if __name__ == "__main__":
    main()
