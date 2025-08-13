# train.py
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from tqdm import tqdm
import os
import config
from dataset import get_dataset
from model import load_model, freeze_encoder, unfreeze_last_n_encoder_layers
from utils import save_checkpoint, load_checkpoint
from peft import LoraConfig, get_peft_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_collate_fn(processor, max_target_length=128):
    tok = processor.tokenizer
    pad_id = tok.pad_token_id

    def collate_fn(batch):
        images = [b["image"] for b in batch]
        texts = [b["text"] for b in batch]

        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        enc = tok(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt"
        )
        labels = enc.input_ids
        labels[labels == pad_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

    return collate_fn


def train_phase(model, dataloader, optimizer, scheduler, num_epochs, processor, phase_name, start_epoch=0):
    model.train()
    for epoch in range(start_epoch, num_epochs):
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

        save_checkpoint(
            model,
            processor,
            epoch+1,
            name=phase_name,
            optimizer=optimizer,
            scheduler=scheduler,
            extra_state={"phase": phase_name, "num_epochs": num_epochs}
        )


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

    # ===== Load Base Model =====
    model = load_model()
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id or processor.tokenizer.pad_token_id

    # ===== Inject LoRA =====
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Adjust depending on model
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.to(device)

    # ==== Resume checkpoint if exists ====
    ckpt_path = os.path.join(config.SAVE_DIR, "last_checkpoint.pt")
    start_phase = "phase1"
    start_epoch = 0
    optimizer, scheduler = None, None

    if os.path.exists(ckpt_path):
        print(f"[INFO] Resuming from checkpoint: {ckpt_path}")
        ckpt_data = load_checkpoint(ckpt_path, model)
        optimizer = ckpt_data["optimizer"]
        scheduler = ckpt_data["scheduler"]
        start_phase = ckpt_data["extra_state"].get("phase", "phase1")
        start_epoch = ckpt_data["epoch"]
    else:
        print("[INFO] No checkpoint found. Starting fresh training.")

    # Phase 1: LoRA + frozen encoder
    if start_phase == "phase1":
        if optimizer is None:
            freeze_encoder(model.base_model)  # freeze base encoder
            optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
            total_steps = len(dataloader) * config.NUM_EPOCHS_PHASE1
            scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        train_phase(model, dataloader, optimizer, scheduler, config.NUM_EPOCHS_PHASE1, processor, "phase1", start_epoch)
        start_epoch = 0
        start_phase = "phase2"

    # Phase 2: Unfreeze last few encoder layers + LoRA
    if start_phase == "phase2":
        unfreeze_last_n_encoder_layers(model.base_model, n=4)
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE/2, weight_decay=config.WEIGHT_DECAY)
        total_steps = len(dataloader) * config.NUM_EPOCHS_PHASE2
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        train_phase(model, dataloader, optimizer, scheduler, config.NUM_EPOCHS_PHASE2, processor, "phase2", start_epoch)


if __name__ == "__main__":
    main()
