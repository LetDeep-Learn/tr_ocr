# train.py
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from tqdm import tqdm
import config
from dataset import get_datasets
from model import load_model, freeze_encoder, unfreeze_last_n_encoder_layers
from utils import save_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_phase(model, dataloader, optimizer, scheduler, num_epochs, processor, phase_name):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        loop = tqdm(dataloader, leave=True)
        
        for batch in loop:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loop.set_description(f"{phase_name} Epoch {epoch+1}/{num_epochs}")
            loop.set_postfix(batch_loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"✅ {phase_name} | Epoch {epoch+1}/{num_epochs} | Average Loss: {avg_loss:.4f}")

        save_checkpoint(model, processor, epoch+1, name=phase_name)

def main():
    train_dataset, processor = get_datasets()
    dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = load_model()
    model.to(device)

    # Phase 1: Freeze encoder
    freeze_encoder(model)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                               num_training_steps=len(dataloader)*config.NUM_EPOCHS_PHASE1)
    train_phase(model, dataloader, optimizer, scheduler, config.NUM_EPOCHS_PHASE1, processor, phase_name="phase1")

    # Phase 2: Unfreeze last N encoder layers
    unfreeze_last_n_encoder_layers(model, n=4)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE/2, weight_decay=config.WEIGHT_DECAY)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                               num_training_steps=len(dataloader)*config.NUM_EPOCHS_PHASE2)
    train_phase(model, dataloader, optimizer, scheduler, config.NUM_EPOCHS_PHASE2, processor, phase_name="phase2")

if __name__ == "__main__":
    main()
