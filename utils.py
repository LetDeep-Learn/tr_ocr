# utils.py
import os
import json
import evaluate  # <-- correct library (NOT your local dataset.py)
import config

# Load metrics once
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_sequence_metrics(pred_texts, ref_texts):
    """
    Compute CER/WER from already-decoded strings.
    Use this in a validation loop if you want quick metrics.
    """
    cer = cer_metric.compute(predictions=pred_texts, references=ref_texts)
    wer = wer_metric.compute(predictions=pred_texts, references=ref_texts)
    return {"cer": cer, "wer": wer}

def save_checkpoint(model, processor, epoch, name="checkpoint"):
    """
    Save model+processor to Drive after each epoch.
    Also save a metadata.json with the epoch number so we can resume later.
    """
    save_path = os.path.join(config.DRIVE_SAVE_DIR, f"{name}_epoch{epoch}")
    os.makedirs(save_path, exist_ok=True)

    # Save model + processor
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    # Save metadata for resuming
    meta_path = os.path.join(save_path, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"epoch": epoch, "name": name}, f)

    print(f"[INFO] Saved checkpoint to {save_path}")

def load_last_checkpoint(model_cls, processor_cls):
    """
    Load the most recent checkpoint from DRIVE_SAVE_DIR.
    Returns: (model, processor, last_epoch) or (None, None, 0) if not found.
    """
    if not os.path.exists(config.DRIVE_SAVE_DIR):
        return None, None, 0

    checkpoints = sorted(os.listdir(config.DRIVE_SAVE_DIR))
    if not checkpoints:
        return None, None, 0

    last_ckpt = os.path.join(config.DRIVE_SAVE_DIR, checkpoints[-1])
    model = model_cls.from_pretrained(last_ckpt)
    processor = processor_cls.from_pretrained(last_ckpt)

    # Load last epoch from metadata
    meta_path = os.path.join(last_ckpt, "metadata.json")
    last_epoch = 0
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            last_epoch = meta.get("epoch", 0)

    print(f"[INFO] Loaded checkpoint from {last_ckpt} (epoch {last_epoch})")
    return model, processor, last_epoch
