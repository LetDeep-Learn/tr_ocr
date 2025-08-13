# utils.py
import os
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
    """
    save_path = os.path.join(config.DRIVE_SAVE_DIR, f"{name}_epoch{epoch}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"[INFO] Saved checkpoint to {save_path}")
