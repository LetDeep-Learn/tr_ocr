# model.py
from transformers import VisionEncoderDecoderModel
import config
import torch.nn as nn

def load_model():
    model = VisionEncoderDecoderModel.from_pretrained(config.MODEL_NAME)
    return model

def freeze_encoder(model):
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("[INFO] Encoder frozen.")

def unfreeze_last_n_encoder_layers(model, n=4):
    total_layers = len(model.encoder.encoder.layers)
    for idx, layer in enumerate(model.encoder.encoder.layers):
        if idx >= total_layers - n:
            for param in layer.parameters():
                param.requires_grad = True
    print(f"[INFO] Last {n} encoder layers unfrozen.")
