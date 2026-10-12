import time
import numpy as np
import torch

def medir_latencia(model, tokenizer, texts, device, n_runs=50):
    model.to(device)
    model.eval()
    latencias = []
    with torch.no_grad():
        for t in texts[:n_runs]:
            inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True).to(device)
            start = time.time()
            _ = model(**inputs)
            latencias.append(time.time() - start)
    return np.mean(latencias), np.std(latencias)
