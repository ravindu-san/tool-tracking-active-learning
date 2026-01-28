from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

def get_embeddings(classifier: Any, X: np.ndarray):
    net = classifier.module_

    # Save state
    prev_mode = net.output_mode
    was_training = net.training

    net.output_mode = "embedding"
    net.eval()

    embeddings = []
    probs = []

    with torch.no_grad():
        for emb, logit in classifier.forward_iter(X, training=False):
            embeddings.append(emb.detach().cpu())
            probs.append(F.softmax(logit, dim=-1).detach().cpu())

    # Restore state
    net.output_mode = prev_mode
    net.train(was_training)

    return torch.cat(embeddings, dim=0), torch.cat(probs, dim=0)
