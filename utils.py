# utils.py
import torch
import numpy as np
from ufal.chu_liu_edmonds import chu_liu_edmonds
from config import DEVICE

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params

def evaluate(model, dataloader):
    model.eval()
    total_tokens, correct_heads, correct_heads_and_rels = 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            head_labels = batch["head"].to(DEVICE)
            deprel_ids = batch["deprel_ids"].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            arc_scores = outputs['arc_scores']
            rel_scores = outputs['rel_scores']

            predicted_heads = torch.argmax(arc_scores, dim=2)
            batch_size, seq_len = predicted_heads.shape
            batch_idx = torch.arange(batch_size)[:, None].expand(-1, seq_len).to(DEVICE)
            dep_idx = torch.arange(seq_len)[None, :].expand(batch_size, -1).to(DEVICE)
            rel_scores_for_predicted_heads = rel_scores[batch_idx, dep_idx, predicted_heads, :]
            predicted_rels = torch.argmax(rel_scores_for_predicted_heads, dim=2)

            mask = head_labels != -100
            correct_heads += torch.sum((predicted_heads == head_labels) & mask).item()
            correct_heads_and_rels += torch.sum(
                (predicted_heads == head_labels) & (predicted_rels == deprel_ids) & mask
            ).item()
            total_tokens += torch.sum(mask).item()

    uas = correct_heads / total_tokens if total_tokens > 0 else 0.0
    las = correct_heads_and_rels / total_tokens if total_tokens > 0 else 0.0
    return {'UAS': uas, 'LAS': las}

def mst_parsing(model, dataloader, device):
    model.eval()
    total_tokens, correct_tokens = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            head_labels = batch["head"].to(device)
            tokens_representing_words = batch["tokens_representing_words"].to(device)

            scores_dict = model(input_ids, attention_mask)
            scores = scores_dict["arc_scores"]
            log_probs = torch.log_softmax(scores, dim=2).cpu().numpy()

            for i in range(input_ids.shape[0]):
                word_token_indices = tokens_representing_words[i].cpu().numpy()
                word_token_indices = word_token_indices[word_token_indices != -1]
                filtered_log_probs = log_probs[i][word_token_indices][:, word_token_indices]
                heads, _ = chu_liu_edmonds(filtered_log_probs.astype(np.float64))
                predicted_heads = word_token_indices[heads]
                gold_heads = head_labels[i].cpu().numpy()[word_token_indices]
                correct_tokens += np.sum(predicted_heads == gold_heads)
                total_tokens += len(gold_heads)

    uas = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return uas