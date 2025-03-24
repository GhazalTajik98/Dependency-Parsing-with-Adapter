# data.py
from datasets import load_dataset
from transformers import XLMRobertaTokenizerFast
import torch
from torch.utils.data import DataLoader
import sys
from config import ALL_DEPRELS, DATASET_NAME, DATASET_PATH

tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
deprel_to_id = {deprel: idx for idx, deprel in enumerate(ALL_DEPRELS)}
id_to_deprel = {idx: deprel for idx, deprel in enumerate(ALL_DEPRELS)}

def strip_none_heads(examples, i):
    tokens = examples["tokens"][i]
    heads = examples["head"][i]
    deprels = examples["deprel"][i]
    non_none = [(t, h, d) for t, h, d in zip(tokens, heads, deprels) if h != "None"]
    return zip(*non_none)

def map_first_occurrence(nums):
    seen = set()
    return {num: i for i, num in enumerate(nums) if num is not None and num not in seen and not seen.add(num)}

def pad_to_same_size(lists, padding_symbol):
    maxlen = max([len(l) for l in lists])
    return [l + (padding_symbol,) * (maxlen - len(l)) for l in lists]

def tokenize_and_align_labels(examples, skip_index=-100):
    examples_tokens, examples_heads, examples_deprels = [], [], []
    for sentence_id in range(len(examples["tokens"])):
        tt, hh, dd = strip_none_heads(examples, sentence_id)
        examples_tokens.append(tt)
        examples_heads.append(hh)
        examples_deprels.append(dd)

    tokenized_inputs = tokenizer(
        examples_tokens, truncation=True, is_split_into_words=True, padding=True
    )

    remapped_heads, deprel_ids, tokens_representing_words, num_words = [], [], [], []
    maxlen_t2w = 0

    for sentence_id, annotated_heads in enumerate(examples_heads):
        deprels = examples_deprels[sentence_id]
        word_ids = tokenized_inputs.word_ids(batch_index=sentence_id)
        word_pos_to_token_pos = map_first_occurrence(word_ids)

        previous_word_idx = None
        heads_here, deprel_ids_here, tokens_representing_word_here = [], [], [0]

        for sentence_position, word_idx in enumerate(word_ids):
            if word_idx is None:
                heads_here.append(skip_index)
                deprel_ids_here.append(skip_index)
            elif word_idx != previous_word_idx:
                if annotated_heads[word_idx] == "None":
                    print("A 'None' head survived!")
                    sys.exit(0)
                head_word_pos = int(annotated_heads[word_idx])
                head_token_pos = 0 if head_word_pos == 0 else word_pos_to_token_pos[head_word_pos - 1]
                heads_here.append(head_token_pos)
                deprel_ids_here.append(deprel_to_id[deprels[word_idx]])
                tokens_representing_word_here.append(sentence_position)
            else:
                heads_here.append(skip_index)
                deprel_ids_here.append(skip_index)
            previous_word_idx = word_idx

        remapped_heads.append(heads_here)
        deprel_ids.append(deprel_ids_here)
        tokens_representing_words.append(tokens_representing_word_here)
        num_words.append(len(tokens_representing_word_here))
        maxlen_t2w = max(maxlen_t2w, len(tokens_representing_word_here))

    for t2w in tokens_representing_words:
        t2w += [-1] * (maxlen_t2w - len(t2w))

    tokenized_inputs["head"] = remapped_heads
    tokenized_inputs["deprel_ids"] = deprel_ids
    tokenized_inputs["tokens_representing_words"] = tokens_representing_words
    tokenized_inputs["num_words"] = num_words
    tokenized_inputs["tokenid_to_wordid"] = [
        tokenized_inputs.word_ids(batch_index=i) for i in range(len(examples_heads))
    ]
    return tokenized_inputs

def explore_some_data(dataset, tokenized_inputs):
    input_ids = tokenized_inputs["input_ids"]
    sentences_tokens_from_input_id = [tokenizer.convert_ids_to_tokens(input_ids[i]) for i in range(len(input_ids))]
    deprel_ids = tokenized_inputs["deprel_ids"]
    deprel_value_by_id = [[id_to_deprel.get(k, "None") for k in ids] for ids in deprel_ids]
    heads = tokenized_inputs["head"]
    tokenid_to_wordid = tokenized_inputs["tokenid_to_wordid"]

    heads_words_list_ids = [
        [tokenid_to_wordid[i][h] if h >= 0 else None for h in heads[i]]
        for i in range(len(heads))
    ]
    heads_words = [
        [dataset[i]["tokens"][idx] if idx else None for idx in heads_words_list_ids[i]]
        for i in range(len(heads_words_list_ids))
    ]

    token_mapping_to_words = []
    for i in range(len(sentences_tokens_from_input_id)):
        raw_sentence = dataset[i]["tokens"]
        tokenized_sentence = sentences_tokens_from_input_id[i]
        token_to_word_ids = tokenid_to_wordid[i]
        temp_token_to_word = {
            token: raw_sentence[word_id] if word_id else None
            for token_idx, token in enumerate(tokenized_sentence)
            for word_id in [token_to_word_ids[token_idx]]
        }
        token_mapping_to_words.append(temp_token_to_word)

    all_tokenized_tokens = [list(temp.keys()) for temp in token_mapping_to_words]
    all_raw_tokens = [list(temp.values()) for temp in token_mapping_to_words]

    for i in range(len(tokenized_inputs)):
        tokenized_tokens = all_tokenized_tokens[i]
        head = heads_words[i]
        deprel = deprel_value_by_id[i]
        words = all_raw_tokens[i]
        for j in range(len(words)):
            print(f"Token : {tokenized_tokens[j]:<10} -> Head: {str(head[j]) if head[j] is not None else 'N/A':<10} -> Deprel: {str(deprel[j]) if deprel[j] is not None else 'N/A':<10} -> Word: {str(words[j]) if words[j] is not None else 'N/A'}")
        print("----------------------------------------")

def dataset_reading_and_encoding(dataset):
    encoded_dataset_loader = {}
    for dataset_type, data in dataset.items():
        index = {"test": 2000, "train": 13000, "validation": 2000}.get(dataset_type, len(data))
        encoded_data = tokenize_and_align_labels(data[:index])
        formatted_data = [
            {
                "input_ids": torch.tensor(encoded_data["input_ids"][i], dtype=torch.long),
                "attention_mask": torch.tensor(encoded_data["attention_mask"][i], dtype=torch.long),
                "head": torch.tensor(encoded_data["head"][i], dtype=torch.long),
                "deprel_ids": torch.tensor(encoded_data["deprel_ids"][i], dtype=torch.long),
                "tokens_representing_words": torch.tensor(encoded_data["tokens_representing_words"][i], dtype=torch.long),
                "num_words": torch.tensor(encoded_data["num_words"][i], dtype=torch.long)
            }
            for i in range(len(encoded_data["input_ids"]))
        ]
        dataloader = DataLoader(formatted_data, batch_size=32, shuffle=True)
        encoded_dataset_loader[dataset_type] = dataloader
    return encoded_dataset_loader

def print_first_batch(dataloader):
    for batch in dataloader:
        print("First Batch:")
        print("input_ids shape:", batch["input_ids"].shape)
        print("attention_mask shape:", batch["attention_mask"].shape)
        print("head shape:", batch["head"].shape)
        print("deprel_ids shape:", batch["deprel_ids"].shape)
        break