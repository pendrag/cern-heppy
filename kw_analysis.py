import os
import yaml
import torch
import numpy as np

from operator import itemgetter
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
from datasets import load_dataset

import data_loader

def main(*args, **kwargs):
    # Paths
    dl_path = cfg['data_loader_path']
    local_data_path = cfg['local_data_path']

    # Load tokenizer and data_collator
    if cfg['fine_tune']:
        checkpoint = cfg['default_checkpoint_path']
    else:
        checkpoint = os.path.join(cfg['trained_checkpoints_path'], cfg['trained_checkpoint'])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Load dataset and split it in train/test
    full_dataset = load_dataset(dl_path, split='hepth', data_dir=local_data_path)

    # Get unique keywords (only kw1 for now) and build a label indexer
    kws = full_dataset['keywords']
    unique_kws = {kw.split(',')[0] for kws_per_record in kws for kw in kws_per_record}
    labels2ids = {label: id for id, label in enumerate(unique_kws)}
    ids2labels = {id: label for id, label in enumerate(unique_kws)}

    tokenized_dataset = data_loader.prepare_features(full_dataset, tokenizer, labels2ids)
    tokenized_dataset = tokenized_dataset.remove_columns(
        ['title', 'subject', 'description', 'keywords']
    )
    tokenized_dataset.set_format('numpy')

    # Count label repetitions and sort them by frequency
    label_freq = [(ids2labels[i], value.item()) for i, value in enumerate(sum(tokenized_dataset['labels']))]
    sorted_list = sorted(label_freq, key=itemgetter(1))[::-1]
    for kw in sorted_list:
        print(f'{labels2ids[kw[0]]}\t{kw[0]}\t{int(kw[1])}')

if __name__ == "__main__":
    # Load configuration file
    cfg_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg)