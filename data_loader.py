import os, sys
import yaml
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

import data_loader


def prepare_features(dataset, tokenizer, labels2ids):
    """Tokenizes inputs and onehot-encodes labels."""
    
    def tokenize_features(samples):
        return tokenizer(samples['description'], truncation=True, max_length=512) # Do not pad here since we'll dynamically pad with the data-collator

    def onehot_encode(samples):
        labels = np.zeros((len(samples['keywords']), len(labels2ids)))
        for id, kws_per_record in enumerate(samples['keywords']):
            for kw in kws_per_record:
                labels[id][labels2ids[kw.split(',')[0]]] = 1
        return {'labels': labels}

    # Tokenize description (abstract)
    dataset = dataset.map(tokenize_features, batched=True)  
    # Onehot-encode labels
    dataset = dataset.map(onehot_encode, batched=True)

    return dataset


def main(*args, **kwargs):
    # Load dataset and split it in train/val
    dl_path = cfg['data_loader_path']
    local_data_path = cfg['local_data_path']

    # Load tokenizer and data_collator
    if cfg['fine_tune']:
        checkpoint = cfg['default_checkpoint_path']
    else:
        checkpoint = os.path.join(cfg['trained_checkpoints_path'], cfg['trained_checkpoint'])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors='pt')

    full_dataset = load_dataset(dl_path, split='hepth', data_dir=local_data_path)
    train_val_split = full_dataset.train_test_split(cfg['val_split_ratio'])

    print("3")
    # Get unique keywords (only kw1 for now)
    kws = full_dataset['keywords']
    # del full_dataset
    unique_kws = {kw.split(',')[0] for kws_per_record in kws for kw in kws_per_record}
    labels2ids = {label: id for id, label in enumerate(unique_kws)}
    ids2labels = {id: label for id, label in enumerate(unique_kws)}

    tokenized_datasets = data_loader.prepare_features(train_val_split, tokenizer, labels2ids)
    tokenized_datasets = tokenized_datasets.remove_columns(['title', 'subject', 'description', 'keywords'])
    tokenized_datasets.set_format('torch')

    train_split = tokenized_datasets['train']
    val_split = tokenized_datasets['test']

    
if __name__ == "__main__":
    # Load configuration file
    cfg_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg)
