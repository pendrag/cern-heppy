import os, sys
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_features(dataset, tokenizer, labels2ids):
    """Tokenizes inputs and onehot-encodes labels."""
    
    def tokenize_features(example):
        return tokenizer(example['description'], truncation=True, max_length=512) # Do not pad here since we'll dynamically pad with the data-collator

    # TODO: Esto debería de venir hecho directamente del dataset (heppy.py)
    def onehot_encode(example):
        labels = np.zeros((len(example['keywords']), len(labels2ids)))
        for id, kws_per_record in enumerate(example['keywords']):
            for kw in kws_per_record:
                labels[id][labels2ids[kw.split(',')[0]]] = 1
        return {'labels': labels}

    # Tokenize description (abstract)
    dataset = dataset.map(tokenize_features, batched=True)  
    # Onehot-encode labels
    dataset = dataset.map(onehot_encode, batched=True)

    return dataset


def main(*args, **kwargs):
    """Testing the module."""

    ds_path = os.path.join(os.path.dirname(__file__), 'datasets/heppy/heppy.py')
    train_split = load_dataset(ds_path, split='hepth[:80%]')
    test_split = load_dataset(ds_path, split='hepth[80%:]')
    #dataset = datasets['hepth']

    checkpoint = 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    train_split, labels2ids = prepare_features(train_split, tokenizer)
    test_split, _ = prepare_features(test_split, tokenizer)
    
    print(train_split)
    print(test_split)

    
if __name__ == "__main__":
    main()
