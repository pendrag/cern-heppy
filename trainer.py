import os
import yaml

from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer

import data_loader

def train(trainer_path, model, train_split, validation_split, tokenizer):
    """Trains the model."""
    training_args = TrainingArguments(trainer_path)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_split,
        eval_dataset=validation_split,
        tokenizer=tokenizer
    )

    trainer.train()

    return trainer

def main(*args, **kwargs):
    # Load dataset and split it in train/val
    dl_path = cfg['data_loader_path']
    local_data_path = cfg['local_data_path']
    ratio = cfg['train_split_ratio']

    full_dataset = load_dataset(dl_path, split='hepth', data_dir=local_data_path)
    train_split = load_dataset(dl_path, split=f'hepth[:{ratio}%]', data_dir=local_data_path)
    val_split = load_dataset(dl_path, split=f'hepth[{ratio}%:]', data_dir=local_data_path)

    # Get unique keywords (only kw1 for now)
    kws = full_dataset['keywords']
    del full_dataset
    unique_kws = {kw.split(',')[0] for kws_per_record in kws for kw in kws_per_record}
    labels2ids = {label: id for id, label in enumerate(unique_kws)}
    ids2labels = {id: label for id, label in enumerate(unique_kws)}

    # Load tokenizer
    checkpoint = cfg['checkpoint_path']
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Tokenize inputs and onehot-encode outputs
    train_split = data_loader.prepare_features(train_split, tokenizer, labels2ids)
    val_split = data_loader.prepare_features(val_split, tokenizer, labels2ids)

    # Training
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(labels2ids))

    trainer = train(cfg['trainer_path'], model, train_split, val_split, tokenizer)
    predictions = trainer.predict(val_split)
    print(predictions.predictions.shape, predictions.label_ids.shape)


if __name__ == "__main__":

    with open('config.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg)