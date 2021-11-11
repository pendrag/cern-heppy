import os
import yaml
import torch
import numpy as np

from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding, 
    EarlyStoppingCallback
)

import data_loader
import trainer

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
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors='pt')

    # Load dataset and split it in train/test
    full_dataset = load_dataset(dl_path, split='hepth', data_dir=local_data_path)
    train_val_split = full_dataset.train_test_split(cfg['val_split_ratio'])

    # Get unique keywords (only kw1 for now) and build a label indexer
    kws = full_dataset['keywords']
    unique_kws = {kw.split(',')[0] for kws_per_record in kws for kw in kws_per_record}
    labels2ids = {label: id for id, label in enumerate(unique_kws)}
    ids2labels = {id: label for id, label in enumerate(unique_kws)}

    # Tokenize inputs and one-hot outputs
    tokenized_datasets = data_loader.prepare_features(train_val_split, tokenizer, labels2ids)
    tokenized_datasets = tokenized_datasets.remove_columns(
        ['title', 'subject', 'description', 'keywords']
    )
    tokenized_datasets.set_format('torch')
    # train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=8, collate_fn=data_collator)
    # test_dataloader = DataLoader(tokenized_datasets['test'], batch_size=8, collate_fn=data_collator)

    train_split = tokenized_datasets['train']
    val_split = tokenized_datasets['test']

    # Load model and create a new trainer
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, 
        num_labels=len(labels2ids)
    )

    # Trainer for the first 20 epochs (fix for the ES problem)
    warmup_trainer = trainer.create_trainer(
        trainer_path=cfg['warmedup_checkpoints_path'], 
        model=model,
        epochs=20,
        train_split=train_split, 
        validation_split=val_split, 
        tokenizer=tokenizer, 
        data_collator=data_collator
    )

    # Post-warmup trainer with early stopping
    early_stopping_trainer = trainer.create_trainer(
        trainer_path=cfg['trained_checkpoints_path'], 
        model=model,
        epochs=100,
        train_split=train_split, 
        validation_split=val_split, 
        tokenizer=tokenizer, 
        data_collator=data_collator,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1_03',
        callbacks=[EarlyStoppingCallback(5)]
    )

    # Train/evaluate
    if cfg['fine_tune']:
        print("Warm up training for 20 epochs")
        #warmup_trainer.train()
        print("Full training for 100 epochs (with Early Stopping)")
        early_stopping_trainer.train(cfg['warmedup_checkpoints_path'] + cfg['warmedup_checkpoint'])
        print("Training finished")
    else:
        predictions = early_stopping_trainer.predict(val_split)

        logits = torch.sigmoid(torch.from_numpy(predictions[0])).numpy()
        preds = (logits>0.3).astype(np.float32)
        conf_matrices = multilabel_confusion_matrix(val_split['labels'], preds)
        precisions = precision_score(val_split['labels'], preds, average=None)
        recalls = recall_score(val_split['labels'], preds, average=None)
        f1s = f1_score(val_split['labels'], preds, average=None)      

        print('TN\tFN\tTP\tFP\tPrecision\tRecall\tF1')
        for conf_matrix, p, r, f in zip(conf_matrices, precisions, recalls, f1s):
            print(
                f'{conf_matrix[0][0]}\t', 
                f'{conf_matrix[1][0]}\t',
                f'{conf_matrix[1][1]}\t',
                f'{conf_matrix[0][1]}\t',
                f'{p}\t',
                f'{r}\t',
                f'{f}'
            )

if __name__ == "__main__":
    # Load configuration file
    cfg_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg)