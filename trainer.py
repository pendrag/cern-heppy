import torch
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    TrainingArguments, 
    Trainer
)


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # Calculate weights: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        # FIXME: ¿Que hacemos con las divisiones entre 0?
        # num_positives_per_column = torch.sum(labels, dim=0)
        # pos_weight = (len(labels)-num_positives_per_column)/num_positives_per_column

        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


def create_trainer(
    trainer_path, 
    model,
    epochs,
    train_split, 
    validation_split, 
    tokenizer, 
    data_collator,
    load_best_model_at_end=False,
    metric_for_best_model=None,
    callbacks=None
):
    """Creates a trainer for the model.
    
    Returns:
        The multilabel trainer already set up.
    """

    def compute_metrics(eval_pred):
        """Computes metrics during evaluation.
        
        Returns:
            A dictionary with the name of the metrics as keys and their score as float values."""

        logits, labels = eval_pred

        logits = torch.sigmoid(torch.from_numpy(logits)).numpy()
        predictions = [(logits>th).astype(np.float32) for th in np.arange(0.1, 1.0, 0.1)]

        precision_metrics = {f'precision_0{i+1}': precision_score(y_true=labels, y_pred=prediction, average='micro') for i, prediction in enumerate(predictions)}
        recall_metrics = {f'recall_0{i+1}': recall_score(y_true=labels, y_pred=prediction, average='micro') for i, prediction in enumerate(predictions)}
        f1_metrics = {f'f1_0{i+1}': f1_score(y_true=labels, y_pred=prediction, average='micro') for i, prediction in enumerate(predictions)}

        return {**precision_metrics, **recall_metrics, **f1_metrics}

    training_args = TrainingArguments(
        output_dir=trainer_path,
        evaluation_strategy='epoch',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model
    )

    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_split, 
        eval_dataset=validation_split,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    return trainer


# def main(*args, **kwargs):
#     # Paths
#     dl_path = cfg['data_loader_path']
#     local_data_path = cfg['local_data_path']

#     # Load tokenizer and data_collator
#     if cfg['fine_tune']:
#         checkpoint = cfg['default_checkpoint_path']
#     else:
#         checkpoint = os.path.join(cfg['trained_checkpoints_path'], cfg['trained_checkpoint'])
#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#     data_collator = DataCollatorWithPadding(tokenizer, return_tensors='pt')

#     # Load dataset and split it in train/test
#     full_dataset = load_dataset(dl_path, split='hepth', data_dir=local_data_path)
#     train_val_split = full_dataset.train_test_split(cfg['val_split_ratio'])

#     # Get unique keywords (only kw1 for now) and build a label indexer
#     kws = full_dataset['keywords']
#     unique_kws = {kw.split(',')[0] for kws_per_record in kws for kw in kws_per_record}
#     labels2ids = {label: id for id, label in enumerate(unique_kws)}
#     ids2labels = {id: label for id, label in enumerate(unique_kws)}

#     # Tokenize inputs and one-hot outputs
#     tokenized_datasets = data_loader.prepare_features(train_val_split, tokenizer, labels2ids)
#     tokenized_datasets = tokenized_datasets.remove_columns(
#         ['title', 'subject', 'description', 'keywords']
#     )
#     tokenized_datasets.set_format('torch')
#     # train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=8, collate_fn=data_collator)
#     # test_dataloader = DataLoader(tokenized_datasets['test'], batch_size=8, collate_fn=data_collator)

#     train_split = tokenized_datasets['train']
#     val_split = tokenized_datasets['test']

#     # Load model and create a new trainer
#     model = AutoModelForSequenceClassification.from_pretrained(
#         checkpoint, 
#         num_labels=len(labels2ids)
#     )

#     trainer = create_trainer(
#         cfg['trained_checkpoints_path'], 
#         model, 
#         train_split, 
#         val_split, 
#         tokenizer, 
#         data_collator
#     )

#     # Train/evaluate
#     if cfg['fine_tune']:
#         trainer.train()
#     else:
#         results = trainer.evaluate(val_split)
#         for key, value in results.items():
#             print(f'{key}:\t{value}')


# if __name__ == "__main__":
#     # Load configuration file
#     cfg_path = os.path.join(os.path.dirname(__file__), 'config.yml')
#     with open(cfg_path, 'r') as f:
#         cfg = yaml.safe_load(f)

#     main(cfg)
