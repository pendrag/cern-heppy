import os
import yaml
import torch

from sklearn.metrics import f1_score, precision_score, recall_score
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding, EarlyStoppingCallback

import data_loader


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # loss_fct = torch.nn.BCEWithLogitsLoss()
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels),
        #                 labels.float().view(-1, self.model.config.num_labels))
        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # TODO: Tiene que haber una forma de recibir directamente tensores en lugar de np.array
    logits = torch.sigmoid(torch.tensor(logits))
    labels = torch.tensor(labels)
    predictions = (logits>0.5).clone().detach().float()

    return {
        'macro_precision': precision_score(y_true=labels, y_pred=predictions, average='macro'),
        'macro_recall': recall_score(y_true=labels, y_pred=predictions, average='macro'),
        'macro_f1': f1_score(y_true=labels, y_pred=predictions, average='macro'),
        'micro_precision': precision_score(y_true=labels, y_pred=predictions, average='micro'),
        'micro_recall': recall_score(y_true=labels, y_pred=predictions, average='micro'),
        'micro_f1': f1_score(y_true=labels, y_pred=predictions, average='micro')
    }
    #return metric.compute(predictions=predictions, references=labels)


def train(trainer_path, model, train_split, validation_split, tokenizer, data_collator):
    """Trains the model."""
    training_args = TrainingArguments(
        output_dir=trainer_path,
        evaluation_strategy='epoch',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=100,
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_micro_f1'
    )

    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_split, 
        eval_dataset=validation_split,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # callbacks=[
        #     EarlyStoppingCallback(
        #         early_stopping_patience=3,
        #         early_stopping_threshold=2.
        #     )
        # ]
    )

    trainer.train()

    return trainer


def main(*args, **kwargs):
    # Load dataset and split it in train/val
    print("1")
    dl_path = cfg['data_loader_path']
    local_data_path = cfg['local_data_path']
    ratio = cfg['train_split_ratio']

    print("2")
    full_dataset = load_dataset(dl_path, split='hepth', data_dir=local_data_path)
    train_split = load_dataset(dl_path, split=f'hepth[:{ratio}%]', data_dir=local_data_path)
    val_split = load_dataset(dl_path, split=f'hepth[{ratio}%:]', data_dir=local_data_path)

    print("3")
    # Get unique keywords (only kw1 for now)
    kws = full_dataset['keywords']
    del full_dataset
    unique_kws = {kw.split(',')[0] for kws_per_record in kws for kw in kws_per_record}
    labels2ids = {label: id for id, label in enumerate(unique_kws)}
    ids2labels = {id: label for id, label in enumerate(unique_kws)}

    print("4")
    # Load tokenizer
    checkpoint = cfg['checkpoint_path']
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors='pt')

    print("5")
    # Tokenize inputs and onehot-encode outputs
    train_split = data_loader.prepare_features(train_split, tokenizer, labels2ids)
    val_split = data_loader.prepare_features(val_split, tokenizer, labels2ids)

    # Training
    #global metric 
    #metric = load_metric(cfg['f1_metric_path'])
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(labels2ids))

    trainer = train(cfg['trainer_path'], model, train_split, val_split, tokenizer, data_collator)

if __name__ == "__main__":
    print("0")
    cfg_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg)
