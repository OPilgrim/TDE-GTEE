import json
import logging
import numpy as np
import pytorch_lightning as pl
import torch
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoConfig,
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from .scorer import multilabel_evaluate

logger = logging.getLogger(__name__)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup
}


class ICModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.hparams.update(vars(args))
        self.model = AutoModelForSequenceClassification.from_pretrained(
                            args.backbone_model, 
                            problem_type=self.hparams.problem_type, 
                            num_labels = self.hparams.num_class)
        
    def forward(self, inputs, labels=None):
        return self.model(**inputs, labels=labels)
    
    def training_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }

        outputs = self.forward(inputs, labels=batch["label"])
        loss = outputs.loss
        logits = outputs.logits

        self.log('train_step_loss', loss, prog_bar=True, logger=True)
        return {"loss": loss, "logits":logits, "label": batch["label"]}
    

    def training_epoch_end(self, training_step_outputs):
        loss, y_trues, y_preds, y_scores = [], [], [], []
        for t in training_step_outputs:
            loss.append(t["loss"])
            for i in t["label"]:
                y_trues.append(i.cpu().numpy())
            for l in t["logits"]:
                y_preds.append(torch.where(torch.softmax(l, dim=0) > 0.5, 1, 0).cpu().numpy())
                y_scores.append([j.item() for j in l])
        
        avg_loss = torch.mean(torch.tensor(loss))
        
        metrics = multilabel_evaluate(mode='train', y_true=y_trues, y_pred=y_preds, y_score=y_scores)
        metrics["train_epoch_loss"] = avg_loss
        
        self.log_dict(metrics, logger=True)


    def validation_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }

        outputs = self.forward(inputs, labels=batch["label"])
        loss = outputs.loss
        logits = outputs.logits
        
        return {"loss": loss, "logits":logits, "label": batch["label"]}


    def validation_epoch_end(self, valid_step_outputs):
        loss, y_trues, y_preds, y_scores = [], [], [], []
        for v in valid_step_outputs:
            loss.append(v["loss"])
            for i in v["label"]:
                y_trues.append(i.cpu().numpy())
            for l in v["logits"]:
                y_preds.append(torch.where(torch.softmax(l, dim=0) > 0.5, 1, 0).cpu().numpy())
                y_scores.append(l.cpu().numpy())
        
        avg_loss = torch.mean(torch.tensor(loss))
        
        metrics = multilabel_evaluate(mode='val', y_true=y_trues, y_pred=y_preds, y_score=y_scores)
        metrics["val_loss"] = avg_loss

        self.log_dict(metrics, logger=True)


    def test_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        outputs = self.forward(inputs, labels=batch["label"])

        label = batch['label']
        doc_key = batch['doc_key']  # list
        evt_type = batch['evt_type']  # list
        sentence = batch['sentence']  # list
        return (doc_key, evt_type, outputs.logits, label, sentence)


    def test_epoch_end(self, outputs):
        doc_keys = []
        evt_types = []
        y_trues = []
        y_preds = []
        y_scores = []

        with open('checkpoints/{}/predictions.jsonl'.format(self.hparams.ckpt_name), 'w', encoding='utf-8') as writer:
            for tup in outputs:
                for idx in range(len(tup[0])):
                    pred = {
                        'doc_key': tup[0][idx],
                        'y_true': tup[3][idx].cpu().numpy().tolist(),
                        'y_score': tup[2][idx].cpu().numpy().tolist(),
                        'y_pred': torch.where(torch.softmax(tup[2][idx], dim=-1) > 0.5, 1, 0).cpu().numpy().tolist(),
                        'evt_type': tup[1][idx],
                        'sentence': tup[4][idx]
                    }
                    writer.write(json.dumps(pred, ensure_ascii=False)+'\n')

                    y_preds.append(pred['y_pred'])
                    y_scores.append(pred['y_score'])
                    y_trues.append(pred['y_true'])
        
        if not self.hparams.inference:
            metrics = multilabel_evaluate(mode='test', y_true=y_trues, y_pred=y_preds, y_score=y_scores)
            self.log_dict(metrics, logger=True)
            

    def get_lr_scheduler(self, lr_scheduler, optimizer, warmup_steps, total_steps):
        get_schedule_func = arg_to_scheduler[lr_scheduler]
        _scheduler = get_schedule_func(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        scheduler = {
            "scheduler": _scheduler, 
            "interval": "step", 
            "monitor": "val_loss",
            "name": lr_scheduler
        }
        return scheduler

    def configure_optimizers(self):
        self.train_len = len(self.trainer.datamodule.train_dataloader())
        if self.hparams.max_steps > 0:
            total_steps = self.hparams.max_steps
            self.hparams.num_train_epochs = self.hparams.max_steps // self.train_len // self.hparams.accumulate_grad_batches + 1
        else:
            total_steps = self.train_len // self.hparams.accumulate_grad_batches * self.hparams.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if self.hparams.adafactor:
            print("using adafactor.")
            optimizer = Adafactor(
                optimizer_grouped_parameters, 
                lr=self.hparams.learning_rate, 
                scale_parameter=False, 
                relative_step=False
            )
        else:
            print("using AdamW.")
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon
            )

        # scheduler is called only once per epoch by default
        warmup_steps = self.hparams.warmup_ratio * total_steps

        print('*'*50)
        logger.info('warmup_steps: {} with {} training steps in total.. '.format(warmup_steps, total_steps))

        scheduler = self.get_lr_scheduler(self.hparams.lr_scheduler, optimizer, warmup_steps, total_steps)

        return [optimizer, ], [scheduler, ]