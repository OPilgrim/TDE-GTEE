import json
import logging

import pytorch_lightning as pl
import torch
from transformers import (
    AdamW, 
    AutoConfig, 
    BartTokenizer,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    Adafactor,
)
from .network import BartGen
from .const import *
from .scorer import evaluate, inference

logger = logging.getLogger(__name__)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup
}

class GenIEModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.hparams.update(vars(args))
        
        self.config = AutoConfig.from_pretrained(args.backbone_model)
        self.tokenizer = BartTokenizer.from_pretrained(args.backbone_model)
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        if self.hparams.model == 'gen' or self.hparams.model == 'constrained-gen':
            self.model = BartGen(self.config, self.hparams, self.tokenizer)
            self.model.resize_token_embeddings()
        else:
            raise NotImplementedError

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch["input_token_ids"],
            "attention_mask": batch["input_attn_mask"],
            "decoder_input_ids": batch['tgt_token_ids'],
            "decoder_attention_mask": batch["tgt_attn_mask"],
            "task": 0
        }

        outputs = self.forward(inputs)
        loss = outputs[0]
        loss = torch.mean(loss)

        log = {
            'train/loss': loss,
        }
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch["input_token_ids"],
            "attention_mask": batch["input_attn_mask"],
            "decoder_input_ids": batch['tgt_token_ids'],
            "decoder_attention_mask": batch["tgt_attn_mask"],
            "task": 0,
        }
        outputs = self.forward(inputs)
        loss = outputs[0]
        loss = torch.mean(loss)

        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.stack(outputs))

        self.log('val/loss', avg_loss)

    def test_step(self, batch, batch_idx):
        if self.hparams.sample_gen:
            pred_token_ids = self.model.generate(
                batch['input_token_ids'],
                do_sample=True,
                top_k=20,
                top_p=0.99,
                max_length=MAX_TGT_LENGTH[self.hparams.dataset],
                num_return_sequences=1,
                num_beams=NUM_BEAM,
            )
        else:
            pred_token_ids = self.model.generate(
                batch['input_token_ids'],
                do_sample=False,
                max_length=MAX_TGT_LENGTH[self.hparams.dataset],
                num_return_sequences=1,
                num_beams=NUM_BEAM,
            )

        pred_token_ids = pred_token_ids.reshape(
            batch['input_token_ids'].size(0), 
            1, 
            -1
        )
        doc_key = batch['doc_key']  # list
        evt_type = batch['evt_type']  # list
        tokens = batch['tokens']  # list
        src_token_ids = batch['input_token_ids']
        tgt_token_ids = batch['tgt_token_ids']

        return (doc_key, pred_token_ids, tgt_token_ids, src_token_ids, tokens, evt_type)

    def test_epoch_end(self, outputs):
        doc_keys = []
        evt_types = []
        pred_strs = []
        text_tokens = []
        dataset = self.hparams.dataset
        dataset_split_name = 'test'
        gold_file_path = self.hparams.test_file

        # dump predictions
        with open('checkpoints/{}/predictions.jsonl'.format(self.hparams.ckpt_name), 'w') as writer:
            for tup in outputs:
                for idx in range(len(tup[0])):
                    pred = {
                        'doc_key': tup[0][idx],
                        'pred': self.tokenizer.decode(tup[1][idx].squeeze(0), skip_special_tokens=True),
                        'gold': self.tokenizer.decode(tup[2][idx].squeeze(0), skip_special_tokens=True) if not self.hparams.inference else None,
                        'tokens': tup[4][idx],
                        'evt_type': tup[5][idx]
                    }
                    writer.write(json.dumps(pred)+'\n')

                    doc_keys.append(pred['doc_key'])
                    evt_types.append(pred['evt_type'])
                    pred_strs.append(pred['pred'])
                    text_tokens.append(pred['tokens'])

        if self.hparams.inference:
            result = inference(
                doc_keys, 
                evt_types, 
                pred_strs, 
                text_tokens,
                dataset,
                self.hparams,
            )
            with open('checkpoints/{}/inferences.json'.format(self.hparams.ckpt_name), 'w', encoding='utf-8') as w:
                w.write(json.dumps(result))
            return

        if self.hparams.az_file:
            print('Evaluating with AZ models...')
        else:
            print('Evaluating without AZ models...')
            print('To get final scores, use "local_evaluate.py"')

        # evaluate F1
        trigger_idn_result, trigger_cls_result, role_idn_result, role_cls_result = evaluate(
            doc_keys, 
            evt_types,
            pred_strs, 
            text_tokens, 
            dataset, 
            dataset_split_name, 
            gold_file_path,
            self.hparams,
        )

        for answer_dict in [trigger_idn_result, trigger_cls_result, role_idn_result, role_cls_result]:
            for key, value in answer_dict.items():
                self.log(key, value)


    def get_lr_scheduler(self, lr_scheduler, optimizer, warmup_steps, total_steps):
        get_schedule_func = arg_to_scheduler[lr_scheduler]
        scheduler = get_schedule_func(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        scheduler = {
            "scheduler": scheduler, 
            "interval": "step", 
            "monitor": "val/loss",
        }
        return scheduler

    def configure_optimizers(self):
        self.train_len = len(self.trainer.datamodule.train_dataloader())
        if self.hparams.max_steps > 0:
            total_steps = self.hparams.max_steps
            self.hparams.num_train_epochs = self.hparams.max_steps // self.train_len // self.hparams.accumulate_grad_batches + 1
        else:
            total_steps = self.train_len // self.hparams.accumulate_grad_batches * self.hparams.num_train_epochs

        logger.info('{} training steps in total.. '.format(total_steps))

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
            optimizer = Adafactor(
                optimizer_grouped_parameters, 
                lr=self.hparams.learning_rate, 
                scale_parameter=False, 
                relative_step=False
            )
        else:
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon
            )

        # scheduler is called only once per epoch by default
        warmup_steps = self.hparams.warmup_ratio * total_steps
        
        print('*'*50)
        print('warmup_steps: ', warmup_steps)
        print('total_steps: ', total_steps)
        scheduler = self.get_lr_scheduler(self.hparams.lr_scheduler, optimizer, warmup_steps, total_steps)

        return [optimizer, ], [scheduler, ]