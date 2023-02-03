import argparse
from ast import arg
import logging
import os
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument( "--dataset", default='ACE', type=str, choices=['ACE', 'ERE'])
    parser.add_argument( "--refresh", action='store_true', default=False, help='Reprocessing datasets even if the processed version exists.')
    parser.add_argument( '--tmp_dir', type=str)
    parser.add_argument( "--ckpt_name", default=None, type=str, help="The output directory where the model checkpoints and predictions will be written.",)
    parser.add_argument( "--load_ckpt", default=None, type=str,)
    parser.add_argument( "--train_file", default='data/ACE05-E/train.oneie.json', type=str, help="The input training file. If a data dir is specified, will look for the file there. If no data dir or train/predict files are specified, will run with tensorflow_datasets.",)
    parser.add_argument( "--val_file", default='data/ACE05-E/dev.oneie.json', type=str, help="The input evaluation file. If a data dir is specified, will look for the file there. If no data dir or train/predict files are specified, will run with tensorflow_datasets.",)
    parser.add_argument( '--test_file', type=str, default='data/ACE05-E/test.oneie.json',)
    parser.add_argument( '--inference', default=False, action="store_true", help="Whether or not to inference.")
    parser.add_argument( '--negative_sampling', action='store_true', default=False, help='use negative samples when training and validating, while always use all negative samples when testing')
    parser.add_argument( '--negative_ratio', type=float, default=1.0, help='negative_ratio')
    parser.add_argument( "--train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument( "--eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument( "--eval_only", action="store_true",)
    parser.add_argument( "--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument( "--accumulate_grad_batches", type=int, default=8, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument( "--weight_decay", default=1e-5, type=float, help="Weight decay if we apply some.")
    parser.add_argument( "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument( "--gradient_clip_val", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument( "--num_train_epochs", default=200, type=int, help="Total number of training epochs to perform.")
    parser.add_argument( "--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
    parser.add_argument( "--warmup_ratio", default=0.1, type=float, help="Linear warmup over warmup_ratio * total_steps.")
    parser.add_argument( "--gpus", default=-1, help='-1 means train on all gpus')
    parser.add_argument( "--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument( "--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    
    parser.add_argument( "--bert_learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument( '--backbone_model', type=str, default='roberta-large', choices=['bert-large-uncased','bert-large-uncased-whole-word-masking', 'bert-large-cased-whole-word-masking', 'roberta-large'])
    parser.add_argument( "--lr_scheduler", default="linear", type=str, choices=['linear', 'cosine', 'cosine_w_restarts', 'polynomial'])
    parser.add_argument( "--adafactor", default=False, action="store_true")
    parser.add_argument( "--NSP", default=False, action="store_true")
    parser.add_argument( "--problem_type", default="multi_label_classification", type=str, choices=['multi_label_classification'])
    parser.add_argument( "--num_class", default=33, type=int, help="The classes of event type.")
    parser.add_argument( "--debug", default=False, action="store_true", help="")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Set seed
    seed_everything(args.seed)

    if not args.ckpt_name:
        d = datetime.now()
        time_str = d.strftime('%m-%dT%H%M')
        args.ckpt_name = '{}_{}lr{}_{}'.format(
            args.model,
            args.train_batch_size * args.accumulate_grad_batches,
            args.learning_rate,
            time_str
        )

    os.makedirs('./checkpoints', exist_ok=True)
    args.ckpt_dir = os.path.join(f'./checkpoints/{args.ckpt_name}')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if not args.tmp_dir:
        args.tmp_dir = os.path.join(f'{args.ckpt_dir}/preprocessed')

    logger.info("Training/evaluation parameters %s", args)

    lr_logger = LearningRateMonitor(logging_interval='step')
    tb_logger = TensorBoardLogger(os.path.join(f'{args.ckpt_dir}/logs'), name='my_model')

    if args.max_steps < 0:
        args.max_epochs = args.min_epochs = args.num_train_epochs
    
    if args.problem_type == "multi_label_classification":
        from src.mlc.OneIE_data_module_az import DataModule
        from src.mlc.model import ICModel
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.ckpt_dir,
            save_top_k=2,
            monitor='val_loss',
            mode='min',
            save_weights_only=True,
            filename='{epoch}',  # this cannot contain slashes
        )

    model = ICModel(args)
    dm = DataModule(args)

    if args.debug:
        limit_train_batches, limit_val_batches, limit_test_batches=0.1, 0.2, 0.3
    else:
        limit_train_batches, limit_val_batches, limit_test_batches=1.0, 1.0, 1.0

    trainer = Trainer(
        logger=tb_logger,
        min_epochs=args.num_train_epochs//2,
        max_epochs=args.num_train_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        num_sanity_val_steps=2,
        val_check_interval=0.5,  # use float to check every n epochs
        precision=16 if args.fp16 else 32,
        callbacks=[lr_logger, checkpoint_callback, ],
        gpus=args.gpus,
        #strategy="ddp",
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )

    if args.load_ckpt:
        model.load_state_dict(torch.load(
            args.load_ckpt,
            map_location=model.device
        )['state_dict'])

    if args.eval_only:
        dm.setup('test')
        trainer.test(model, datamodule=dm)  # also loads training dataloader
    else:
        dm.setup('fit')
        trainer.fit(model, dm)


if __name__ == "__main__":
    main()
