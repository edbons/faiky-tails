import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

import transformers
from src.model.data_full import PromtDataset
import argparse
from src.model.pipeline import train_eval_loop, init_random_seed
import datetime


def init(args):
    print("Creating directories")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.experiment_name), exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, args.experiment_name, "checkpoints")
    log_dir = os.path.join(args.output_dir, args.experiment_name, "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    init_random_seed(args.seed)
    return checkpoint_dir, log_dir


def main(args: argparse.ArgumentParser):
    checkpoint_dir, log_dir = init(args)

    exp_dir = os.path.join(args.output_dir, args.experiment_name)

    model_name = args.hf_model
    postfix = ""
    if args.use_ner:
        postfix = "_ner" 

    text_encoder = GPT2Tokenizer.from_pretrained(model_name, add_prefix_space=True)
    
    text_encoder.add_special_tokens({'bos_token': '<s>',                                     
                                     'eos_token': '</s>',
                                     'additional_special_tokens': ['[SEP]', '_kw_', '_endkw_']
                                    })

    model = GPT2LMHeadModel.from_pretrained(model_name)    
    model.resize_token_embeddings(len(text_encoder))

    train_dataset = PromtDataset(data_file=os.path.join(args.data_dir, 'train' + postfix), tokenizer=text_encoder, pad_len=args.pad_len, max_samples=args.max_samples, n_ctx=args.n_ctx)
    val_dataset = PromtDataset(data_file=os.path.join(args.data_dir, 'val' + postfix), tokenizer=text_encoder, pad_len=args.pad_len, max_samples=args.max_samples, n_ctx=args.n_ctx)

    scheduler = lambda optim: \
    torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5, verbose=True)

    best_val_loss, best_model = train_eval_loop(model=model,
                                            train_dataset=train_dataset,
                                            val_dataset=val_dataset,                        
                                            lr=args.lr,
                                            epoch_n=args.num_epochs,
                                            batch_size=args.n_batch,
                                            l2_reg_alpha=0,
                                            lr_scheduler_ctor=scheduler,
                                            early_stopping_patience=2,
                                            log_dir=log_dir)

    torch.save(best_model, os.path.join(checkpoint_dir, "checkpoint.pt"))
    
    with open (os.path.join(exp_dir,'results.txt'), 'w', encoding='utf-8') as f:
        print('timestamp','experiment', 'model', 'use_ner', 'val_loss', file=f, sep='\t')
        print(str(datetime.datetime.today()), args.experiment_name, args.hf_model, args.use_ner, best_val_loss, file=f, sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--n_batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=6.25e-5)   
    parser.add_argument('--output_dir', type=str, default='savedir', help='directory to save logs and checkpoints to')
    parser.add_argument('--data_dir', type=str, default='dataset/full', help='directory with train, dev, test files')
    parser.add_argument('--experiment_name', type=str, required=True, help='name of this experiment will be included in output')
    parser.add_argument('--gen_len', type=int, default=512, help='max generation length + 1 for end token')
    parser.add_argument('--pad_len', type=int, default=1024, help='max input length')
    parser.add_argument('--n_ctx', type=int, default=70, help='keyword tokens length')
    parser.add_argument('--hf_model', type=str, default="sberbank-ai/rugpt3small_based_on_gpt2", help='name for GPT2 or GPT3 model from Hugginface')
    parser.add_argument('--max_samples', type=int, default=None, help='limit samples count in dataset')
    parser.add_argument('--use_ner', action='store_true', help='Use dataset with NER promt')  

    
    args = parser.parse_args()
    print(transformers.__version__)
    print(args)
    main(args)