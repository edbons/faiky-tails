import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import torch
from torch.utils.data import DataLoader

import transformers
from src.model.data_full import FullDataset, RawFilesDataset
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.model.pipeline import train_eval_loop, init_random_seed
import datetime
import pickle


def run_batch(batch, model, device):
        input_ids, mask, label = batch['sample'], batch['mask'], batch['label']
        input_ids = input_ids.to(device)
        mask = mask.to(device)
        label = label.to(device)
        outputs = model(input_ids, attention_mask=mask, labels=label)
        return outputs


def train_epoch(model, loader, test_loader, optimizer, epoch_num, device, log_interval=10, checkpoint_path=None, accum_iter=2, desc=None):
    losses = []
    avg_loss = []
    step = 1
    train_bar = tqdm(iterable=loader, desc=desc)
    for i, batch in enumerate(train_bar):
        outputs = run_batch(batch, model, device)
        loss, _ = outputs[:2]
        avg_loss.append(loss.detach().item())
        loss.backward()
        if i % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
        if step % log_interval == 0:
            val_loss = sum(avg_loss) / len(avg_loss)
            losses.append(val_loss)
            avg_loss = []            
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 
            os.path.join(checkpoint_path, "checkpoint.pt"))
            train_bar.set_postfix(loss=val_loss)
            # print('epoch {}\t[{}/{}]\tloss = {:.4f}'.format(epoch_num, step, len(loader), val_loss))         
        step += 1
    
    if losses:
        return sum(losses) / len(losses)
    else:
        return sum(avg_loss) / len(avg_loss)


# def get_average_scores(hyps: List[str], refs: List[str]):       
#     rouge_scorer = rouge.Rouge()
#     averaged_scores = rouge_scorer.get_scores(hyps, refs, avg=True)
#     return averaged_scores

# def generate_story(batch: dict, model: GPT2LMHeadModel, text_encoder: GPT2Tokenizer, device: str, beam: int, k: int, p: float, repetition_penalty: float, n_ctx: int=100, gen_len: int=1024, gen_temperature: float=0.9):
#     ctx_strs, tgt_strs, gen_strs = [], [], []

#     input_ids, mask = batch['sample'], batch['mask']
#     septok = text_encoder.convert_tokens_to_ids('[SEP]')
    
#     #TO-DO divide batch to kw and tartget
#     sep_idx = torch.where(input_ids == septok)
#     keyword_ids = input_ids[:, :sep_idx + 1] # + BOS + __kw__ + cls
#     target_toks = input_ids[:, sep_idx + 1:]

#     # keyword_ids = input_ids[:, :n_ctx + 1 + 1 + 1] # + BOS + __kw__ + cls
#     # target_toks = input_ids[:, n_ctx + 1 + 1 + 1:]

#     # mask = torch.ones(keyword_ids.size()).type_as(mask)

#     keyword_ids = keyword_ids.to(device)
#     target_toks = target_toks.to(device)
#     # mask = mask.to(device) 

#     gen_params = {'input_ids': keyword_ids,
#                 'do_sample': True,                
#                 'num_beams': beam,
#                 'temperature': gen_temperature,
#                 'max_length': gen_len,
#                 'min_length': 20,
#                 'top_k': k,
#                 'top_p': p,
#                 'repetition_penalty': repetition_penalty,
#                 'bos_token_id': text_encoder.bos_token_id,
#                 'pad_token_id': text_encoder.pad_token_type_id,  # text_encoder.pad_token_type_id
#                 'eos_token_id': text_encoder.eos_token_id,
#                 # 'attention_mask': mask,
#                 # 'decoder_start_token_id': text_encoder.cls_token_id
#                 }
    
#     outputs = model.generate(**gen_params)
#     generated_batch = outputs[:, n_ctx + 1 + 1 + 1:]
    
#     for i, generated_toks in enumerate(generated_batch):
#         ctx_str = text_encoder.decode(keyword_ids[i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
#         ctx_strs.append(ctx_str)
#         tgt_str = text_encoder.decode(target_toks[i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
#         tgt_strs.append(tgt_str)
#         gen_str = text_encoder.decode(generated_toks, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#         gen_strs.append(gen_str)
    
#     return ctx_strs, tgt_strs, gen_strs


def evaluate(val_loader: DataLoader, 
            model: GPT2LMHeadModel, 
            device: str, 
            show_progress=True):
    
    avg_loss = []
    
    if show_progress:
        eval_bar = tqdm(iterable=val_loader, desc="Evaluate")
    else:
        eval_bar = val_loader
    
    for batch in eval_bar:
        with torch.no_grad():
            outputs = run_batch(batch, model, device)
            loss, _ = outputs[:2]
            avg_loss.append(loss.detach().item())
    
    return sum(avg_loss) / len(avg_loss)

def init(args):
    print("Creating directories")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.experiment_name), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.experiment_name), exist_ok=True)
    save_dir = os.path.join(args.output_dir, args.experiment_name, "checkpoints")
    log_dir = os.path.join(args.output_dir, args.experiment_name, "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    init_random_seed(args.seed)


def main(args: argparse.ArgumentParser):
    init(args)

    save_dir = os.path.join(args.output_dir, args.experiment_name, "checkpoints")
    model_name = args.hf_model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_encoder = GPT2Tokenizer.from_pretrained(model_name, add_prefix_space=True)
    
    if args.dataset == 'tails':
        text_encoder.add_special_tokens({'bos_token':'_start_',
                                            'cls_token':'_classify_',
                                            'eos_token':'_end_',
                                            'additional_special_tokens': ['_kw_', '_endkw_']
                                        })
    elif args.dataset == 'all':
        text_encoder.add_special_tokens({'bos_token': '<s>',                                     
                                     'eos_token': '</s>',
                                     'additional_special_tokens': ['[SEP]', '_kw_', '_endkw_']
                                    })

    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # text_encoder.pad_token = text_encoder.eos_token
    # model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(text_encoder))

    if args.dataset == 'tails':
        train_dataset = FullDataset(os.path.join(args.data_dir, 'train_full'), text_encoder, args.pad_len, max_samples=args.max_samples, n_ctx=args.n_ctx)
        val_dataset = FullDataset(os.path.join(args.data_dir, 'val_full'), text_encoder, args.pad_len, max_samples=args.max_samples, n_ctx=args.n_ctx)

    elif args.dataset == 'all':
        corpus1_path = 'dataset/raw'
        corpus2_path = 'dataset/raw_other'
        corpus1_files = [os.path.join(corpus1_path, name) for name in os.listdir(corpus1_path)]
        corpus2_files = [os.path.join(corpus2_path, name) for name in os.listdir(corpus2_path)]     
        train, val_test = train_test_split(corpus1_files, test_size=0.4)
        val, test = train_test_split(val_test, test_size=0.5)
 
        with open(os.path.join(args.output_dir, args.experiment_name, 'test_dataset'), 'wb') as f:
            pickle.dump(test, file=f)
        

        train.extend(corpus2_files)
        train_dataset = RawFilesDataset(data_files=train, tokenizer=text_encoder, pad_len=args.pad_len, max_samples=args.max_samples, n_ctx=args.n_ctx)
        val_dataset = RawFilesDataset(data_files=val, tokenizer=text_encoder, pad_len=args.pad_len, max_samples=args.max_samples, n_ctx=args.n_ctx)


    scheduler = lambda optim: \
    torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5, verbose=True)

    # for epoch in range(args.num_epochs):
    #     ep_loss = train_epoch(model, train_loader, val_loader, optimizer, epoch, device, log_interval=args.train_log_interval, checkpoint_path=save_dir, accum_iter=args.accum_iter, desc="FT Training Epoch [{}/{}]".format(epoch + 1, args.num_epochs))        
    #     val_loss = evaluate(val_loader, model, text_encoder, device=device)
    #     print(f"{epoch} train loss: {ep_loss}, val loss: {val_loss}")

    best_val_loss, best_model = train_eval_loop(model=model,
                                            train_dataset=train_dataset,
                                            val_dataset=val_dataset,                        
                                            lr=args.lr,
                                            epoch_n=args.num_epochs,
                                            batch_size=args.n_batch,
                                            l2_reg_alpha=0,
                                            lr_scheduler_ctor=scheduler,
                                            early_stopping_patience=3)

    torch.save(best_model, os.path.join(save_dir, "checkpoint.pt"))
    
    with open (os.path.join(args.output_dir, args.experiment_name,'results.txt'), 'w', encoding='utf-8') as f:
        print('timestamp','experiment', 'model', 'dataset', 'val_loss', file=f, sep='\t')
        print(str(datetime.datetime.today()), args.experiment_name, args.hf_model, args.dataset, best_val_loss, file=f, sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--n_batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    # parser.add_argument('--b1', type=float, default=0.9)
    # parser.add_argument('--b2', type=float, default=0.999)
    # parser.add_argument('--e', type=float, default=1e-8)    
    parser.add_argument('--output_dir', type=str, default='savedir', help='directory to save logs and checkpoints to')
    parser.add_argument('--data_dir', type=str, default='dataset/full', help='directory with train, dev, test files')
    parser.add_argument('--experiment_name', type=str, required=True, help='name of this experiment will be included in output')
    # parser.add_argument('--train_log_interval', type=int, default=10, help='number of train steps before logging training progress')
    # parser.add_argument('--val_log_interval', type=int, default=10, help='number of train steps before logging validation progress')
    # parser.add_argument('--beam', type=int, default=1, help='beam size for beam search')
    # parser.add_argument('--k', type=int, default=0, help='k for TopK sampling')
    # parser.add_argument('--p', type=float, default=0.9, help='p for Nucleus sampling')
    # parser.add_argument('--temperature', type=float, default=0.7, help='temperature for text generation')
    # parser.add_argument('--accum_iter', type=int, default=2, help='number of batches to accumulate gradients before doing backprop')
    parser.add_argument('--gen_len', type=int, default=512, help='max generation length + 1 for end token')
    parser.add_argument('--pad_len', type=int, default=1024, help='max input length')
    parser.add_argument('--n_ctx', type=int, default=70, help='keyword tokens length')
    parser.add_argument('--max_samples', type=int, default=None, help='limit dataset')
    parser.add_argument('--show_progress', action='store_true')    
    # parser.add_argument('--repeattheta', type=float, default=1.4, help='how much to penalize repitition (1 is not at all, > 1 is more penalty)')    
    parser.add_argument('--checkpoint', type=str, default=None, help='location of a previous checkpoint')
    parser.add_argument('--hf_model', type=str, default="sberbank-ai/rugpt3small_based_on_gpt2", help='name for GPT2 or GPT3 model from Hugginface')
    parser.add_argument('--dataset', type=str, default="tails", help='type of dataset: tails/all')

    
    args = parser.parse_args()
    print(transformers.__version__)
    print(args)
    main(args)