import os
import argparse
import pickle
import csv

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
from src.model.data_full import RawFilesDataset, PromtDataset
from src.model.pipeline import copy_data_to_device, init_random_seed
from typing import List, Tuple, Union
import warnings
    
warnings.simplefilter("ignore")


def clean_text(text: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(text, str):
        return text.replace('\r\n', ' ').replace('\n', ' ').replace('<s>', '').replace('</s>', '').replace('<pad>', '').replace('_kw_', '').replace('_endkw_', '').replace('[SEP]', '').strip() 
    elif isinstance(text, list):
        texts = [item.replace('\r\n', ' ').replace('\n', ' ').replace('<s>', '').replace('</s>', '').replace('<pad>', '').replace('_kw_', '').replace('_endkw_', '').replace('[SEP]', '').strip() for item in text]
        return texts


def write_stories(data: tuple, path: str='', label: str=""):
    columns = ['context', 'refs', 'hyps']
    assert len(columns) == len(data[0])    
    with open(os.path.join(path, f'gen_{label}.csv'), 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='|', lineterminator='\n')
        writer.writerow(columns)
        writer.writerows(data)

class StoryGenerator:
    def __init__(self, tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel, device: str='cpu') -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def generate_stories(self, data: str="", n_ctx: int=100, batch_size: int=2, max_samples: int=None, gen_len: int=512, **args) -> List[Tuple[str, str, str]]:        

        # dataset = RawFilesDataset(data, self.tokenizer, pad_len=2048, n_ctx=n_ctx, max_samples=max_samples)
        dataset = PromtDataset(data, self.tokenizer, pad_len=2048, n_ctx=n_ctx, max_samples=max_samples)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.n_ctx = n_ctx
        self.params = args        

        result = []
    
        for batch in tqdm(loader): 
            
            context, refs, hyps = self.__generate_batch(batch=batch, gen_len=gen_len)
            result.extend([item for item in zip(context, refs, hyps)])            

        return result

    def __generate_batch(self, batch: dict, gen_len: int=512)-> Tuple[str, str, str]:
        septok = self.tokenizer.convert_tokens_to_ids('[SEP]')
        endtok = self.tokenizer.eos_token_id
        input_ids, mask = batch['sample'], batch['mask'] 
        
        context = input_ids[:, :self.n_ctx]
        ctx_mask = mask[:, :self.n_ctx]       

        target = input_ids[:, self.n_ctx:gen_len + self.n_ctx]
        context = copy_data_to_device(context, self.device)
        ctx_mask = copy_data_to_device(ctx_mask, self.device)

        sample_output = self.model.generate(
                                        context,
                                        attention_mask=ctx_mask,                                                     
                                        max_length=gen_len + self.n_ctx, 
                                        do_sample=True,                     
                                        eos_token_id=endtok,
                                        bos_token_id=self.tokenizer.bos_token_id,
                                        decoder_start_token_id=septok,
                                        pad_token_id=0,
                                        min_length=gen_len,
                                        num_return_sequences=1,                   
                                        no_repeat_ngram_size=3, 
                                        forced_eos_token_id = endtok,
                                        early_stopping=True,  
                                        **self.params
                                    )
        
        context_txt = self.tokenizer.batch_decode(context, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        refs = self.tokenizer.batch_decode(target, skip_special_tokens=False, clean_up_tokenization_spaces=False)        
        hyps = self.tokenizer.batch_decode(sample_output[:, self.n_ctx:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        
        return context_txt, refs, hyps

def init(args):
    init_random_seed(args.seed)
    if args.experiment_name == 'baseline':
        print("Creating directories")
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, args.experiment_name), exist_ok=True)
        

def main(args: argparse.ArgumentParser):
    init(args)
    output_dir = os.path.join(args.output_dir, args.experiment_name)    

    text_encoder = GPT2Tokenizer.from_pretrained(args.hf_model, add_prefix_space=True)
    text_encoder.add_special_tokens({'bos_token': '<s>',                                     
                                        'eos_token': '</s>',
                                        'additional_special_tokens': ['[SEP]', '_kw_', '_endkw_']
                                    })

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    if args.experiment_name == 'baseline':
        model = model = GPT2LMHeadModel.from_pretrained(args.hf_model)    
        model.resize_token_embeddings(len(text_encoder))
    else:
        with open(os.path.join(output_dir,'checkpoints/checkpoint.pt'), 'rb') as f:
            model = torch.load(f, map_location=device)

    postfix = ""
    if args.use_ner:
        postfix = "_ner" 
    
    test = os.path.join(args.data_dir, 'test' + postfix)
    
    params = {'num_beams': args.num_beams,
                'top_p': args.p,
                'top_k': args.k,
                'temperature': args.temperature,
                'repetition_penalty': args.repeattheta
            }

    generator = StoryGenerator(tokenizer=text_encoder, model=model, device=device)
    data = generator.generate_stories(test, 
                                    n_ctx=args.n_ctx, 
                                    batch_size=args.n_batch, 
                                    max_samples=args.max_samples, 
                                    gen_len=args.gen_len,
                                    **params)
    
    data = [ ( clean_text(item[0]), clean_text(item[1]), clean_text(item[2]) ) for item in data]
    label = f'beams{args.num_beams}_p{str(args.p).replace(".","")}_k{args.k}_t{str(args.temperature).replace(".","")}_rep{str(args.repeattheta).replace(".","")}'
    write_stories(data, output_dir, label)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='dataset/full', help='directory with train, dev, test files')
    parser.add_argument('--output_dir', type=str, default='savedir', help='directory to save logs and checkpoints to')
    parser.add_argument('--experiment_name', type=str, default='baseline', required=True, help='name of this experiment will be included in output. Value "baseline" is for generate with original hf_model (without checkpoints)')
    parser.add_argument('--num_beams', type=int, default=5, help='beam size for beam search')
    parser.add_argument('--k', type=int, default=3, help='k for TopK sampling')
    parser.add_argument('--p', type=float, default=0.95, help='p for Nucleus sampling')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for text generation')
    parser.add_argument('--repeattheta', type=float, default=2.0, help='how much to penalize repitition (1 is not at all, > 1 is more penalty)')  
    parser.add_argument('--gen_len', type=int, default=512, help='max generation length + 1 for end token')
    parser.add_argument('--n_ctx', type=int, default=70, help='keyword tokens length')  
    parser.add_argument('--hf_model', type=str, default="sberbank-ai/rugpt3small_based_on_gpt2", help='name for GPT2 or GPT3 model from Hugginface')
    parser.add_argument('--n_batch', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None, help='limit dataset')
    parser.add_argument('--use_ner', action='store_true')  
    args = parser.parse_args()
    print(args)
    main(args)