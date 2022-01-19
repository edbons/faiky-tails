import os
import argparse
import pickle
import csv

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
from data_full import RawFilesDataset
from pipeline import copy_data_to_device
from typing import List, Tuple


def flat_text(text: str="") -> str:
    return text.replace('\r\n',' ').replace('\n',' ').strip()


def write_stories(data: tuple, path: str=''):
    columns = ['context', 'refs', 'hyps']
    assert len(columns) == len(data[0])    
    with open(os.path.join(path, 'generated_stories.csv'), 'w', encoding='utf-8') as f:
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

    def generate_stories(self, data: list, n_ctx: int=100, batch_size: int=2, max_iter: int=None, gen_len: int=512) -> List[Tuple[str, str, str]]:        

        dataset = RawFilesDataset(data, self.tokenizer, pad_len=2048, n_ctx=n_ctx)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)        

        result = []
    
        for i, batch in tqdm(enumerate(loader)):
            if max_iter is not None and i==max_iter:
                break       
            
            context, refs, hyps = self.__generate_batch(batch=batch, gen_len=gen_len)
            result.append( (context, refs, hyps) )            

        return result

    def __generate_batch(self, batch: dict, gen_len: int=512)-> Tuple[str, str, str]:
        septok = self.tokenizer.convert_tokens_to_ids('[SEP]')
        endtok = self.tokenizer.eos_token_id
        input_ids = batch['sample']
        input_ids = copy_data_to_device(input_ids, self.device)

        sep_idx = torch.where(input_ids[0] == septok)[0].item()
        eos_idx = torch.where(input_ids[0] == endtok)[0][0].item()
        context = input_ids[:, :sep_idx+1]
        target_txt = input_ids[:, sep_idx+1:eos_idx+1]

        context_txt = self.tokenizer.decode(context[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        refs = self.tokenizer.decode(target_txt[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        sample_output = self.model.generate(
                                        context,                     
                                        max_length=gen_len, 
                                        do_sample=True,
                                        num_beams = 20,  # https://arxiv.org/pdf/2108.03502.pdf 
                                        top_p=0.95, # https://arxiv.org/pdf/2108.03502.pdf 
                                        top_k=3, # https://arxiv.org/pdf/2108.03502.pdf
                                        eos_token_id=endtok,
                                        bos_token_id=self.tokenizer.bos_token_id,
                                        decoder_start_token_id = septok,
                                        pad_token_id=endtok,
                                        min_length = 100,
                                        num_return_sequences=1, 
                                        temperature=1.0, # https://arxiv.org/pdf/2108.03502.pdf
                                        repetition_penalty=2.0,  # https://arxiv.org/pdf/2108.03502.pdf
                                        no_repeat_ngram_size=3, # https://arxiv.org/pdf/2108.03502.pdf
                                        forced_eos_token_id = endtok,
                                        early_stopping=True  # https://arxiv.org/pdf/2108.03502.pdf
                                    )
        hyps = self.tokenizer.decode(sample_output[0][sep_idx+1:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        
        return context_txt, refs, hyps


def main(args: argparse.ArgumentParser):
    output_dir = os.path.join(args.output_dir, args.experiment_name)

    text_encoder = GPT2Tokenizer.from_pretrained(args.hf_model, add_prefix_space=True)
    text_encoder.add_special_tokens({'bos_token': '<s>',                                     
                                        'eos_token': '</s>',
                                        'additional_special_tokens': ['[SEP]', '_kw_', '_endkw_']
                                    })

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    with open(os.path.join(output_dir,'checkpoints/checkpoint.pt'), 'rb') as f:
        model = torch.load(f, map_location=device)

    with open(os.path.join(output_dir,'test_dataset'), 'rb') as f:
        test = pickle.load(f)


    generator = StoryGenerator(tokenizer=text_encoder, model=model, device=device)
    data = generator.generate_stories(test, 
                                    n_ctx=args.n_ctx, 
                                    batch_size=args.n_batch, 
                                    max_iter=args.max_samples, 
                                    gen_len=args.gen_len)
    
    data = [ ( flat_text(item[0]), flat_text(item[1]), flat_text(item[2]) ) for item in data]
    write_stories(data, output_dir)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='savedir', help='directory to save logs and checkpoints to')
    parser.add_argument('--experiment_name', type=str, required=True, help='name of this experiment will be included in output')
    # parser.add_argument('--beam', type=int, default=1, help='beam size for beam search')
    # parser.add_argument('--k', type=int, default=0, help='k for TopK sampling')
    # parser.add_argument('--p', type=float, default=0.9, help='p for Nucleus sampling')
    # parser.add_argument('--temperature', type=float, default=0.7, help='temperature for text generation')
    # parser.add_argument('--repeattheta', type=float, default=1.4, help='how much to penalize repitition (1 is not at all, > 1 is more penalty)')  
    parser.add_argument('--gen_len', type=int, default=512, help='max generation length + 1 for end token')
    parser.add_argument('--n_ctx', type=int, default=70, help='keyword tokens length')  
    parser.add_argument('--hf_model', type=str, default="sberbank-ai/rugpt3small_based_on_gpt2", help='name for GPT2 or GPT3 model from Hugginface')
    parser.add_argument('--n_batch', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None, help='limit dataset')  
    args = parser.parse_args()
    print(args)
    main(args)