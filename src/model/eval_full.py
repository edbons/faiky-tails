import os
import rouge
import argparse
import pickle
import csv

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
from data_full import RawFilesDataset


def rouge_scores(hyps, refs):       
    rouge_scorer = rouge.Rouge()
    averaged_scores = rouge_scorer.get_scores(hyps, refs, avg=True)
    return averaged_scores

def evaluate_batch(batch: dict, text_encoder: GPT2Tokenizer, model: GPT2LMHeadModel, gen_len: int=512)-> tuple:
    septok = text_encoder.convert_tokens_to_ids('[SEP]')
    endtok = text_encoder.eos_token_id
    input_ids = batch['sample']

    sep_idx = torch.where(input_ids[0] == septok)[0].item()
    eos_idx = torch.where(input_ids[0] == endtok)[0][0].item()
    context = input_ids[:, :sep_idx+1]
    target_txt = input_ids[:, sep_idx+1:eos_idx+1]

    context_txt = text_encoder.decode(context[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    refs = text_encoder.decode(target_txt[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    sample_output = model.generate(
                                    context,                     
                                    max_length=gen_len, 
                                    do_sample=True,
                                    num_beams = 20,  # https://arxiv.org/pdf/2108.03502.pdf 
                                    top_p=0.95, # https://arxiv.org/pdf/2108.03502.pdf 
                                    top_k=3, # https://arxiv.org/pdf/2108.03502.pdf
                                    eos_token_id=endtok,
                                    bos_token_id=text_encoder.bos_token_id,
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
    hyps = text_encoder.decode(sample_output[0][sep_idx+1:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    rouge_score = rouge_scores(hyps, refs)
    return context_txt, refs, hyps, rouge_score

def flat_text(text: str="") -> str:
    return text.replace('\r\n',' ').replace('\n',' ').strip()

def write_evaluate_result(data: tuple, path: str=''):
    columns = ['context', 'refs', 'hyps', 'ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1']
    assert len(columns) == len(data[0])    
    with open(os.path.join(path, 'evaluate_results.csv'), 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='|', lineterminator='\n')
        writer.writerow(columns)
        writer.writerows(data)

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

    test_dataset = RawFilesDataset(test, text_encoder, 2048, n_ctx=args.n_ctx)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.to(device)
    model.eval()

    data = []
    for batch in tqdm(test_loader):            
        context, refs, hyps, score = evaluate_batch(batch=batch, text_encoder=text_encoder, model=model, gen_len=args.gen_len)
        data.append( (context, flat_text(refs), flat_text(hyps), score['rouge-1']['f'], score['rouge-2']['f'], score['rouge-l']['f']) )
    
    write_evaluate_result(data, output_dir)


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
    
    args = parser.parse_args()
    print(args)
    main(args)