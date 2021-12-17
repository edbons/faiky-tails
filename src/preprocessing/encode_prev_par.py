import pandas as pd
from src.model.generate_stories import tfmclassifier
import pickle
from transformers import GPT2Model, GPT2Tokenizer
import argparse
import torch


def encode_pars(input_file: str, model: str, device: str, gen_len=922): 
    df = pd.read_csv(input_file, sep='\t')
    df['[PREVIOUS_PARAGRAPH]'].fillna('NA', inplace=True)
    prevpars = df['[PREVIOUS_PARAGRAPH]'].to_list()
    
    tokenizer = GPT2Tokenizer.from_pretrained(model)
    model = GPT2Model.from_pretrained(model, output_hidden_states=True)

    output = [(0,0,0)]
    for i, par in enumerate(prevpars, start=1):    
        output.append((i, par, tfmclassifier([par], model, tokenizer, gen_len=gen_len, device=device)))

    output_file = input_file.split('.')[0] + '.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(output, f) 


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encode_pars(args.input_file, args.model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help="Path to file with plots")
    parser.add_argument('--model', default="sberbank-ai/rugpt3small_based_on_gpt2", action='model name from huggingface')
    args = parser.parse_args()
    print(args)
    main(args)