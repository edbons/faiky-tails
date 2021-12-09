import pandas as pd
from src.model.generate_stories import tfmclassifier
import pickle
from transformers import AutoTokenizer, AutoModelWithLMHead
import argparse
import torch


def encode_pars(input_file: str, output_file: str, device: str): 
    df = pd.read_csv(input_file, sep='\t')
    df['[PREVIOUS_PARAGRAPH]'].fillna('dummy', inplace=True)
    prevpars = df['[PREVIOUS_PARAGRAPH]'].to_list()
    
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
    model = AutoModelWithLMHead.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

    output = [(0,0,0)]
    for i, par in enumerate(prevpars):    
        output.append((i, par, tfmclassifier([par], model, tokenizer, gen_len=922, device='cuda')))

    with open(output_file, 'wb') as f:
        pickle.dump(output, f) 


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encode_pars(args.input_file, args.output, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help="Path to file with plots")
    parser.add_argument('--output', action='Path to output pkl file')
    args = parser.parse_args()
    print(args)
    main(args)