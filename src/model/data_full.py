from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import torch

class PromtDataset(Dataset):
    def __init__(self, 
                 data_file: str="", 
                 tokenizer: GPT2Tokenizer=None, 
                 pad_len: int=2048,
                 max_samples: int=None,
                 n_ctx: int=100
                 ):
        
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
            self.data.pop(0)

        self.tokenizer = tokenizer
        self.pad_len = pad_len
        if max_samples is not None:
            self.data = self.data[:max_samples]

        self.n_ctx = n_ctx

    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, indx):
        target_txt = self.data[indx]        
        context, target_txt = target_txt.split('|')
        target_txt = target_txt.replace('[EOP]', '\n')             

        context = self.tokenizer.encode(context)
        target_txt = self.tokenizer.encode(target_txt)
        
        septok = self.tokenizer.convert_tokens_to_ids('[SEP]')
        starttok = self.tokenizer.convert_tokens_to_ids('<s>')
        endtok = self.tokenizer.convert_tokens_to_ids('</s>')
        endkeytok = self.tokenizer.convert_tokens_to_ids('_endkw_')

        ctx_mask = []

        if len(context) > (self.n_ctx - 3):  # [starttok] + [endkeytok] + [septok]
            context = context[:self.n_ctx - 3] 
        
        context = [starttok] + context + [endkeytok] + [septok]

        if len(context) < self.n_ctx:
            ctx_mask = [1] * len(context) + [0] * (self.n_ctx - len(context))
            context = context + [0] * (self.n_ctx - len(context))

        else:
            ctx_mask = [1] * len(context)

        target_txt = target_txt + [endtok]
        sample = context + target_txt
        
        if len(sample) <= self.pad_len:            
            mask = ctx_mask + [1] * len(target_txt) + [0] * (self.pad_len - len(sample))
            label = [-100] * len(context) + target_txt + [-100] * (self.pad_len - len(sample))  # не считать лосс по контексту
            sample = sample + [endtok] * (self.pad_len - len(sample))
        else:
            target_txt = target_txt[:self.pad_len - len(context)]
            sample = context + target_txt
            sample[-1] = endtok
            mask = ctx_mask + [1] * len(target_txt) + [0] * (self.pad_len - len(sample))
            label = [-100] * len(context) + sample[len(context):] 


        sample = torch.LongTensor(sample)
        mask = torch.LongTensor(mask)
        label = torch.LongTensor(label)

        return {
            'sample': sample, 
            'mask': mask, 
            'label': label
        }

if __name__ == "__main__":
    pass 