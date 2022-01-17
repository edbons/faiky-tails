from re import split
from rake_nltk import Rake

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import torch

class FullDataset(Dataset):
    def __init__(self, 
                 data_file: str=None, 
                 tokenizer: GPT2Tokenizer=None, 
                 pad_len: int=1024,
                 max_samples: int=None,
                 n_ctx: int=100
                 ):

        with open(data_file, "rb") as f:
            data = f.readlines()

        self.data = []
        for d in range(1, len(data)):
            t = data[d].decode("utf-8", "ignore").strip().split('\t')
            if len(t) == 2:
                self.data.append(t)

        self.tokenizer = tokenizer
        self.pad_len = pad_len
        if max_samples is not None:
            self.data = self.data[:max_samples]

        self.n_ctx = n_ctx
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, indx):
        context, target_txt = self.data[indx]
        context = context.replace('[SEP]', ' ')        

        context = self.tokenizer.encode(context)
        target_txt = self.tokenizer.encode(target_txt)
        
        clstok = self.tokenizer.cls_token_id
        keytok = self.tokenizer.convert_tokens_to_ids('_kw_')

        if len(context) > self.n_ctx:
            context = context[:self.n_ctx]
        
        sample = [self.tokenizer.bos_token_id] + \
                 [keytok] + context + [0] * (self.n_ctx - len(context)) + \
                 [clstok] + target_txt + \
                 [self.tokenizer.eos_token_id]
        
        if len(sample) <= self.pad_len:            
            mask = [1] * len(sample) + [0] * (self.pad_len - len(sample))
            label = sample + [-100] * (self.pad_len - len(sample))
            sample = sample + [self.tokenizer.bos_token_id] * (self.pad_len - len(sample))
        else:
            sample = sample[:self.pad_len]
            sample[-1] = self.tokenizer.eos_token_id
            mask = [1] * len(sample) + [0] * (self.pad_len - len(sample))
            label = sample + [-100] * (self.pad_len - len(sample))

        sample = torch.LongTensor(sample)
        mask = torch.LongTensor(mask)
        label = torch.LongTensor(label)

        return {
            'sample': sample, 
            'mask': mask, 
            'label': label
        }

class RawFilesDataset(Dataset):
    def __init__(self, 
                 data_files: list=None, 
                 tokenizer: GPT2Tokenizer=None, 
                 pad_len: int=2048,
                 max_samples: int=None,
                 n_ctx: int=100
                 ):

        self.data = []
        for file in data_files:
            with open(file, "rb") as f:
                data = f.read().decode('utf-8', "ignore").strip()
                self.data.append(data)

        self.tokenizer = tokenizer
        self.pad_len = pad_len
        if max_samples is not None:
            self.data = self.data[:max_samples]

        self.n_ctx = n_ctx

        self.rake = Rake(language='russian', stopwords=stopwords.words())

    
    def __len__(self):
        return len(self.data)

    def extract_context(self, text: str="", topK=10):
        try:            
            self.rake.extract_keywords_from_text(text)
            top_features = self.rake.get_ranked_phrases()

            if len(top_features) > topK:
                top_features = top_features[:topK]
            
            return " ".join(top_features)

        except Exception as e:
            print("Fail Rake on text:", text)
            print("Exception:", e)
        
        
        
    def __getitem__(self, indx):
        target_txt = self.data[indx]
        context = self.extract_context(target_txt)              

        context = self.tokenizer.encode(context)
        target_txt = self.tokenizer.encode(target_txt)
        
        # clstok = self.tokenizer.cls_token_id
        septok = self.tokenizer.convert_tokens_to_ids('[SEP]')
        starttok = self.tokenizer.convert_tokens_to_ids('<s>')
        endtok = self.tokenizer.convert_tokens_to_ids('</s>')

        if len(context) > self.n_ctx:
            context = context[:self.n_ctx]
        
        # не считать лосс по контексту
        context = [starttok] + context + [septok]
        target_txt = target_txt + [endtok]
        sample = context + target_txt
        
        # sample = [starttok] + context + \
        #          [septok] + target_txt + \
        #          [endtok]
        

        if len(sample) <= self.pad_len:            
            mask = [1] * len(sample) + [0] * (self.pad_len - len(sample))
            label = [-100] * len(context) + target_txt + [-100] * (self.pad_len - len(sample))  # не считать лосс по контексту
            # label = sample + [-100] * (self.pad_len - len(sample))
            sample = sample + [endtok] * (self.pad_len - len(sample))
        else:
            sample = sample[:self.pad_len]
            sample[-1] = endtok
            mask = [1] * len(sample) + [0] * (self.pad_len - len(sample))
            label = [-100] * len(context) + sample[len(context):] # не считать лосс по контексту
            # label = sample + [-100] * (self.pad_len - len(sample))  # TO DO не считать лосс по контексту

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