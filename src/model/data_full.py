import re
from rake_nltk import Rake, Metric

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

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

        self.rake = Rake(language='russian', 
                        stopwords=stopwords.words('russian'), 
                        ranking_metric=Metric.WORD_DEGREE, 
                        max_length=5, 
                        include_repeated_phrases=False)

    
    def __len__(self):
        return len(self.data)

    def preprocess_text(self, text: str=""):
        text = re.sub('\.\.\.', '.', text)
        text = re.sub('—', '-', text)
        return text
    
    def extract_context(self, text: str="", topK=20):
        try:            
            self.rake.extract_keywords_from_text(text)
            top_features = self.rake.get_ranked_phrases()

            if len(top_features) > topK:
                top_features = top_features[:topK]
            
            return top_features

        except Exception as e:
            print("Fail Rake on text:", text)
            print("Exception:", e)
                
        
    def __getitem__(self, indx):
        target_txt = self.data[indx]
        target_txt = self.preprocess_text(target_txt)
        context = self.extract_context(target_txt)
        context = " _kw_ ".join(context)              

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