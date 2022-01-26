import datasets
from typing import List
from transformers import AutoTokenizer



BERT_MODEL = 'DeepPavlov/rubert-base-cased'

def evaluate_bertscore(hyps: List[str], refs: List[str]) -> dict:
    print('Evaluating BERTScore...')
    layers = [11, 12]       
    bertscore = datasets.load_metric("bertscore")
    
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    ref_idxs = tokenizer(text=hyps, truncation=True, max_length = 512, is_split_into_words=False)['input_ids']
    hypo_idxs = tokenizer(text=refs, truncation=True, max_length = 512, is_split_into_words=False)['input_ids']
    hyps = tokenizer.batch_decode(ref_idxs, skip_special_tokens=True, clean_up_tokenization_spaces=False) 
    refs = tokenizer.batch_decode(hypo_idxs, skip_special_tokens=True, clean_up_tokenization_spaces=False) 

    results = {}
    for layer in layers:
        scores = bertscore.compute(predictions=hyps, references=refs, model_type='DeepPavlov/rubert-base-cased', num_layers=layer, batch_size=2)
        results [f'bertscore_f1_l{layer}'] = sum(scores['f1']) / len(scores['f1'])
    
    return results
