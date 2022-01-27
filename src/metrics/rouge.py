import rouge
from typing import List

def evaluate_rouge(hypo_texts: List[str], ref_texts: List[str]) -> dict:
    print('Evaluating ROUGE...')       
    rouge_scorer = rouge.Rouge()
    averaged_scores = rouge_scorer.get_scores(hyps=hypo_texts, refs=ref_texts, avg=True)
    
    results = {}    
    for k, v in averaged_scores.items():
        results[k] = v['f']

    return results
