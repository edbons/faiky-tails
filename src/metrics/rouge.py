import rouge
from typing import List

def evaluate_rouge(hyps: List[str], refs: List[str]) -> dict:       
    rouge_scorer = rouge.Rouge()
    averaged_scores = rouge_scorer.get_scores(hyps, refs, avg=True)
    
    results = {
        'n_ref': len(refs),
        'n_hypo': len(hyps)
    }

    for k, v in averaged_scores.items():
        results[k] = v['f']

    return results
