import nltk
from tqdm import trange
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List

import multiprocessing


def get_bleus(tokenized_ref: List[list], tokenized_hypo: List[list], n: int) -> float:
    scores = []
    for i in trange(len(tokenized_hypo), desc=f'Evaluating Self BLEU{n}'):
        
        refs = tokenized_ref[:i] + tokenized_ref[i+1:]
        hyp = tokenized_hypo[i]
        scores.append(sentence_bleu(
            references=refs,
            hypothesis=hyp,
            weights=[1 / n] * n,
            smoothing_function=SmoothingFunction().method1))

    return get_avg(scores)


def evaluate_self_bleu(hypo_texts: List[str], ref_texts: List[str]) -> dict:
    print('Evaluating Self BLEU...')
    tokenized_hypo = [nltk.word_tokenize(text) for text in hypo_texts]

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)

    self_bleu_n = pool.starmap(
        get_bleus, [(tokenized_hypo, tokenized_hypo, n) for n in [2, 3, 4, 5]])
    results = {}
    for n, score in enumerate(self_bleu_n, 2):
        results[f'self_bleu{n}'] = score

    return results


def get_avg(l: list) -> float:
    return sum(l) / len(l)