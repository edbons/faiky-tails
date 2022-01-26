import argparse
import os
import pickle
import pandas as pd
from glob import glob

from src.metrics.ms_jaccard import evaluate_ms_jaccard
from src.metrics.frechet_bert_distance import evaluate_frechet_bert_distance
from src.metrics.tfidf_distance import evaluate_tfidf_distance
from src.metrics.forward_backward_bleu import evaluate_forward_backward_bleu
from src.metrics.rouge import evaluate_rouge
from src.metrics.bertscore import evaluate_bertscore

def eval_all_metrics(ref_texts: list, hypo_texts: list, output_dir: str, label: str) -> None:

    msj_results = evaluate_ms_jaccard(hypo_texts=hypo_texts, ref_texts=ref_texts)
    pickle.dump(msj_results, open(f'{output_dir}/ms_jaccard_{label}.pickle', 'wb'))
    print(msj_results)

    wfd_results = evaluate_tfidf_distance(hypo_texts=hypo_texts, ref_texts=ref_texts)
    pickle.dump(wfd_results, open(f'{output_dir}/tfidf_distance_{label}.pickle', 'wb'))
    print(wfd_results)
    
    fbd_results = evaluate_frechet_bert_distance(hypo_texts=hypo_texts, ref_texts=ref_texts)
    pickle.dump(fbd_results, open(f'{output_dir}/frechet_bert_distance_{label}.pickle', 'wb'))
    print(fbd_results)

    bleu_results = evaluate_forward_backward_bleu(hypo_texts=hypo_texts, ref_texts=ref_texts)
    pickle.dump(bleu_results, open(f'{output_dir}/forward_backward_bleu_{label}.pickle', 'wb'))
    print(bleu_results)

    rouge_results = evaluate_rouge(hyps=hypo_texts, refs=ref_texts)
    pickle.dump(rouge_results, open(f'{output_dir}/rouge_{label}.pickle', 'wb'))
    print(rouge_results)

    bertscore_results = evaluate_bertscore(hyps=hypo_texts, refs=ref_texts)
    pickle.dump(bertscore_results, open(f'{output_dir}/bertscore_{label}.pickle', 'wb'))
    print(bertscore_results)


def main(args):    

    logs_dir = os.path.join(args.output_dir, args.experiment_name, 'eval_logs')
    os.makedirs(logs_dir, exist_ok=True)
    filenames = glob(f'{os.path.join(args.output_dir, args.experiment_name)}/gen_*.csv')
    for filename in filenames:        
        test_examples = pd.read_csv(filename, sep='|')
        ref_texts = test_examples.refs.to_list()
        hypo_texts = test_examples.hyps.to_list()
        label = filename.split('gen_')[-1].strip('.csv').strip()
        eval_all_metrics(ref_texts, hypo_texts, output_dir=logs_dir, label=label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='savedir', help='directory to save logs and checkpoints to')
    parser.add_argument('--experiment_name', type=str, required=True, help='name of this experiment will be included in output')
    args = parser.parse_args()
    print(args)
    main(args)