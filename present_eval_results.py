import pickle
from glob import glob
import argparse
import os
import pandas as pd

METRICS = ['ms_jaccard', 'tfidf_distance', 'frechet_bert_distance', 'forward_backward_bleu', 'rouge', 'bertscore']

def main(args):
    experiments = os.listdir(args.output_dir)

    results = []
    for experiment in experiments:
        log_dir = os.path.join(args.output_dir, experiment, 'eval_logs')     
        
        exp_results=[]
        for metric in METRICS:
            metric_results = {}
            filenames = glob(f'{log_dir}/{metric}*.pickle')
            for name in filenames:
                label = f"{experiment}_{os.path.basename(name).replace(f'{metric}_','').strip('.pickle').strip()}"
                with open(name, 'rb') as f:
                    metric_results[label]=pickle.load(f)
            exp_results.append(pd.DataFrame.from_dict(metric_results, orient='index')) 
        
        results.append(pd.concat(exp_results, axis=1))
    
    df = pd.concat(results, axis=0)
    df.drop(['n_ref', 'n_hypo'], axis=1, inplace=True)
    df.to_csv('experiments_results.csv', encoding='utf-8', sep='|')
    print("Results saved to experiments_results.csv!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='savedir', help='directory to save logs and checkpoints to')
    args = parser.parse_args()
    print(args)
    main(args)
