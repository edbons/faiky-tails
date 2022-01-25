import pickle
from glob import glob
import argparse
import os
import pandas as pd

METRICS = ['ms_jaccard', 'tfidf_distance', 'frechet_bert_distance', 'forward_backward_bleu', 'rouge']

def main(args):
    experiments = os.listdir(args.output_dir)

    for experiment in experiments:
        log_dir = os.path.join(args.output_dir, experiment, 'eval_logs')     
        
        results = []
        for metric in METRICS:
            metric_results = {}
            filenames = glob(f'{log_dir}/{metric}*.pickle')
            for name in filenames:
                label = name.strip(f'{metric}_').strip('.pickle').strip()
                with open(name, 'rb') as f:
                    metric_results[label]=pickle.load(f)
            results.append(pd.DataFrame.from_dict(metric_results, orient='index')) 
        
        df = pd.concat(results, axis=1)
        print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='savedir', help='directory to save logs and checkpoints to')
    args = parser.parse_args()
    print(args)
    main(args)
