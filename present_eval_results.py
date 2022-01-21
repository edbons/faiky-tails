import pickle
from glob import glob
import argparse
from prettytable import PrettyTable
import os


# TO DO переделать на сводную таблицу со всеми метриками и всеми экспериментами

def main(args):
    log_dir = os.path.join(args.output_dir, args.experiment_name, 'eval_logs')     

    results = {}
    keys = None
    for filename in glob(f'{log_dir}/{args.metric}*.pickle'):
        label = args.experiment_name
        results[label] = pickle.load(open(filename, 'rb'))
        keys = results[label].keys()

    labels = sorted(results.keys())

    table = PrettyTable([' '] + labels)
    for key in keys:
        new_raw = [key]
        for label in labels:
            if isinstance(results[label][key], float):
                if args.metric == 'tfidf_distance':
                    new_raw.append(f'{results[label][key] * 1000:.4f}')
                else:
                    new_raw.append(f'{results[label][key]:.4f}')
            else:
                new_raw.append(results[label][key])
        table.add_row(new_raw)

    print(table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='savedir', help='directory to save logs and checkpoints to')
    parser.add_argument('--experiment_name', type=str, required=True, help='name of this experiment will be included in output')
    parser.add_argument('--metric', type=str, required=True, help='Metric from list: ms_jaccard, tfidf_distance, frechet_bert_distance, forward_backward_bleu, rouge')
    args = parser.parse_args()
    print(args)
    main(args)
