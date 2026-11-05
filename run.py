"""Portable command line entry point. Run from the project root."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect', help='Check dataset and print feature conversion audit')
    tr = sub.add_parser('train', help='Train all models and evaluate final checkpoints')
    tr.add_argument('--output', default='results/main')
    tr.add_argument('--train-size', type=int, default=2048, help='0 uses full training partition')
    tr.add_argument('--val-size', type=int, default=512, help='0 uses full validation partition')
    tr.add_argument('--epochs', type=int, default=40)
    tr.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    tr.add_argument('--no-ablation', action='store_true')
    for command in ['evaluate', 'demo', 'explain', 'plots']:
        p = sub.add_parser(command)
        p.add_argument('--output', default='results/main')
        if command in ('demo', 'explain'):
            p.add_argument('--index', type=int, default=0, help='Position within saved test partition')
    args = parser.parse_args()
    from housing.data import load_data, FEATURES
    if args.command == 'inspect':
        print(json.dumps(load_data()[2], indent=2))
    elif args.command == 'train':
        if args.train_size < 0 or args.val_size < 0 or args.epochs < 1:
            parser.error('Sizes must be nonnegative and epochs positive.')
        from housing.experiment import run_training
        print(json.dumps(run_training(args.output, args.train_size, args.val_size,
                         args.epochs, tuple(args.seeds), not args.no_ablation), indent=2))
    elif args.command == 'evaluate':
        from housing.experiment import evaluate
        print(json.dumps(evaluate(args.output), indent=2))
    elif args.command == 'plots':
        from housing.plots import make_plots
        make_plots(args.output)
    elif args.command == 'explain':
        from housing.explain import explain_prediction
        explain_prediction(args.output, args.index)
    elif args.command == 'demo':
        import torch
        from threadpoolctl import threadpool_limits
        from housing.experiment import load_run, predict_saved
        torch.set_num_threads(2)
        x, y, prep, splits, records = load_run(args.output)
        if not 0 <= args.index < len(splits['test']):
            parser.error('Index is outside the test partition.')
        row = splits['test'][args.index]
        print(f'HELD-OUT EXAMPLE | raw row index {row} (zero-based)')
        for name, value in zip(FEATURES, x[row]):
            print(f'  {name:12s} {value:.4f}')
        print(f'Actual historical median value: ${y[row] * 100000:,.0f}')
        seen = set()
        with threadpool_limits(limits=2):
            for rec in records:
                name = rec['model']
                if name in seen or name == 'qnn_1layer':
                    continue
                seen.add(name)
                pred = predict_saved(args.output, name, rec['seed'], x[row:row+1], prep)[0]
                print(f'  {name:16s} prediction ${pred * 100000:,.0f}; absolute error ${abs(pred-y[row]) * 100000:,.0f}')
        print('One example is illustrative; use the full test metrics for conclusions.')


if __name__ == '__main__':
    main()
