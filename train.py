"""Train the original CNN with a corrected data split."""
import argparse
import json
from pathlib import Path
import torch
from go_model import load_samples, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--limit', type=int, help='Smoke-test subset; not a benchmark')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or (args.limit is not None and args.limit < 2):
        parser.error('threads must be positive and limit must be at least 2')
    torch.set_num_threads(args.threads)
    samples = load_samples(args.data)
    if args.limit:
        samples = samples[:args.limit]
    model, history, metrics = train(samples, args.epochs, args.batch_size, args.seed, args.device)
    metrics['subset_limit'] = args.limit
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), out/'model.pt')
    (out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
