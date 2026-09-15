"""Export black-win probabilities in input order, one float per line."""
import argparse
from pathlib import Path
import torch
from go_model import GoBoardCNN, load_samples, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='artifacts/predictions.txt')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = GoBoardCNN()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    values = predict(model, load_samples(args.data))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(f'{value:.8f}\n' for value in values))
    print(f'Wrote {len(values)} predictions to {output}')


if __name__ == '__main__':
    main()
