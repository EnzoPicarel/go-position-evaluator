"""Go value network, adapted from the 2026 coursework by Numa Guiot and Enzo Picarel."""
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
BOARD_SIZE = 8

def name_to_coord(s):
    assert s != 'PASS'
    indexLetters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
    col = indexLetters[s[0]]
    lin = int(s[1:]) - 1
    return (col, lin)

def stones_to_board(black_stones, white_stones):
    board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for stone in black_stones:
        col, row = name_to_coord(stone)
        board[0, row, col] = 1.0
    for stone in white_stones:
        col, row = name_to_coord(stone)
        board[1, row, col] = 1.0
    return board

def board_symmetries(board, target):
    variants = []
    for k in range(4):
        rotated = np.rot90(board, k=k, axes=(1, 2)).copy()
        variants.append((rotated, target))
        variants.append((np.flip(rotated, axis=2).copy(), target))
    return variants

def build_dataset(samples, augment=True):
    x_list = []
    y_list = []
    for sample in samples:
        board = stones_to_board(sample['black_stones'], sample['white_stones'])
        target = float(sample['black_wins']) / float(sample['rollouts'])
        variants = board_symmetries(board, target) if augment else [(board, target)]
        for transformed_board, transformed_target in variants:
            x_list.append(transformed_board)
            y_list.append(transformed_target)
    x = np.stack(x_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return (x, y)

class GoBoardCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(2, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.01), nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.01), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.01))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, 128), nn.LeakyReLU(0.01), nn.Dropout(0.3), nn.Linear(128, 64), nn.LeakyReLU(0.01), nn.Dropout(0.3), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x.squeeze(-1)

def load_samples(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def position_key(sample):
    """Group identical model inputs and all eight board symmetries together."""
    board = stones_to_board(sample['black_stones'], sample['white_stones'])
    return min(b.tobytes() for b, _ in board_symmetries(board, 0.0))


def split_samples(samples, validation_fraction=0.33, seed=7):
    """Split symmetry groups BEFORE augmenting training examples."""
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must be between zero and one.')
    groups = {}
    for sample in samples:
        groups.setdefault(position_key(sample), []).append(sample)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError('At least two distinct board symmetry groups are required.')
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_valid = min(len(keys)-1, max(1, round(len(keys)*validation_fraction)))
    valid_keys = set(keys[:n_valid])
    train = [s for k in keys if k not in valid_keys for s in groups[k]]
    valid = [s for k in keys if k in valid_keys for s in groups[k]]
    return train, valid


def predict(model, samples, device='cpu', batch_size=128):
    if not samples:
        return np.empty(0, dtype=np.float32)
    model = model.to(device)
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            boards = np.stack([stones_to_board(s['black_stones'], s['white_stones'])
                               for s in samples[start:start+batch_size]])
            values.extend(model(torch.from_numpy(boards).to(device)).cpu().tolist())
    return np.asarray(values)


def train(samples, epochs=20, batch_size=128, seed=7, device='cpu'):
    if epochs < 1 or batch_size < 1:
        raise ValueError('Epochs and batch size must be positive.')
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_samples, valid_samples = split_samples(samples, seed=seed)
    x, y = build_dataset(train_samples, augment=True)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    vx, vy = build_dataset(valid_samples, augment=False)
    valid_loader = DataLoader(TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy)),
                              batch_size=batch_size)
    model = GoBoardCNN().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(bx)
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for bx, by in valid_loader:
                loss = criterion(model(bx.to(device)), by.to(device))
                validation_loss += loss.item()*len(bx)
        row = {'epoch': epoch+1, 'train_bce': total/len(x),
               'validation_bce': validation_loss/len(vx)}
        history.append(row)
        print(json.dumps(row), flush=True)
    probabilities = predict(model, valid_samples, device, batch_size)
    baseline = float(np.mean([s['black_wins']/s['rollouts'] for s in train_samples]))
    metrics = {'train_positions': len(train_samples), 'validation_positions': len(valid_samples),
               'augmented_train_positions': len(x),
               'validation_mae': float(np.abs(probabilities-vy).mean()),
               'constant_baseline_mae': float(np.abs(baseline-vy).mean()),
               'seed': seed, 'epochs': epochs, 'device': device,
               'split': 'board symmetry groups; training-only augmentation'}
    return model, history, metrics
