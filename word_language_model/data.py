import os
from io import open
import torch
import random
from typing import overload

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path: str, height: int = None, width: int = None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'), height, width)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), height, width)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), height, width)

    @overload
    def tokenize(self, path: str) -> torch.Tensor:
        """Tokenizes a text file."""
        ...

    @overload
    def tokenize(self, path: str, height: int, width: int) -> torch.Tensor:
        """Tokenizes a text file in spiral order."""
        ...
    
    def tokenize(self, path: str, height: int = None, width: int = None) -> torch.Tensor:
        assert os.path.exists(path)

        if height is not None or width is not None:
            # Spiral order tokenization
            # Add words to the dictionary without <eos>
            with open(path, 'r', encoding="utf8") as f:
                for line in f:
                    words = line.split()
                    for word in words:
                        self.dictionary.add_word(word)

            # Tokenize file content
            with open(path, 'r', encoding="utf8") as f:
                # Read all lines and split into words
                words = []
                for line in f:
                    words.extend(line.strip().split())

            # Ensure the total number of words matches height * width
            assert len(words) == height * width, "Number of words does not match grid size."

            # Convert words to indices
            word_indices = [self.dictionary.word2idx[word] for word in words]
            grid = torch.tensor(word_indices, dtype=torch.int64).reshape(height, width)

            # Get spiral indices
            spiral_idx = spiral_indices(height, width)

            # Flatten the grid and reorder in spiral order
            ids = grid.flatten()[spiral_idx]
        else:
            # Normal tokenization
            # Add words to the dictionary with <eos>
            with open(path, 'r', encoding="utf8") as f:
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        self.dictionary.add_word(word)

            # Tokenize file content
            with open(path, 'r', encoding="utf8") as f:
                idss = []
                for line in f:
                    words = line.split() + ['<eos>']
                    ids = []
                    for word in words:
                        ids.append(self.dictionary.word2idx[word])
                    idss.append(torch.tensor(ids).type(torch.int64))
                ids = torch.cat(idss)

        return ids
            
    
def spiral_indices(height: int, width: int) -> torch.Tensor:
    """
    Generate spiral order indices for a 2D grid of size height x width.
    Args:
        height: Height of the grid.
        width: Width of the grid.
    Returns:
        torch.Tensor: A tensor of indices representing spiral order, shape: (height*width,)
    """
    grid = torch.arange(height * width).reshape(height, width)
    spiral_idx = []

    while grid.numel() > 0:
        # Top row
        spiral_idx.extend(grid[0, :].tolist())
        grid = grid[1:, :]  # Remove top row
        if grid.numel() == 0:
            break

        # Right column
        spiral_idx.extend(grid[:, -1].tolist())
        grid = grid[:, :-1]  # Remove right column
        if grid.numel() == 0:
            break

        # Bottom row (reversed)
        spiral_idx.extend(grid[-1, :].flip(0).tolist())
        grid = grid[:-1, :]  # Remove bottom row
        if grid.numel() == 0:
            break

        # Left column (reversed)
        spiral_idx.extend(grid[:, 0].flip(0).tolist())
        grid = grid[:, 1:]  # Remove left column

    # Reverse the spiral order to match the spiral order of the grid
    spiral_idx.reverse()

    return torch.tensor(spiral_idx, dtype=torch.long)

def generate_patterned_grids(height, width, num_samples, noise_level=0.1):
    grids = []
    for _ in range(num_samples):
        grid = []
        for i in range(height):
            row = []
            for j in range(width):
                # Custom pattern logic based on observed patterns
                if (i + j) % 4 == 0:
                    value = 'a'
                elif (i - j) % 3 == 0:
                    value = 'b'
                elif (i * j) % 2 == 0:
                    value = 'c'
                else:
                    value = 'd'
                # Add noise to the pattern
                if random.random() < noise_level:
                    value = random.choice(['a', 'b', 'c', 'd'])
                row.append(value)
            grid.append(row)
        grids.append(grid)
    return grids

def save_grids(filename, grids):
    with open(filename, 'w') as f:
        for grid in grids:
            for row in grid:
                f.write(' '.join(row) + '\n')
            f.write('\n')  # Separate grids with an empty line
