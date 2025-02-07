###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model.
#
###############################################################################
import argparse
import torch

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')
# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--height', type=int, default=20,
                    help='height of the 2D grid')
parser.add_argument('--width', type=int, default=20,
                    help='width of the 2D grid')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default=80,
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")
if torch.backends.mps.is_available():
    if not args.mps:
        print("WARNING: You have mps device, to enable macOS GPU run with --mps.")
        
use_mps = args.mps and torch.backends.mps.is_available()
if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3.")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location=device, weights_only=False)
model.eval()

corpus = data.Corpus(args.data, args.height, args.width)
ntokens = len(corpus.dictionary)

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)
# input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device) # Shape: (1, 1) Randomly initialize the <start> token
# input = torch.zeros(1, 1, dtype=torch.long).to(device)  # Shape: (1, 1)
input = corpus.train[:100].unsqueeze(1).to(device)  # Shape: (200, 1)
input_copy = input.clone()
# input = torch.LongTensor([[97]]).to(device)
# input = torch.randint(ntokens, (16, 1), dtype=torch.long).to(device)
# import os
# if os.path.exists(args.outf):
#     os.remove(args.outf)

word_indices = []
with torch.no_grad():  # no tracking history
    for i in range(args.height * args.width-100):
        if is_transformer_model:
            output = model(input, False)
            word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)
            input = torch.cat([input, word_tensor], 0) # Shape: (i+1, 1)
        else:
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)

        word_indices.append(word_idx)

        # word = corpus.dictionary.idx2word[word_idx]

        # outf.write(word + ('\n' if i % 20 == 19 else ' '))

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.height * args.width))

# Convert the list of word indices to a tensor
word_indices_tensor = torch.tensor(word_indices, dtype=torch.long).to(device)  # Shape: (height * width)

# Concatenate the original input with the generated words
word_indices_tensor = torch.cat([input_copy.squeeze(), word_indices_tensor], 0)

# Get the spiral indices
spiral_indices = data.spiral_indices(args.height, args.width)

# Create an empty tensor to hold the grid
grid = torch.zeros(args.height * args.width, dtype=torch.long).to(device)

# Assign the word indices to the grid
grid[spiral_indices] = word_indices_tensor  # Shape: (height * width)

# Reshape the grid to a 2D tensor
grid = grid.reshape(args.height, args.width)  # Shape: (height, width)

# Convert the grid to a list of words
words = [[corpus.dictionary.idx2word[word_idx] for word_idx in row] for row in grid]

with open(args.outf, 'w') as outf:
    for row in words:
        outf.write(' '.join(row) + '\n')
