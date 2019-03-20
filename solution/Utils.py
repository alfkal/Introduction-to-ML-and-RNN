from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
import torch
import random
import time
import math


all_letters = string.ascii_letters + " .,;'#"
n_letters = len(all_letters)


def find_files(path):
    return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# Turn lines into a <line_length x batch_size x n_letters>,
# or an array of one-hot letter vectors
def lines_to_tensor(lines, max_length, batch_size):
    tensor = torch.zeros(max_length, batch_size, n_letters)
    for i in range(batch_size):
        for li, letter in enumerate(lines[i]):
            tensor[li][i][letter_to_index(letter)] = 1
    return tensor


def category_from_output(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


# Fetches a random training batch and transforms it into tensors
def random_training_batch_example(all_categories, category_lines, batch_size):
    lines = []
    category_tensor = torch.zeros(batch_size, dtype=torch.long)
    category = random_choice(all_categories)
    for i in range(batch_size):
        lines.append(random_choice(category_lines[category]))
        category_tensor[i] = all_categories.index(category)
    max_length = len(max(lines, key=len))
    # category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    lines_tensor = lines_to_tensor(lines, max_length, batch_size)
    return category, lines, category_tensor, lines_tensor


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# This is the training method. It loops over the training batch and calculates the loss based on the real category
# and performs one optimization step
def train(rnn, criterion, optimizer, category_tensor, line_tensor, batch_size):
    hidden = rnn.init_hidden(batch_size)
    optimizer.zero_grad()
    output = ''
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()
