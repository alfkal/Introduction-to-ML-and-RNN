from Classifier import *
from Evaluate import *
import torch.optim as optim
import os


print(find_files('../data/names/*.txt'))
print(unicode_to_ascii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []


for filename in find_files('../data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

# Initiate the model with it parameters
n_categories = len(all_categories)
# TODO: You can play with these parameters to get different results for the model.
n_hidden = 128
batch_size = 1
rnn = RNN(n_letters, n_hidden, n_categories)
print(rnn)

# Performing a test to see if the network runs
input = lines_to_tensor(['Albert', 'Per'], 6, 2)
hidden = torch.zeros(2, n_hidden)
output, next_hidden = rnn(input[0], hidden)


# Here we take a look at how our training data looks by looping through the categories and picks one random example
for i in range(10):
    category, line, category_tensor, line_tensor = random_training_batch_example(all_categories, category_lines, 1)
    print('category =', category, '/ line =', line)


# Initiating our optimizer
criterion = nn.NLLLoss()
# TODO: Play with optimizer and learning rate
learning_rate = 0.0005  # If you set this too high, it might explode. If too low, it might not learn
optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

# Here we define how many iteration we should perform and how often we should print
# TODO: play with this parameter to get better results. Remember to increase print_every and plot_every as well.
n_iters = 50000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

start = time.time()

for iter in range(1, n_iters + 1):
    # Generating a batch with examples and use them to train the model
    category, line, category_tensor, line_tensor = random_training_batch_example(all_categories, category_lines, batch_size)
    output, loss = train(rnn, criterion, optimizer, category_tensor, line_tensor, batch_size)

    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = category_from_output(output, all_categories)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, time_since(start), loss, line[0], guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# Do some static predictions to test the model
predict(rnn, 'Dovesky', all_categories)
predict(rnn, 'Jackson', all_categories)
predict(rnn, 'Satoshi', all_categories)

# Plots loss and how certain the model is on different categories.
plot_figure(rnn, all_losses, all_categories, category_lines, n_categories)
