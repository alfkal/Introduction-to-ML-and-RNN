from Utils import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def evaluate(rnn, line_tensor):
    hidden = rnn.init_hidden(1)

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def plot_figure(rnn, all_losses, all_categories, category_lines, n_categories):
    plt.figure()
    plt.plot(all_losses)
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = random_training_batch_example(all_categories, category_lines, 1)
        output = evaluate(rnn, line_tensor)
        guess, guess_i = category_from_output(output, all_categories)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
    print(confusion.diag())
    exit()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


def predict(rnn, input_line, all_categories, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(rnn, lines_to_tensor(input_line, max_length=len(input_line), batch_size=1))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])
