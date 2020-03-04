import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_wrangler import train_sentences, train_vocab, words_w_tags_train, dev_sentences, dev_vocab, words_w_tags_dev, \
    train_path, unique_tokens, sen_length_constant

torch.manual_seed(1)
elements = 0
array_results = list()
L_RATE = 0.1
BATCH_SIZE = 64
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
EPOCHS = 20


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim*embedding_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.permute(1, 0, 2).view(-1, self.hidden_dim*len(sentence)))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def print_results():
    global L_RATE, BATCH_SIZE, EPOCHS
    layers = "hidden layers: " + str(4)
    batch_info = "batch_size: " + str(BATCH_SIZE)
    epoch_info = "epoch count: " + str(EPOCHS)
    learn_info = "learning rate: " + str(L_RATE)
    print(batch_info)
    print(learn_info)
    print(epoch_info)
    print(layers)


def get_batches(matrix):
    global BATCH_SIZE
    word_count = len(matrix)
    batches = []
    end = BATCH_SIZE
    i = 0
    while i < word_count:
        batch = matrix[i:end]
        batches.append(batch)
        end += BATCH_SIZE
        i += BATCH_SIZE
    return batches


def train_model(batch_matrix, target_matrix, vocab_size, tagset_size):
    global L_RATE, BATCH_SIZE, EMBEDDING_DIM, HIDDEN_DIM, EPOCHS
    batches = get_batches(batch_matrix)
    targets = get_batches(target_matrix)
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, tagset_size)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=L_RATE)
    for t in range(EPOCHS):
        i = 0
        for batch in batches:
            model.zero_grad()
            batch_ten = torch.tensor(batch).long()
            target_ten = torch.tensor(targets[i]).float()
            tag_scores = model.forward(batch_ten)

            loss = loss_function(tag_scores, target_ten)
            loss.backward()
            optimizer.step()
            i += 1
    print_results()
    return model, batches, targets


def check_model(train_mod, batchez):
    with torch.no_grad():
        for batch in batchez:
            tag_scores = train_mod(batch)
            print(tag_scores)


word_matrix = np.load('word_matrix.npy')
gold_matrix = np.load('gold_matrix.npy')
trained_model, batches, targets = train_model(word_matrix, gold_matrix, len(train_vocab), len(unique_tokens))
check_model(trained_model, batches)
