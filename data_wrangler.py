import numpy as np
import sys
from conllu import parse_incr, TokenList
from io import open
from itertools import islice


words_w_tags = list()
unique_tokens = set()
vocab_size = 10002  # an "unknown word" token and a "padding" token, plus 10000 vocab from data
sentence_lengths = list()
sentences = list()
sen_length_constant = int()
total_tokens = int()
cut_tokens = int()
count_dict = dict()
dict_list = list()


def update_token_stats(orig_length, cut_length):
    global total_tokens, cut_tokens
    total_tokens += orig_length
    cut_tokens += orig_length - cut_length


def pad_sentence(orig_length, cut_sentence):
    """
    This sentence appends pads to the end of the cut_sentence
    :param orig_length: an integer, the length of the original sentence
    :param cut_sentence: a TokenList, a tokenized sentence of size < sen_length_constant
    :return: a TokenList of size orig_length with orig_length - len(cut_sentence) pads
    """
    pad = "PAD"
    padded_sentence = cut_sentence.copy()
    for i in range(len(cut_sentence), int(orig_length)):
        padded_sentence.append(pad)
    return padded_sentence


def trunc_and_pad():
    """
    This method reads through all sentences and checks their lengths against the sen_length_constant. It then cuts the
    sentences to length if necessary, and calls the pad_sentence method.
    :return: a list, containing all cleaned sentences
    """
    global sen_length_constant, sentences, sentence_lengths
    assert(len(sentences) == len(sentence_lengths))
    new_sentences = list()
    for i in range(len(sentence_lengths)):
        old_sen = sentences[i]
        if sentence_lengths[i] > sen_length_constant:  # don't need to pad, just cut
            new_sen = old_sen[:int(sen_length_constant)]
            assert(len(new_sen) == sen_length_constant)
            new_sentences.append(new_sen)
            update_token_stats(sentence_lengths[i], len(new_sen))
        elif sentence_lengths[i] < sen_length_constant:  # don't need to cut, just pad
            padded_sen = pad_sentence(sen_length_constant, old_sen)
            assert(len(padded_sen) == sen_length_constant)
            new_sentences.append(padded_sen)
            update_token_stats(sentence_lengths[i], sentence_lengths[i])
        else:  # sentence length == sen_length_constant
            new_sentences.append(old_sen)
            update_token_stats(sentence_lengths[i], sentence_lengths[i])
    return new_sentences


def take(length, data_structure):
    """
    :param length: desired length of returned structure
    :param data_structure: original structure from which to take [length] objects from
    :return: a new list of length [length] with data from [data_structure]
    """
    return list(islice(data_structure, length))


def sort_and_cut(word_counts):
    """
    :param word_counts: a dictionary, containing words from training and their counts
    :return: a dictionary of length=300 that consists of the highest count affixes in the data-set
    """
    sort_word_counts = {k: v for k, v in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)}
    cut_sort_counts = take(10000, sort_word_counts.keys())
    assert(len(cut_sort_counts) == 10000)
    return cut_sort_counts


def update_word_counts(word):
    global count_dict
    if word in count_dict.keys():
        old_count = count_dict.get(word)
        new_count = old_count + 1
    else:
        new_count = 1
    count_dict.update({word: new_count})


def get_pos(sentence, length):
    """
    This method is used to iterate through words of a sentence and their respective POS tags. Important count global
    variables are also updated here.
    :param sentence: a line read from the conllu file
    :param length:  the length of the sentence
    """
    global sentence_lengths, sentences, count_dict, unique_tokens, words_w_tags
    i = 0
    # iterates through sentence and returns dict for each token
    temp_dict = dict()
    sen = list()
    for i in range(length):
        line = sentence[i]
        word = line["lemma"]
        sen.append(word)
        tags = line["feats"]
        update_word_counts(word)
        if tags is None:
            tags = set('-')
        else:
            tags = set(list(tags)[0].split(";"))
        unique_tokens.update(tags)
        pair = {word: tags}
        temp_dict.update(pair)
        words_w_tags.append([word, tags])
    sentences.append(sen)
    return temp_dict


def read_data(data):
    global sentence_lengths, words_w_tags, sen_length_constant, cut_tokens, total_tokens, count_dict, sentences,\
        unique_tokens, dict_list
    """
    :param data: this is the data set to be used for successive computations. All data-set variables are listed at the
    top of this.py file.
    :return:
        affixes, a total list of the affixes found
        feature_counts, a dictionary of all affixes and their occurrence count in the data
        pop_features, a list of the 300 most-often occuring affixes in the data set, including the counts
        words, a list of all words found in the data set
        affixes_w_words, a list of sublists that match word with their n-gram affixes
    """
    with open(data, "r", encoding='utf-8') as data_file:
        for sentence in parse_incr(data_file):
            sen_len = len(sentence)
            sentence_lengths.append(sen_len)
            sent_dict = get_pos(sentence, sen_len)
            dict_list.append(sent_dict)
        if 'train' in data:
            sen_length_constant = int(np.percentile(sentence_lengths, 95))
        filtered_sentences = trunc_and_pad()
        # print("Data-set: ", data)
        # print("Cut tokens: ", cut_tokens)
        # print("Percentage of tokens that were cut: ", cut_tokens/total_tokens)
        ret_dict_list = dict_list
        popular_words = sort_and_cut(count_dict)
        popular_words.append("UNK")
        popular_words.append("PAD")
        assert("UNK" in popular_words and "PAD" in popular_words)
        ret_words_w_tags = words_w_tags
        sentence_lengths, words_w_tags, sentences, dict_list = list(), list(), list(), list()
    return filtered_sentences, popular_words, ret_words_w_tags, unique_tokens, ret_dict_list


code_file = sys.argv[0]
train_path = sys.argv[1]
dev_path = sys.argv[2]
assert(len(sys.argv) == 3)


train_sentences, train_vocab, words_w_tags_train, unique_tokens_train, dict_list_train = read_data(train_path)
unique_tokens_train = list(unique_tokens_train)
dev_sentences, dev_vocab, words_w_tags_dev, filler_unique_tokens, dict_list_dev = read_data(dev_path)
print("data_wrangler.py is done running!!")
