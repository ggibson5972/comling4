import numpy as np
from data_wrangler import sen_length_constant, train_sentences, train_vocab, words_w_tags_dev, dev_sentences,\
    dev_vocab, words_w_tags_train, unique_tokens_train, dict_list_train


def gen_zero_matrix(sentences):
    """
    This method generates a zero matrix with #(rows) = #(words) and #(columns) = #(total tag dimensions)
    :param sentences: a list, containing all words from the data set
    :return:
    1) a zero matrix representing words x sen_length_constant
    2) the height of the matrix
    """
    rows = len(sentences)  # number of words == number of rows
    zero_matrix_target = np.zeros((rows, sen_length_constant, len(unique_tokens_train)))
    zero_matrix_input = np.zeros((rows, sen_length_constant))
    return zero_matrix_target, zero_matrix_input, rows


def update_matrix(bare_tar_matrix, bare_train_matrix, sentences, vocab, dict_list, matrix_height):
    """
    This method makes a copy of the bare_matrix and updates the copy according to which features (c) are associated with
    the example word (r)
    :param sentences, a list of TokenList sentences from data-set
    :param vocab, a list of 10000 most popular words plus two for fixing
    :param dict_list, a list of dictionaries
    :param bare_tar_matrix:  the 3D zero matrix (target)
    :param bare_train_matrix: the 2D zero matrix (input)
    :param matrix_height:  an integer
    :return: a binary matrix reflecting tag presence per word
    """
    print("Matrix updating...")
    tri_matrix = bare_tar_matrix.copy()
    bi_matrix = bare_train_matrix.copy()
    for i in range(matrix_height):  # for sentence in matrix
        sen = sentences[i]
        words_n_tags = dict_list[i]
        if i == (matrix_height/2):
            print("Halfway done!!")
        for j in range(sen_length_constant):  # for word in sentence
            word = sen[j]
            if word in vocab:
                value = vocab.index(word)
            else:
                value = vocab.index('UNK')
            bi_matrix[i, j] = value
            if word == 'PAD' or word == 'UNK':
                tri_matrix[i, j] = 0
            else:
                tags = words_n_tags.get(word)
                if len(tags) == 0:
                    tri_matrix[i, j] = 0
                else:
                    for k in range(len(unique_tokens_train)):  # for tag in language
                        tag = unique_tokens_train[k]
                        if tag in tags:
                            tri_matrix[i, j, k] = 1
                        else:
                            tri_matrix[i, j, k] = 0
    print("Done!!")
    return bi_matrix, tri_matrix


def matrix_calls(sentences, vocab, dict_list):
    """
    This method is home to all calls that lead to the generation of the tag dimension binary matrix for a language.
    :param sentences, a list of sentences from data-set
    :param vocab, a list of 10000 most popular words plus two for fixing
    :param dict_list, a list of [word, tags] pairs
    :return: a binary matrix reflecting tag dimension assign. per word
    """
    zero_matrix_target, zero_matrix_train, matrix_height = gen_zero_matrix(sentences)
    dimension_matrix = update_matrix(zero_matrix_target, zero_matrix_train, sentences, vocab, dict_list, matrix_height)
    return dimension_matrix


word_matrix, gold_matrix = matrix_calls(train_sentences, train_vocab, dict_list_train)
np.save('word_matrix', word_matrix)
np.save('gold_matrix', gold_matrix)
#dev_matrix = matrix_calls(dev_sentences, dev_vocab, tag_tokens_dec)
print("matrix.py is done running!!")
