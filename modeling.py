# TODO: part 2 of homework
import torch
from dimensions import all_tags_train
from matrix import feature_matrix_dev, feat_indexes, dim_indexes, word_indexes
from train import lin_layer, get_batches


predictions = list()

corr_select = 0
incorr_nonselect = 0
incorr_select = 0


def update_selected_counts(predicted_pos, true_pos):
    global corr_select, incorr_nonselect, incorr_select
    predict_set = set(predicted_pos)
    true_set = set(true_pos)
    corr_sel = predict_set.intersection(true_set)  # predicted tags that are correct and appear in the actual tags
    incorr_unsel = predict_set.difference(true_set)  # actual tags that are correct and don't appear in predicted tags
    incorr_sel = true_set.difference(predict_set)  # predicted tags that are wrong and don't appear in the actual tags
    corr_select += len(corr_sel)
    incorr_nonselect += len(incorr_unsel)
    incorr_select += len(incorr_sel)


def get_recall():
    return corr_select / (corr_select + incorr_nonselect)


def get_precision():
    return corr_select / (corr_select + incorr_select)


def get_fscore(precision, recall):
    score = (2 * precision * recall) / (precision + recall)
    return score


def match_predictions(probs, d_indexes, w_indexes):
    global predictions
    predict_indexes = dict()
    rows = list(probs.size())[0]
    columns = list(probs.size())[1]
    for row in range(rows):  # for each word
        features = set()
        for feature in range(columns):  # for each tag
            if probs[row, feature] >= 0.5:  # if prob(tag) > 50%
                features.add(feature)
        predict_indexes.update({row: features})  # add word/tag pair to list
    for x, y in predict_indexes.items():
        tags = set()
        word_index = x
        dim_index = y  # could contain many dimensions
        word = w_indexes[word_index]  # get word through index
        if y == set():
            tags.add('_')
        else:
            for index in dim_index:
                dim = d_indexes[index]  # get tag dimensions through index
                tags.add(dim)
        predictions.append([word, tags])


def run_model(linear_layer, dev_feat_data, d_indexes, w_indexes, train_tags):
    global predictions, corr_select, incorr_nonselect, incorr_select
    batch_size = 32  # batch_size
    batches = get_batches(dev_feat_data, batch_size)
    print("Batches created!")
    for batch in batches:
        torch.no_grad()
        batch_ten = torch.tensor(batch).float()
        scores = linear_layer(batch_ten)  # prediction
        probs = torch.sigmoid(scores)  # probability of prediction
        match_predictions(probs, d_indexes, w_indexes)
    for predict in predictions:
        predict_tags = predict[1]
        word = predict[0]
        expect_tags = set()
        for i in range(len(train_tags)):
            expect_tag = train_tags[i][1]
            if train_tags[i][0] == word:  # if word in predict matches word in training
                expect_tags.update(expect_tag)
        update_selected_counts(predict_tags, expect_tags)
    recall = get_recall()
    precision = get_precision()
    print("correctly selected dims: ", corr_select)
    print("incorrectly selected dims: ", incorr_select)
    print("correctly not selected dims: ", incorr_nonselect)
    print("recall: ", recall)
    print("precision: ", precision)
    print("fscore: ", get_fscore(precision, recall))
    corr_select, incorr_nonselect, incorr_select = 0, 0, 0


run_model(lin_layer, feature_matrix_dev, dim_indexes, word_indexes, all_tags_train)
