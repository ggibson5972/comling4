import subprocess
import sys


chinese_data = "C:/Users/grace/PycharmProjects/5802_data/task2/UD_Chinese-GSD/zh_gsd-um-train.conllu"
english_data = "C:/Users/grace/PycharmProjects/5802_data/task2/UD_English-EWT/en_ewt-um-train.conllu"
portu_data = "C:/Users/grace/PycharmProjects/5802_data/task2/UD_Portuguese-GSD/pt_gsd-um-train.conllu"
russian_data = "C:/Users/grace/PycharmProjects/5802_data/task2/UD_Russian-GSD/ru_gsd-um-train.conllu"
spanish_data = "C:/Users/grace/PycharmProjects/5802_data/task2/UD_Spanish-Ancora/es_ancora-um-train.conllu"
turkish_data = "C:/Users/grace/PycharmProjects/5802_data/task2/UD_Turkish-IMST/tr_imst-um-train.conllu"
chinese_dev = "C:/Users/grace/PycharmProjects/5802_data/task2/UD_Chinese-GSD/zh_gsd-um-dev.conllu"
english_dev = "C:/Users/grace/PycharmProjects/5802_data/task2/UD_English-EWT/en_ewt-um-dev.conllu"
portu_dev = "C:/Users/grace/PycharmProjects/5802_data/task2/UD_Portuguese-GSD/pt_gsd-um-dev.conllu"
russian_dev = "C:/Users/grace/PycharmProjects/5802_data/task2/UD_Russian-GSD/ru_gsd-um-dev.conllu"
spanish_dev = "C:/Users/grace/PycharmProjects/5802_data/task2/UD_Spanish-Ancora/es_ancora-um-dev.conllu"
turkish_dev = "C:/Users/grace/PycharmProjects/5802_data/task2/UD_Turkish-IMST/tr_imst-um-dev.conllu"

training = [chinese_data, english_data, portu_data, russian_data, spanish_data, turkish_data]
dev = [chinese_dev, english_dev, portu_dev, russian_dev, spanish_dev, turkish_dev]

# code_file = sys.argv[0]
# train_path = sys.argv[1]
# dev_path = sys.argv[2]
i = 0
while i < len(training):
    train_set = training[1]
    dev_set = dev[1]

    print("Working on " + train_set + " data now!!")
    # subprocess.run(['python', 'data_wrangler.py', train_set, dev_set])
    # subprocess.run(['python', 'matrix.py', train_set, dev_set])
    subprocess.run(['python', 'train.py', train_set, dev_set])
    # subprocess.run(['python', 'modeling.py', train_set, dev_set])
    break
