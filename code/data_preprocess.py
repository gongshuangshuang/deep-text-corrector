import nltk
from sklearn.model_selection import train_test_split

# -------------数据处理------------
def data_process():
    with open('../data/movie_lines.txt', 'rb') as fin, \
            open('../data/preprocessed_movie_lines.txt', 'w', encoding='utf-8') as fout:
        for lines in fin.readlines():
            line = str(lines).strip().split('+++$+++')[-1].split('\\n')[0].replace('\\', '')
            s = line.strip().lower()
            preprocessed_line = ' '.join(nltk.word_tokenize(s))
            fout.write(preprocessed_line+'\n')

# -------------数据切分------------
def data_split():
    with open('../data/preprocessed_movie_lines.txt', 'r', encoding='utf-8') as fin, \
        open('../data/movie_dialog_train.txt', 'w', encoding='utf-8') as f_train, \
        open('../data/movie_dialog_val.txt', 'w', encoding='utf-8') as f_val, \
        open('../data/movie_dialog_test.txt', 'w', encoding='utf-8') as f_test:
        data = fin.readlines()
        train_data, valtest_data = train_test_split(data, test_size=0.2, random_state=0)
        val_data, test_data = train_test_split(valtest_data, test_size=0.5, random_state=0)
        for line in train_data:
            f_train.write(line)
        for line in val_data:
            f_val.write(line)
        for line in test_data:
            f_test.write(line)
        print(len(train_data), len(val_data), len(test_data))
data_process()
data_split()
