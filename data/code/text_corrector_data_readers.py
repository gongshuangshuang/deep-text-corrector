from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
from data_reader import DataReader, PAD_TOKEN, EOS_TOKEN, GO_TOKEN

class MovieDialogReader(DataReader):
    UNKNOWN_TOKEN = "UNK"

    DROPOUT_TOKENS = {"a", "an", "the", "'ll", "'s", "'m", "'ve", "to"}
    REPLACEMENTS = {"there": "their", "their": "there", "then": "than",
                    "than": "then", "be": "to"}

    def __init__(self, config, train_path=None, token_to_id=None,
                 dropout_prob=0.25, replacement_prob=0.25, dataset_copies=2):
        super(MovieDialogReader, self).__init__(
            config, train_path=train_path, token_to_id=token_to_id,
            special_tokens=[PAD_TOKEN, GO_TOKEN, EOS_TOKEN, MovieDialogReader.UNKNOWN_TOKEN],
            dataset_copies=dataset_copies)
        self.dropout_prob = dropout_prob
        self.replacement_prob = replacement_prob
        self.UNKNOWN_ID = self.token_to_id[MovieDialogReader.UNKNOWN_TOKEN]

    # 构建错误数据集，其中source为修改后的错误数据集，target为对应的原始正确数据集
    def read_samples_by_string(self, path):
        for tokens in self.read_tokens(path):
            source = []
            target = []
            for token in tokens:
                target.append(token)
                dropout_token = (token in MovieDialogReader.DROPOUT_TOKENS and
                                 random.random() < self.dropout_prob)
                replace_token = (token in MovieDialogReader.REPLACEMENTS and
                                 random.random() < self.replacement_prob)
                if replace_token:
                    source.append(MovieDialogReader.REPLACEMENTS[token])
                elif not dropout_token:
                    source.append(token)
            yield source, target

    def read_tokens(self, path):
        with open(path, "r", encoding='utf-8') as f:
            for line in f:
                yield line.lower().strip().split()

    def unknown_token(self):
        return MovieDialogReader.UNKNOWN_TOKEN

