from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
# 定义与常用特殊情况标记关联的常量
PAD_ID = 0
GO_ID = 1
EOS_ID = 2

PAD_TOKEN = "PAD"
EOS_TOKEN = "EOS"
GO_TOKEN = "GO"
class DataReader(object):
    def __init__(self, config, train_path=None, token_to_id=None,
                 special_tokens=(), dataset_copies=1):
        self.config = config
        self.dataset_copies = dataset_copies
        # 建立词典
        max_vocabulary_size = self.config.max_vocabulary_size
        if train_path is None:
            self.token_to_id = token_to_id
        else:
            token_counts = Counter()
            for tokens in self.read_tokens(train_path):
                token_counts.update(tokens)
            self.token_counts = token_counts
            # 获取最大词表数量词建成词典
            count_pairs = sorted(token_counts.items(),
                                 key=lambda x: (-x[1], x[0]))
            vocabulary, _ = list(zip(*count_pairs))
            vocabulary = list(vocabulary)
            # 在词表开头插入特殊字符
            vocabulary[0:0] = special_tokens
            full_token_and_id = zip(vocabulary, range(len(vocabulary)))
            full_token_and_id = list(full_token_and_id)
            self.full_token_to_id = dict(full_token_and_id)
            self.token_to_id = dict(full_token_and_id[:max_vocabulary_size])
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    # 异常处理
    def read_tokens(self, path):
        raise NotImplementedError("Must implement read_tokens")

    def read_samples_by_string(self, path):
        raise NotImplementedError("Must implement read_word_samples")

    def unknown_token(self):
        raise NotImplementedError("Must implement read_word_samples")

    # token编码
    def convert_token_to_id(self, token):
        token_with_id = token if token in self.token_to_id else \
            self.unknown_token()
        return self.token_to_id[token_with_id]

    def convert_id_to_token(self, token_id):
        return self.id_to_token[token_id]

    def is_unknown_token(self, token):
        return token not in self.token_to_id or token == self.unknown_token()

    # sentence编码
    def sentence_to_token_ids(self, sentence):
        return [self.convert_token_to_id(word) for word in sentence.split()]

    # 将word-ids 列表转成对应的words
    def token_ids_to_tokens(self, word_ids):
        return [self.convert_id_to_token(word) for word in word_ids]

    # 样本数据token编码迭代处理
    def read_samples(self, path):
        for source_words, target_words in self.read_samples_by_string(path):
            source = [self.convert_token_to_id(word) for word in source_words]
            target = [self.convert_token_to_id(word) for word in target_words]
            target.append(EOS_ID)
            yield source, target
    # 创建多个副本数据集，以便用以合成不同的dropouts
    # 利用bucketing方式提升训练速度和推理速度
    def build_dataset(self, path):
        dataset = [[] for _ in self.config.buckets]
        for _ in range(self.dataset_copies):
            for source, target in self.read_samples(path):
                for bucket_id, (source_size, target_size) in enumerate(
                        self.config.buckets):
                    if len(source) < source_size and len(
                            target) < target_size:
                        dataset[bucket_id].append([source, target])
                        break
        return dataset





















