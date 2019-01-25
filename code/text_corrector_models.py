from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import seq2seq
from data_reader import PAD_ID, GO_ID
class TextCorrectorModel(object):
    def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
                 num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, use_lstm=False,
                 num_samples=512, forward_only=False, config=None,
                 corrective_tokens_mask=None):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.config = config

        # 数据输入
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        # 定义encoder 和 decoder 输入的占位符
        # encoder_inputs 这个列表对象中的每一个元素表示一个占位符，其名字分别为encoder0, encoder1,…,encoderN，
        # encoder{i}的几何意义是编码器在时刻i的输入
        for i in range(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(
                                                          i)))
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(
                                                         i)))
            # target_weights 是一个与 decoder_outputs 大小一样的0-1矩阵
            # 该矩阵将目标序列长度以外的其他位置填充为标量值0
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                      name="weight{0}".format(
                                                          i)))
        # corrective_tokens one-hot编码
        corrective_tokens_tensor = tf.constant(corrective_tokens_mask if
                                                corrective_tokens_mask else
                                                np.zeros(self.target_vocab_size),
                                                shape=[self.target_vocab_size],
                                                dtype=tf.float32)

        # 批量的corrective_tokens one-hot编码
        batched_corrective_tokens = tf.stack([corrective_tokens_tensor] * self.batch_size)
        self.batch_corrective_tokens_mask = batch_corrective_tokens_mask = \
            tf.placeholder(
                tf.float32,
                shape=[None, None],
                name="corrective_tokens")
        # 跟language model类似，targets变量是decoder inputs平移一个单位的结果
        targets = [self.decoder_inputs[i + 1]
                    for i in range(len(self.decoder_inputs) - 1)]

        # Sampled softmax
        output_projection = None
        softmax_loss_function = None
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w = tf.get_variable("proj_w", [size, self.target_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [self.target_vocab_size])

            output_projection = (w, b)

            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_t, b, labels, logits,
                                                    num_samples,
                                                    self.target_vocab_size)

            softmax_loss_function = sampled_loss
        # 为RNN创建内部多层单元
        single_cell = tf.nn.rnn_cell.GRUCell(size)
        if use_lstm:
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        cell = single_cell
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # seq2seq模型: embedding and attention model
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            if do_decode:  # 测试阶段
                # 修改偏差，使模型偏向于选择输入句子中的单词
                input_bias = self.build_input_bias(encoder_inputs,
                                                        batch_corrective_tokens_mask)

                # seq2seq模型: embedding and attention model
                # 重新定义了seq2seq，以允许加入一个输入偏置项的解码函数
                return seq2seq.embedding_attention_seq2seq(
                        encoder_inputs, decoder_inputs, cell,
                        num_encoder_symbols=source_vocab_size,
                        num_decoder_symbols=target_vocab_size,
                        embedding_size=size,
                        output_projection=output_projection,  # 不设定的话输出维数可能很大(取决于词表大小)，设定的话投影到一个低维向量
                        feed_previous=do_decode,
                        loop_fn_factory=
                        apply_input_bias_and_extract_argmax_fn_factory(input_bias))
            else:  # 训练阶段
                return seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, cell,
                    num_encoder_symbols=source_vocab_size,
                    num_decoder_symbols=target_vocab_size,
                    embedding_size=size,
                    output_projection=output_projection,
                    feed_previous=do_decode)

        # 训练，输出结果和损失
        if forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function)
            if output_projection is not None:
                for b in range(len(buckets)):
                    # 在解码模型评估时，应用相同的输入偏差
                    input_bias = self.build_input_bias(
                        self.encoder_inputs[:buckets[b][0]],
                        batch_corrective_tokens_mask)
                    self.outputs[b] = [
                        project_and_apply_input_bias(output, output_projection,
                                                        input_bias)
                        for output in self.outputs[b]]

        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function)

        # 参数更新
        params = tf.trainable_variables()
        # 只有训练阶段才需要计算梯度和参数更新
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.RMSPropOptimizer(0.001) if self.config.use_rms_prop \
                else tf.train.GradientDescentOptimizer(self.learning_rate)
            # opt = tf.train.AdamOptimizer()
            for b in range(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)  # 计算损失函数关于参数的梯度
                clipped_gradients, norm = tf.clip_by_global_norm(  # clip gradients 防止梯度爆炸
                    gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params),
                    global_step=self.global_step))  # 更新参数
        self.saver = tf.train.Saver(tf.global_variables())

    # 构建偏置函数
    def build_input_bias(self, encoder_inputs, batch_corrective_tokens_mask):
        packed_one_hot_inputs = tf.one_hot(indices=tf.stack(
            encoder_inputs, axis=1), depth=self.target_vocab_size)
        return tf.maximum(batch_corrective_tokens_mask,
                            tf.reduce_max(packed_one_hot_inputs,
                                        reduction_indices=1))

    # 模型运行一步训练
    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, forward_only, corrective_tokens=None):
        # 尺寸异常检验
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError(
                "Encoder length must be equal to the one in bucket,"
                " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError(
                "Decoder length must be equal to the one in bucket,"
                " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError(
                "Weights length must be equal to the one in bucket,"
                " %d != %d." % (len(target_weights), decoder_size))
        # feed数据
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # 训练中学习corrective tokens
        corrective_tokens_vector = (corrective_tokens
                                    if corrective_tokens is not None else
                                    np.zeros(self.target_vocab_size))
        batch_corrective_tokens = np.repeat([corrective_tokens_vector],
                                            self.batch_size, axis=0)
        input_feed[self.batch_corrective_tokens_mask.name] = (
            batch_corrective_tokens)
        # feed最后一个解码输入数据
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            # Gradient norm, loss, no outputs.
            return outputs[1], outputs[2], None
        else:
            # No gradient norm, loss, outputs.
            return None, outputs[0], outputs[1:]

    # 批处理
    def get_batch(self, data, bucket_id):
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        # 从数据中随机获取一定批次量的输入数据
        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # 编码输入数据填充0，且翻转
            encoder_pad = [PAD_ID] * (
                    encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # 解码输入加入GO字符，并填充0
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([GO_ID] + decoder_input +
                                  [PAD_ID] * decoder_pad_size)

        # 从上面选择的数据中创建批batch-major vectors
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # 批量编码输入重索引
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)],
                         dtype=np.int32))

        # 批量解码输入重索引
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)],
                         dtype=np.int32))
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

def project_and_apply_input_bias(logits, output_projection, input_bias):
    if output_projection is not None:
        logits = nn_ops.xw_plus_b(
            logits, output_projection[0], output_projection[1])
    # 应用SoftMax以确保所有tokens都具有正值
    probs = tf.nn.softmax(logits)
    # 应用输入偏差，这是一个形状为[批处理，词汇长度]的掩码，
    # 输入中的每个标记以及所有“corrective”标记都设置为1.0
    return tf.multiply(probs, input_bias)

def apply_input_bias_and_extract_argmax_fn_factory(input_bias):
    def fn_factory(embedding, output_projection=None, update_embedding=True):
        def loop_function(prev, _):
            prev = project_and_apply_input_bias(prev, output_projection,
                                                input_bias)

            prev_symbol = math_ops.argmax(prev, 1)
            emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
            if not update_embedding:
                emb_prev = array_ops.stop_gradient(emb_prev)
            return emb_prev, prev_symbol

        return loop_function

    return fn_factory
































