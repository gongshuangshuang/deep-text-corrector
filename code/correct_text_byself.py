from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf

from data_reader import EOS_ID
from text_corrector_data_readers import MovieDialogReader

from text_corrector_models import TextCorrectorModel
class DefaultMovieDialogConfig():
    buckets = [(10, 10), (15, 15), (20, 20), (40, 40)]
    steps_per_checkpoint = 100
    max_steps = 2000
    max_vocabulary_size = 2000
    size = 512
    num_layers = 4
    max_gradient_norm = 5.0
    batch_size = 64
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    use_lstm = True
    use_rms_prop = False
    projection_bias = 0.0
def create_model(session, forward_only, model_path, config):
    """Create translation model and initialize or load parameters in session."""
    model = TextCorrectorModel(
        config.max_vocabulary_size,
        config.max_vocabulary_size,
        config.buckets,
        config.size,
        config.num_layers,
        config.max_gradient_norm,
        config.batch_size,
        config.learning_rate,
        config.learning_rate_decay_factor,
        use_lstm=config.use_lstm,
        forward_only=forward_only,
        config=config)
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model
def decode(sess, model, data_reader, data_to_decode, corrective_tokens=set(),
           verbose=True):
    model.batch_size = 1

    corrective_tokens_mask = np.zeros(model.target_vocab_size)
    corrective_tokens_mask[EOS_ID] = 1.0
    for token in corrective_tokens:
        corrective_tokens_mask[data_reader.convert_token_to_id(token)] = 1.0
    for tokens in data_to_decode:
        token_ids = [data_reader.convert_token_to_id(token) for token in tokens]
        matching_buckets = [b for b in range(len(model.buckets))
                            if model.buckets[b][0] > len(token_ids)]
        if not matching_buckets:
            continue
        bucket_id = min(matching_buckets)
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)
        _, _, output_logits = model.step(
            sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
            True, corrective_tokens=corrective_tokens_mask)
        oov_input_tokens = [token for token in tokens if
                            data_reader.is_unknown_token(token)]
        outputs = []
        next_oov_token_idx = 0
        for logit in output_logits:
            max_likelihood_token_id = int(np.argmax(logit, axis=1))
            if max_likelihood_token_id == EOS_ID:
                break
            token = data_reader.convert_id_to_token(max_likelihood_token_id)
            if data_reader.is_unknown_token(token):
                if next_oov_token_idx < len(oov_input_tokens):
                    token = oov_input_tokens[next_oov_token_idx]
                    next_oov_token_idx += 1
                else:
                    pass
            outputs.append(token)
        if verbose:
            decoded_sentence = " ".join(outputs)

            print("Input: {}".format(" ".join(tokens)))
            print("Output: {}\n".format(decoded_sentence))
        yield outputs
def get_corrective_tokens(data_reader, train_path):
    corrective_tokens = set()
    for source_tokens, target_tokens in data_reader.read_samples_by_string(
            train_path):
        corrective_tokens.update(set(target_tokens) - set(source_tokens))
    return corrective_tokens
def evaluate_accuracy(sess, model, data_reader, corrective_tokens, test_path,
                      max_samples=None):
    import nltk
    baseline_hypotheses = defaultdict(list)  # The model's input
    model_hypotheses = defaultdict(list)  # The actual model's predictions
    targets = defaultdict(list)  # Groundtruth

    errors = []

    n_samples_by_bucket = defaultdict(int)
    n_correct_model_by_bucket = defaultdict(int)
    n_correct_baseline_by_bucket = defaultdict(int)
    n_samples = 0
    for source, target in data_reader.read_samples_by_string(test_path):

        matching_buckets = [i for i, bucket in enumerate(model.buckets) if
                            len(source) < bucket[0]]
        if not matching_buckets:
            continue

        bucket_id = matching_buckets[0]

        decoding = next(
            decode(sess, model, data_reader, [source],
                   corrective_tokens=corrective_tokens, verbose=False))
        model_hypotheses[bucket_id].append(decoding)
        if decoding == target:
            n_correct_model_by_bucket[bucket_id] += 1
        else:
            errors.append((decoding, target))

        baseline_hypotheses[bucket_id].append(source)
        if source == target:
            n_correct_baseline_by_bucket[bucket_id] += 1
        targets[bucket_id].append([target])

        n_samples_by_bucket[bucket_id] += 1
        n_samples += 1

        if max_samples is not None and n_samples > max_samples:
            break
    for bucket_id in targets.keys():
        baseline_bleu_score = nltk.translate.bleu_score.corpus_bleu(
            targets[bucket_id], baseline_hypotheses[bucket_id])
        model_bleu_score = nltk.translate.bleu_score.corpus_bleu(
            targets[bucket_id], model_hypotheses[bucket_id])
        print("Bucket {}: {}".format(bucket_id, model.buckets[bucket_id]))
        print("\tBaseline BLEU = {:.4f}\n\tModel BLEU = {:.4f}".format(
            baseline_bleu_score, model_bleu_score))
        print("\tBaseline Accuracy: {:.4f}".format(
            1.0 * n_correct_baseline_by_bucket[bucket_id] /
            n_samples_by_bucket[bucket_id]))
        print("\tModel Accuracy: {:.4f}".format(
            1.0 * n_correct_model_by_bucket[bucket_id] /
            n_samples_by_bucket[bucket_id]))

    return errors
def train(config, model_path, train_path, val_path):
    data_reader = MovieDialogReader(config, train_path)
    train_data = data_reader.build_dataset(train_path)
    val_data = data_reader.build_dataset(val_path)
    with tf.Session() as sess:
        model = create_model(sess, False, model_path, config=config)
        train_bucket_sizes = [len(train_data[b]) for b in
                            range(len(config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
            for i in range(len(train_bucket_sizes))]
        # 循环训练
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while current_step < config.max_steps:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                            if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_data, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, False)
            print(step_loss)
            step_time += (time.time() - start_time) / config.steps_per_checkpoint
            loss += step_loss / config.steps_per_checkpoint
            current_step += 1
            if current_step % config.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("global step %d learning rate %.4f step-time %.2f "
                    "perplexity %.2f" % (
                        model.global_step.eval(), model.learning_rate.eval(),
                        step_time, perplexity))
                if len(previous_losses) > 2 and loss > max(
                        previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                checkpoint_path = os.path.join(model_path, "translate.ckpt")
                model.saver.save(sess, checkpoint_path,
                                global_step=model.global_step)
                # 在开发集上评估
                step_time, loss = 0.0, 0.0
                for bucket_id in range(len(config.buckets)):
                    if len(val_data[bucket_id]) == 0:
                        print(" eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = \
                        model.get_batch(val_data, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs,
                                             decoder_inputs,
                                             target_weights, bucket_id,
                                             True)
                    eval_ppx = math.exp(
                        float(eval_loss)) if eval_loss < 300 else float("inf")
                    print(" eval: bucket %d perplexity %.2f" % (
                        bucket_id, eval_ppx))
                sys.stdout.flush()
def test(config, model_path, test_path):
    data_reader = MovieDialogReader(config, test_path)
    with tf.Session() as session:
        print("Loaded model. Beginning decoding.")
        model = create_model(session, True, model_path, config=config)
        corrective_tokens = get_corrective_tokens(data_reader, test_path)
        evaluate_accuracy(session, model, data_reader, corrective_tokens, test_path,
                          max_samples=None)
if __name__ == '__main__':
    train_path = 'movie_dialog_train.txt'
    val_path = 'movie_dialog_val.txt'
    test_path = 'movie_dialog_test.txt'
    model_path = 'movie_dialog_model/'
    config = DefaultMovieDialogConfig()
    #train(config, model_path, train_path, val_path)
    test(config, model_path, test_path)

















