# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tokenization
import tensorflow as tf
import time
from fuzzywuzzy import fuzz
import spacy
import ast

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
  "input_file_np", None,
  "Input raw text file of strings of numpy arrays containing noun phrase vectors (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float("lim_prob", 0.5, "LIM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(input_files, input_files_np, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng, lim_prob, tokenizer_spacy):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]
  all_documents_np = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.

  # I should also read in input_files_no like that
  inputs = zip(input_files, input_files_np)
  for input in inputs:
    with tf.gfile.GFile(input[1], "r") as reader_np:
      with tf.gfile.GFile(input[0], "r") as reader:
        while True:
          line = tokenization.convert_to_unicode(reader.readline())
          np_line = reader_np.readline().strip()
          if np_line:
            np_line = ast.literal_eval(np_line)
          if not line:
            break
          line = line.strip()

          # Empty lines are used as document delimiters, falls eine leere line kommt, wird eine neue liste hinzugefügt in die dann die neuen tokens des neuen dokuments kommen
          if not line:
            all_documents.append([])
            all_documents_np.append([])
          tokens = tokenizer.tokenize(line) # hier wird encoded, das heißt hier sollte ich auch die funktion von line, tokens und noun phrase line eingeben
          noun_phrases = extend_np(line, tokens, np_line, tokenizer_spacy) #noun phrases has to be a list, as tokens is too, line is a str, np_line is a list
          if tokens and len(tokens) == len(noun_phrases):
            all_documents[-1].append(tokens)
            all_documents_np[-1].append(noun_phrases)

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  all_documents_np = [x for x in all_documents_np if x]

  # shuffle the lists together
  assert len(all_documents) == len(all_documents_np)
  all_documents_zipped = list(zip(all_documents, all_documents_np))
  rng.shuffle(all_documents_zipped)

  all_documents, all_documents_np = zip(*all_documents_zipped)

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)):
      instances.extend(
          create_instances_from_document(
              all_documents, document_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng, lim_prob, all_documents_np))

  rng.shuffle(instances)
  return instances


def extend_np(line, tokens_bert, np_line, tokenizer_spacy):
  """
  Returns noun phrase label list for the encoded sequence, assumption that encoding does not merge tokens which are
  splitted in spacy tokenizer (checked that in BERT and SciBERT vocab no double words are contained as one token)
  :param line: string of sentence
  :param tokens_bert: list of tokens encoding the sentence in line
  :param np_line: list with noun phrase labels of spacy tokenizer
  :return: list of noun phrase labels for encoded sentence
  """
  tokens_spacy = [tok for tok in tokenizer_spacy(line)]
  # Check if the number of tokens splitted by spacy tokenizer and the number of labels for these tokens match
  assert len(tokens_spacy) == len(np_line),'assertion here {0} {1} {2}'.format(np_line, tokens_spacy, line)

  np_bert = []
  index_bert = 0
  # Index of tokens from spacy tokenizer and the corresponding index of the list with noun phrase labels
  for index in range(len(tokens_spacy)):
    # Checks if encoded token is the same as spacy token or if the token got split up in the encoding
    try:
      if tokens_bert[index_bert] == tokens_spacy[index].text:
        # Append the label of the encoded token to the noun phrase label of the encoded string
        np_bert.append(np_line[index])
        index_bert += 1
      else:
        # Falls ein [UNK] token im encoding ist, dann wird dieser Fall hier auch abgedeckt, und [UNK] wird auch nicht aufgesplittet
        np_bert.append(np_line[index])
        joined = tokens_bert[index_bert]
        if index_bert + 1 < len(tokens_bert):
          index_bert += 1
        else:
          break
        # While the next tokens in the encoding got split up, the label of the whole token is appended to list
        while tokens_bert[index_bert].startswith('##') or tokens_bert[index_bert] in tokens_spacy[index].text or tokens_bert[index_bert] == '[UNK]':
          np_bert.append(np_line[index])
          joined = ''.join([joined, tokens_bert[index_bert]]).replace('##', '')
          index_bert += 1
          # If sentence ends with splitted word we need to exit the while loop if we already have the length of the tokens_bert
          if index_bert > len(tokens_bert)-1:
            break
          if joined == tokens_spacy[index].text:
            break
          if '[UNK]' in joined:
            if fuzz.ratio(joined.replace('[UNK]', ''), tokens_spacy[index].text) > 90:
              break
    except IndexError as index_error:
      np_bert = []
      #print('out of index error at this line: {0}'.format(tokens_spacy))
      break

  # Check if the number of encoded tokens and the number of labels for these tokens match
  #assert len(np_bert) == len(tokens_bert)
  #try:
  #  assert len(np_bert) == len(tokens_bert)
  #except AssertionError as e:
  #  e.args += (line, tokens_spacy, np_line, tokens_bert, np_bert)
  #  raise

  return np_bert


def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng, lim_prob, all_documents_np):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]
  document_np = all_documents_np[document_index]

  # hier muss ich einbauen, dass all_documents_np genauso eingelesen und gesampled wird wie all_documents und dann auch
  # 0 labels für die [Sep] und [cls hinzugefügt werden]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_chunk_np = []
  current_length = 0
  i = 0
  while i < len(document):
    segment = document[i]
    current_chunk.append(segment)

    assert len(document_np[i]) == len(segment)

    current_chunk_np.append(document_np[i])
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        np_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])
          np_a.extend(current_chunk_np[j])

        tokens_b = []
        np_b = []
        # Random next
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5:
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index]
          random_document_np = all_documents_np[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            np_b.extend(random_document_np[j])
            if len(tokens_b) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments
        # Actual next
        else:
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
            np_b.extend(current_chunk_np[j])
        # truncates the tokens and also the corresponding noun phrase vectors
        truncate_seq_pair(tokens_a, tokens_b, np_a, np_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = []
        noun_phrases = []
        segment_ids = []
        tokens.append("[CLS]")
        noun_phrases.append(0)
        segment_ids.append(0)

        assert len(np_a) == len(tokens_a)
        assert len(np_b) == len(tokens_b)

        for index, token in enumerate(tokens_a):
          tokens.append(token)
          segment_ids.append(0)
          noun_phrases.append(np_a[index])

        assert len(noun_phrases) == len(tokens)

        tokens.append("[SEP]")
        noun_phrases.append(0)
        segment_ids.append(0)

        for index, token in enumerate(tokens_b):
          tokens.append(token)
          segment_ids.append(1)
          noun_phrases.append(np_b[index])

        tokens.append("[SEP]")
        segment_ids.append(1)
        noun_phrases.append(0)

        assert len(noun_phrases) == len(tokens)

        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, lim_prob, noun_phrases)
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = []
      current_chunk_np = []
      current_length = 0
    i += 1

  return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng, lim_prob, noun_phrases):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
        token.startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  assert len(cand_indexes)+3 == len(noun_phrases)

  # If masking probability is greater than 0.5, we apply linguistically informed masking with a probability of lim_prob
  # otherwise we do normal random masking with all candidates
  if lim_prob > 0.5:
    cand_np_indexes = candidates_lim_masking(cand_indexes, noun_phrases, lim_prob, rng)
  else:
    cand_np_indexes = candidates_mlm_masking(cand_indexes)

  rng.shuffle(cand_np_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_np_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def candidates_lim_masking(cand_indexes, noun_phrases, lim_prob, rng):
  """
  Returns candidates of tokens for masking which tokens belong to a noun phrase and therefore apply linguistically
  informed masking
  :param cand_indexes: list of candidates for masking
  :param noun_phrases: noun phrase indices for the belonging tokens
  :param lim_prob: probability of noun phrase masking
  :param rng: random mode
  :return: list of candidates for masking which contain with a probability of lim_prob only noun phrase tokens and with
  a probability of (1-lim_prob) contain only candidates which not belong to a noun phrase
  """
  noun_phrase_indexes = []
  if rng.random() < lim_prob:
    for index in cand_indexes:
      if noun_phrases[index[0]] >= 1:
        noun_phrase_indexes.append(index)
  else:
    for index in cand_indexes:
      if noun_phrases[index[0]] == 0:
        noun_phrase_indexes.append(index)
  return noun_phrase_indexes


def candidates_mlm_masking(cand_indexes):
  return cand_indexes


def truncate_seq_pair(tokens_a, tokens_b, np_a, np_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    trunc_nps = np_a if len(np_a) > len(np_b) else np_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
      del trunc_nps[0]
    else:
      trunc_tokens.pop()
      trunc_nps.pop()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
      input_files.extend(tf.gfile.Glob(input_pattern))

  if FLAGS.input_file_np:
    input_files_np = []
    for input_pattern in FLAGS.input_file_np.split(","):
      input_files_np.extend(tf.gfile.Glob(input_pattern))

  assert len(input_files) == len(input_files_np)

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
      tf.logging.info("  %s", input_file)

  for input_file in input_files_np:
      tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)

  nlp = spacy.load("en_core_web_sm")
  tokenizer_spacy = nlp.Defaults.create_tokenizer(nlp)

  instances = create_training_instances(
        input_files, input_files_np, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        rng, FLAGS.lim_prob, tokenizer_spacy)

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
      tf.logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("input_file_np")
  tf.app.run()