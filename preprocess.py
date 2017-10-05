import argparse
import collections
import dill
import logging
import sys
import torch
import torchtext.vocab

import utils.io
import utils.tensor

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="preprocessing")

parser.add_argument("--train_conll_file", required=True,
                    help="The parsed sentences used for training, in CoNLL-U format.")
parser.add_argument("--train_oracle_file", required=True,
                    help="The oracle transition sequence for the training data.")
parser.add_argument("--dev_conll_file", required=True,
                    help="The parsed sentences used for dev, in CoNLL-U format.")
parser.add_argument("--dev_oracle_file", required=True,
                    help="The oracle transition sequence for the dev data.")

parser.add_argument("--save_data", required=True,
                    help="Path for the binarized training & dev data.")
parser.add_argument("--vocab_size", default=50000, type=int,
                    help="Maximum vocabulary size. (default: 50000)")
parser.add_argument("--sent_len", default=80, type=int,
                    help="Maximum allowed sentence length. (default: 80)")
parser.add_argument("--action_seq_len", default=150, type=int,
                    help="Maximum allowed gold parse sequence length. (default = 150)")

def conll_indice_mapping_without_padding(path, vocab, postag2idx):
  """
  Don't do padding on the sentences, so advanced batching could be applied.

  :param path:
  :param vocab:
  :param postag2idx:
  :retrun [(sent_len,)]
  """
  conll_reader = utils.io.CoNLLReader(open(path))
  ret_tokens = []
  ret_postags = []
  token_unk_idx = vocab.stoi["<unk>"]
  postag_unk_idx = postag2idx["<unk>"]
  row_n = 0
  for sent in conll_reader:
    sent_tokens = []
    sent_postags = []
    for row in sent:
      sent_tokens.append(vocab.stoi.get(row["FORM"], token_unk_idx))
      sent_postags.append(postag2idx.get(row["UPOSTAG"], postag_unk_idx))
      row_n += 1
      if row_n % 10000 == 0:
        sys.stderr.write(".")
        sys.stderr.flush()
    sent_tokens = torch.LongTensor(sent_tokens)
    sent_postags = torch.LongTensor(sent_postags)
    ret_tokens.append(sent_tokens)
    ret_postags.append(sent_postags)
  return ret_tokens, ret_postags

def conll_indice_mapping(path, vocab, postag2idx, options):
  """

  :param path:
  :param vocab:
  :param postag2idx:
  :param options:
  :return: (num_sent, sent_len)
  """
  conll_reader = utils.io.CoNLLReader(open(path))
  ret_tokens = torch.zeros((1, options.sent_len)).long()
  ret_postags = torch.zeros((1, options.sent_len)).long()
  token_pad_idx = vocab.stoi["<pad>"]
  token_unk_idx = vocab.stoi["<unk>"]
  postag_pad_idx = postag2idx["<pad>"]
  postag_unk_idx = postag2idx["<unk>"]
  row_n = 0
  for sent in conll_reader:
    sent_tokens = []
    sent_postags = []
    for row in sent:
      sent_tokens.append(vocab.stoi.get(row["FORM"], token_unk_idx))
      sent_postags.append(postag2idx.get(row["UPOSTAG"], postag_unk_idx))
      row_n += 1
      if row_n % 10000 == 0:
        sys.stderr.write(".")
        sys.stderr.flush()
    sent_tokens = torch.LongTensor(sent_tokens)
    sent_tokens = utils.tensor.truncate_or_pad(sent_tokens, 0, options.sent_len, token_pad_idx)
    sent_postags = torch.LongTensor(sent_postags)
    sent_postags = utils.tensor.truncate_or_pad(sent_postags, 0, options.sent_len, postag_pad_idx)
    ret_tokens = torch.cat((ret_tokens, sent_tokens.unsqueeze(0)), dim=0)
    ret_postags = torch.cat((ret_postags, sent_postags.unsqueeze(0)), dim=0)
  return ret_tokens[1:], ret_postags[1:]

def oracle_indice_mapping_without_padding(path, action2idx):
  """
  Don't do padding on the sentences, so advanced batching could be applied.

  :param path:
  :param action2idx:
  :return: (num_sent, action_seq_len)
  """
  oracle_reader = utils.io.OracleReader(open(path))
  ret_actions = []
  action_unk_idx = action2idx["<unk>"]
  row_n = 0
  for sent in oracle_reader:
    sent_actions = []
    for row in sent:
      actionidx = action2idx.get("|".join(row.values()), action_unk_idx)
      sent_actions.append(actionidx)
      row_n += 1
      if row_n % 10000 == 0:
        sys.stderr.write(".")
        sys.stderr.flush()
    sent_actions = torch.LongTensor(sent_actions)
    ret_actions.append(sent_actions)
  return ret_actions

def oracle_indice_mapping(path, action2idx, options):
  """

  :param path:
  :param action2idx:
  :param options:
  :return: (num_sent, action_seq_len)
  """
  oracle_reader = utils.io.OracleReader(open(path))
  ret_actions = torch.zeros((1, options.action_seq_len)).long()
  action_pad_idx = action2idx["<pad>"]
  action_unk_idx = action2idx["<unk>"]
  row_n = 0
  for sent in oracle_reader:
    sent_actions = []
    for row in sent:
      actionidx = action2idx.get("|".join(row.values()), action_unk_idx)
      sent_actions.append(actionidx)
      row_n += 1
      if row_n % 10000 == 0:
        sys.stderr.write(".")
        sys.stderr.flush()
    sent_actions = torch.LongTensor(sent_actions)
    sent_actions = utils.tensor.truncate_or_pad(sent_actions, 0, options.action_seq_len, action_pad_idx)
    ret_actions = torch.cat((ret_actions, sent_actions.unsqueeze(0)), dim=0)
  return ret_actions[1:]

def main(options):
  # first pass: collecting vocab
  conll_reader = utils.io.CoNLLReader(open(options.train_conll_file))
  tokens = []
  postags = []
  for sent in conll_reader:
    for row in sent:
      tokens.append(row["FORM"])
      postags.append(row["UPOSTAG"])
  conll_reader.close()
  vocab = torchtext.vocab.Vocab(collections.Counter(tokens), max_size=options.vocab_size)
  postags = list(set(postags))
  postags.append("<pad>")
  postags.append("<unk>")
  postag2idx = dict((pair[1], pair[0]) for pair in enumerate(postags))

  oracle_reader = utils.io.OracleReader(open(options.train_oracle_file))
  actions = []
  for sent in oracle_reader:
    for row in sent:
      actions.append("|".join(row.values()))
  actions = list(set(actions))
  actions.append("<pad>")
  actions.append("<unk>")
  action2idx = dict((pair[1], pair[0]) for pair in enumerate(actions))

  # second pass: map data
  logging.info("input indice mapping w/ training...")
  # train_data, train_postag = conll_indice_mapping(options.train_conll_file, vocab, postag2idx, options)
  train_data, train_postag = conll_indice_mapping_without_padding(options.train_conll_file, vocab, postag2idx)
  logging.info("input indice mapping w/ dev...")
  # dev_data, dev_postag = conll_indice_mapping(options.dev_conll_file, vocab, postag2idx, options)
  dev_data, dev_postag = conll_indice_mapping_without_padding(options.dev_conll_file, vocab, postag2idx)
  logging.info("oracle indice mapping w/ training...")
  # train_action = oracle_indice_mapping(options.train_oracle_file, action2idx, options)
  train_action = oracle_indice_mapping_without_padding(options.train_oracle_file, action2idx)
  logging.info("oracle indice mapping w/ dev...")
  # dev_action = oracle_indice_mapping(options.dev_oracle_file, action2idx, options)
  dev_action = oracle_indice_mapping_without_padding(options.dev_oracle_file, action2idx)
  torch.save((train_data, train_postag, train_action), open(options.save_data + ".train", 'wb'),
             pickle_module=dill)
  torch.save((dev_data, dev_postag, dev_action), open(options.save_data + ".dev", 'wb'),
             pickle_module=dill)
  torch.save((vocab, postags, actions), open(options.save_data + ".dict", 'wb'),
             pickle_module=dill)

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
          parser.parse_known_args()[1]))
  main(options)
