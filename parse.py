from preprocess import conll_indice_mapping_without_padding

import torch
from torch import cuda
from torch.autograd import Variable
import utils

import argparse
import dill
import logging
import math
import os
import pdb

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

opt_parser =\
  argparse.ArgumentParser(description="Reimplementation of\n" +
                                      "Dyer et al. 2015\n" +
                                      "Transition-Based Dependency Parsing with Stack Long Short-Term Memory.\n" +
                                      "This is the inference code.")

# io & system
opt_parser.add_argument("--input_file", required=True,
                        help="Input to the parser.")
opt_parser.add_argument("--model_file", required=True,
                        help="Parser model dump.")
opt_parser.add_argument("--data_file", required=True,
                        help="Data dump from preprocessing script -- only .dict and .pre are used.")
opt_parser.add_argument("--output_file", required=True,
                        help="Output parse file.")
opt_parser.add_argument("--pre_emb_file", default=None,
                        help="Preprocessed pre-trained word embedding vocabulary file.")
opt_parser.add_argument("--batch_size", default=80, type=int,
                        help="Size of the test batch. (default=80)")
opt_parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                        help="ID of gpu device to use. Empty implies cpu usage.")
# parser model definition
opt_parser.add_argument("--max_step_length", default=150, type=int,
                        help="Maximum step length allowed for decoding. (default=150)")
opt_parser.add_argument("--stack_size", default=25, type=int,
                        help="Stack size reserved for the stack LSTM components. (default=25)")

def main(options):
  use_cuda = (len(options.gpuid) >= 1)
  if options.gpuid:
    cuda.set_device(options.gpuid[0])

  vocab, postags, actions = torch.load(open(options.data_file + ".dict", 'rb'), pickle_module=dill) # always load to cpu
  use_pretrained_emb = os.path.isfile(options.data_file + ".pre")
  if use_pretrained_emb:
    _, _, pre_vocab = torch.load(open(options.data_file + ".pre", 'rb'), pickle_module=dill)
  else:
    pre_vocab = None

  postag2idx = dict((pair[1], pair[0]) for pair in enumerate(postags))
  test_data, test_postag = conll_indice_mapping_without_padding(options.input_file, vocab, postag2idx)
  if use_pretrained_emb:
    test_data_pre, _ = conll_indice_mapping_without_padding(options.input_file, pre_vocab, postag2idx)
  test_data_lengths = torch.FloatTensor([ len(test_data[i]) for i in range(len(test_data)) ]).long()
  test_data_lengths = utils.tensor.truncate_or_pad(test_data_lengths, 0,
                               math.ceil(len(test_data_lengths) / options.batch_size) * options.batch_size)
  batchized_test_data, _ = utils.tensor.advanced_batchize_no_sort(test_data, options.batch_size, vocab.stoi["<pad>"])
  batchized_test_postag, _ = utils.tensor.advanced_batchize_no_sort(test_postag, options.batch_size, postags.index("<pad>"))
  if use_pretrained_emb:
    batchized_test_data_pre, _ = utils.tensor.advanced_batchize_no_sort(test_data_pre, options.batch_size, pre_vocab.stoi["<pad>"])

  parser = torch.load(open(options.model_file, 'rb'), map_location=lambda storage, loc: storage)
  parser.gpuid = options.gpuid
  if use_cuda:
    parser.cuda()
  else:
    parser.cpu()
    parser.dtype = torch.FloatTensor
    parser.long_dtype = torch.LongTensor

  pad_idx = actions.index("<pad>")
  total_sent_n = len(test_data)
  writer = utils.io.OracleWriter(open(options.output_file, 'w'), actions, pad_idx, total_sent_n)
  for batch_i in range(len(batchized_test_data)):
    logging.debug("{0} batch updates calculated.".format(batch_i))
    test_data_batch = Variable(batchized_test_data[batch_i], volatile=True)
    test_postag_batch = Variable(batchized_test_postag[batch_i], volatile=True)
    test_data_lengths_batch = Variable(test_data_lengths[batch_i * options.batch_size: (batch_i + 1) * options.batch_size], volatile=True)
    if use_pretrained_emb:
      test_data_pre_batch = Variable(batchized_test_data_pre[batch_i], volatile=True)
    else:
      test_data_pre_batch = None
    if use_cuda:
      test_data_batch.cuda()
      test_postag_batch.cuda()
      test_data_lengths_batch.cuda()
      if use_pretrained_emb:
        test_data_pre_batch.cuda()

    output_batch = parser(test_data_batch, pre_tokens=test_data_pre_batch, postags=test_postag_batch, tokens_lengths=test_data_lengths_batch) # (max_seq_len, batch_size, len(actions))
    _, output_actions = output_batch.max(dim=2) # (max_seq_len, batch_size)
    writer.writesent(output_actions)

  writer.close()

if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
          opt_parser.parse_known_args()[1]))

  # several options that are pending:
  main(options)
