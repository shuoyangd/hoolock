import model.Loss
import model.StackLSTMParser
import utils.tensor
import utils.rand

import torch
from torch import cuda
from torch.autograd import Variable

import argparse
import dill
import logging
import pdb

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

opt_parser =\
  argparse.ArgumentParser(description="Reimplementation of\n" +
                                      "Dyer et al. 2015:\n" +
                                      "Transition-Based Dependency Parsing with Stack Long Short-Term Memory")

# io & system
opt_parser.add_argument("--data_file", required=True,
                        help="Preprocessed training data.")
opt_parser.add_argument("--model_file", required=True,
                        help="Path where model should be saved.")
opt_parser.add_argument("--batch_size", default=80, type=int,
                        help="Size of the training batch.")
opt_parser.add_argument("--dev_batch_size", default=150, type=int,
                        help="Size of the dev batch.")
opt_parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                        help="ID of gpu device to use. Empty implies cpu usage.")

# parser model definition
opt_parser.add_argument("--transSys", default=model.StackLSTMParser.TransitionSystems.AER, type=int,
                        help="Choice of transition system: 0 for ASd, 1 for AER, 2 for AH. (default=1)")
opt_parser.add_argument("--word_emb_dim", default=32, type=int,
                        help="Size of updated word embedding. (default=32)")
opt_parser.add_argument("--pre_word_emb_dim", default=100, type=int,
                        help="Size of the pre-trained word embedding. (default=100)")
opt_parser.add_argument("--postag_emb_dim", default=12, type=int,
                        help="Size of the POS tag embedding. (default=12)")
opt_parser.add_argument("--action_emb_dim", default=16, type=int,
                        help="Size of the transition action embedding. (default=16)")
opt_parser.add_argument("--hid_dim", default=100, type=int,
                        help="Size of the LSTM hidden embedding. (default=100)")
opt_parser.add_argument("--input_dim", default=50, type=int,
                        help="Size of the token composition embedding. (default=50, not stated in paper)")
opt_parser.add_argument("--state_dim", default=20, type=int,
                        help="Size of the parser state pointer (p_t). (default=20, not clearly stated in paper)")
opt_parser.add_argument("--dropout_rate", default=0.0, type=float,
                        help="Dropout rate of the LSTM and stack LSTM components. (default=0.0)")
opt_parser.add_argument("--stack_size", default=25, type=int,
                        help="Stack size reserved for the stack LSTM components. (default=25)")
opt_parser.add_argument("--max_step_length", default=150, type=int,
                        help="Maximum step length allowed for decoding.")
opt_parser.add_argument("--exposure_eps", default=1.0, type=float,
                        help="The fraction of timesteps where the parser has exposure to" +
                             "the golden transition operations during training.")

# optimizer
opt_parser.add_argument("--epochs", default=20, type=int,
                        help="Epochs through the data.")
opt_parser.add_argument("--optimizer", default="Adam",
                        help="Choice of optimzier: SGD|Adadelta|Adam. (default=Adam)")
opt_parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                        help="Learning rate used across all optimizers. (default=1e-3)")
opt_parser.add_argument("--momentum", default=0.0, type=float,
                        help="Momentum for SGD. (default=0.0)")

def main(options):

  use_cuda = (len(options.gpuid) >= 1)
  if options.gpuid:
    cuda.set_device(options.gpuid[0])

  train_data, train_postag, train_action = torch.load(open(options.data_file + ".train", 'rb'), pickle_module=dill)
  dev_data, dev_postag, dev_action = torch.load(open(options.data_file + ".dev", 'rb'), pickle_module=dill)
  vocab, postags, actions = torch.load(open(options.data_file + ".dict", 'rb'), pickle_module=dill)

  # (num_batched_instances, seq_len, batch_size)
  batchized_train_data, sort_index = utils.tensor.advanced_batchize(train_data, options.batch_size, vocab.stoi["<pad>"])
  batchized_train_postag, _ = utils.tensor.advanced_batchize(train_postag, options.batch_size, postags.index("<pad>"))
  batchized_train_action = utils.tensor.advanced_batchize_no_sort(train_action, options.batch_size, actions.index("<pad>"), sort_index)

  batchized_dev_data, sort_index = utils.tensor.advanced_batchize(dev_data, options.dev_batch_size, vocab.stoi["<pad>"])
  batchized_dev_postag, _ = utils.tensor.advanced_batchize(dev_postag, options.dev_batch_size, postags.index("<pad>"))
  batchized_dev_action = utils.tensor.advanced_batchize_no_sort(dev_action, options.dev_batch_size, actions.index("<pad>"), sort_index)

  parser = model.StackLSTMParser.StackLSTMParser(vocab, actions, options, postags=postags)

  if use_cuda:
    parser.cuda()
  else:
    parser.cpu()

  # prepare optimizer and loss
  # loss = torch.nn.CrossEntropyLoss()
  loss = model.Loss.MaxProbLoss()
  optimizer = eval("torch.optim." + options.optimizer)(parser.parameters(), options.learning_rate)

  # main training loop
  for epoch_i in range(options.epochs):
    logging.info("At {0} epoch.".format(epoch_i))
    for i, batch_i in enumerate(utils.rand.srange(len(batchized_train_data))):
    # for i, batch_i in enumerate(range(len(batchized_train_data))):
      logging.debug("{0} batch updates calculated, with batch {1}.".format(i, batch_i))
      train_data_batch = Variable(batchized_train_data[batch_i]) # (seq_len, batch_size) with dynamic seq_len
      train_postag_batch = Variable(batchized_train_postag[batch_i]) # (seq_len, batch_size) with dynamic seq_len
      train_action_batch = Variable(batchized_train_action[batch_i]) # (seq_len, batch_size) with dynamic seq_len
      if use_cuda:
        train_data_batch = train_data_batch.cuda()
        train_postag_batch = train_postag_batch.cuda()
        train_action_batch = train_action_batch.cuda()

      output_batch = parser(train_data_batch, train_postag_batch, train_action_batch) # (seq_len, batch_size, len(actions)) with dynamic seq_len
      output_batch = output_batch.view(-1, len(actions))
      train_action_batch = train_action_batch.view(-1)
      loss_output = loss(output_batch, train_action_batch)

      optimizer.zero_grad()
      loss_output.backward()
      optimizer.step()

      """
      train_action_batch = Variable(batchized_train_action[batch_i]) # (seq_len, batch_size) with dynamic seq_len
      post_output_batch = parser(train_data_batch, train_postag_batch, train_action_batch)  # (seq_len, batch_size, len(actions)) with dynamic seq_len
      post_output_batch = post_output_batch.view(-1, len(actions))
      train_action_batch = train_action_batch.view(-1)
      post_loss_output = loss(post_output_batch, train_action_batch)
      """

      logging.debug(loss_output)
      _, pred = output_batch.max(dim=1)
      hit = sum(map(lambda x: 1 if x[0] == x[1] else 0, zip(pred.data.tolist(), train_action_batch.data.tolist())))
      logging.debug("pred accuracy: {0}".format(hit / len(output_batch)))
      # logging.debug("pred: {0}".format(pred))
      # logging.debug("gold: {0}".format(train_action_batch))

      """
      logging.debug(post_loss_output)
      _, post_pred = post_output_batch.max(dim=1)
      hit = sum(map(lambda x: 1 if x[0] == x[1] else 0, zip(post_pred.data.tolist(), train_action_batch.data.tolist())))
      logging.debug("post pred accuracy: {0}".format(hit / len(post_output_batch)))
      """

    dev_loss = 0.0
    for i, batch_i in enumerate(range(len(batchized_dev_data))):
      logging.debug("{0} dev batch calculated.".format(i))
      dev_data_batch = Variable(batchized_dev_data[batch_i]) # (seq_len, dev_batch_size) with dynamic seq_len
      dev_postag_batch = Variable(batchized_dev_postag[batch_i]) # (seq_len, dev_batch_size) with dynamic seq_len
      dev_action_batch = Variable(batchized_dev_action[batch_i]) # (seq_len, dev_batch_size) with dynamic seq_len
      if use_cuda:
        dev_data_batch = dev_data_batch.cuda()
        dev_postag_batch = dev_postag_batch.cuda()
        dev_action_batch = dev_action_batch.cuda()

      output_batch = parser(dev_data_batch, dev_postag_batch, dev_action_batch) # (max_seq_len, dev_batch_size, len(actions))
      output_batch = output_batch[0:len(dev_action_batch), :].view(-1, len(actions))
      dev_action_batch = dev_action_batch[0:len(output_batch), :].view(-1)
      dev_loss_output = loss(output_batch, dev_action_batch)
      dev_loss += dev_loss_output

    logging.info("End of {0} epoch: ".format(epoch_i))
    logging.info("Training loss: {0}".format(loss_output))
    logging.info("Dev loss: {0}".format(dev_loss / i))

    logging.info("Saving model...")
    torch.save(parser, open(options.model_file + ".{0}".format(epoch_i)), pickle_module=dill)
    logging.info("Done.")

if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
          opt_parser.parse_known_args()[1]))

  # several options that are pending:
  main(options)
