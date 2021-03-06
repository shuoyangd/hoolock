import model.StackLSTMParser
import utils.tensor
import utils.rand

import torch
from torch import cuda
# from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler

import argparse
import dill
import logging
import os
import pdb
import random

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

opt_parser =\
  argparse.ArgumentParser(description="A GPU-friendly Reimplementation of\n" +
                                      "Dyer et al. 2015\n" +
                                      "Transition-Based Dependency Parsing with Stack Long Short-Term Memory.\n" +
                                      "This is the training code.")

# io & system
opt_parser.add_argument("--data_file", required=True,
                        help="Preprocessed training data.")
opt_parser.add_argument("--model_file", required=True,
                        help="Path where model should be saved.")
opt_parser.add_argument("--batch_size", default=128, type=int,
                        help="Size of the training batch. (default=80)")
opt_parser.add_argument("--dev_batch_size", default=150, type=int,
                        help="Size of the dev batch. (default=150)")
opt_parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                        help="ID of gpu device to use. Empty implies cpu usage. (default=[])")
opt_parser.add_argument("--manual_seed", default=0, type=int,
                        help="Manual seed for reproducibility. Set to -1 to introduce real randomness. (default=0)")

# parser model definition
opt_parser.add_argument("--transSys", default=3, type=int,
                        help="Choice of transition system: 0 for ASd, 1 for AER, 2 for AES, 3 for AH. (default=1)")
opt_parser.add_argument("--word_emb_dim", default=32, type=int,
                        help="Size of updated word embedding. (default=32)")
opt_parser.add_argument("--postag_emb_dim", default=12, type=int,
                        help="Size of the POS tag embedding. (default=12)")
opt_parser.add_argument("--action_emb_dim", default=48, type=int,
                        help="Size of the transition action embedding. (default=48)")
opt_parser.add_argument("--hid_dim", default=200, type=int,
                        help="Size of the LSTM hidden embedding. (default=200)")
opt_parser.add_argument("--input_dim", default=60, type=int,
                        help="Size of the token composition embedding. (default=60)")
opt_parser.add_argument("--state_dim", default=200, type=int,
                        help="Size of the parser state pointer (p_t). (default=200)")
opt_parser.add_argument("--dropout_rate", default=0.0, type=float,
                        help="Dropout rate of the LSTM and stack LSTM components. (default=0.0)")
opt_parser.add_argument("--stack_size", default=100, type=int,
                        help="Stack size reserved for the stack LSTM components. (default=100)")
opt_parser.add_argument("--max_step_length", default=150, type=int,
                        help="Maximum step length allowed for decoding. (default=150)")
opt_parser.add_argument("--exposure_eps", default=1.0, type=float,
                        help="The fraction of timesteps where the parser has exposure to" +
                             "the golden transition operations during training. (default=1.0)")
opt_parser.add_argument("--num_lstm_layers", default=2, type=int,
                        help="Number of StackLSTM and buffer-side LSTM layers. (default=2)")

# optimizer
opt_parser.add_argument("--epochs", default=30, type=int,
                        help="Epochs through the data. (default=30)")
opt_parser.add_argument("--min_lr", default=1e-5, type=float,
                        help="Stop training when learning rate falls below this threshold. (default=1e-5)")
opt_parser.add_argument("--optimizer", default="Adam",
                        help="Choice of optimzier: SGD|Adadelta|Adam. (default=Adam)")
opt_parser.add_argument("--learning_rate", "-lr", default=0.02, type=float,
                        help="Learning rate used across all optimizers. (default=0.02)")
opt_parser.add_argument("--momentum", default=0.0, type=float,
                        help="Momentum for SGD. (default=0.0)")
opt_parser.add_argument("--grad_clip", default=-1.0, type=float,
                        help="Gradient clipping bound. Pass any negative number to disable this functionality. (default=-1.0)")
opt_parser.add_argument("--l2_norm", default=1e-6, type=float,
                        help="L2 norm coefficient to the loss function. (default=1e-6)")
opt_parser.add_argument("--lr_decay", default="ReduceLROnPLateau",
                        help="Learning rate decay scheduler: Dyer|ExponentialLR|ReduceLROnPLateau|None. Dyer means the scheduler used in (Dyer et al. 2015). (default=ReduceLROnPLateau)")
opt_parser.add_argument("--lr_decay_factor", default=0.5, type=float,
                        help="Decay factor of the learning rate. (default=0.5)")
opt_parser.add_argument("--warmup_init_lr", default=5e-4, type=float,
                        help="The initial learning rate before warmup. (default=5e-4)")
opt_parser.add_argument("--warmup_epochs", default=5, type=int,
                        help="Epochs of warmup you want to use. Set to 0 if you don't want warmup. (default=5)")


def main(options):

  use_cuda = (len(options.gpuid) >= 1)
  if options.gpuid:
    cuda.set_device(options.gpuid[0])
    occupy = torch.rand(1) # occupy gpu asap
    occupy.cuda()

  if options.manual_seed != -1:
    torch.manual_seed(options.manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(options.manual_seed)

  train_data, train_postag, train_action = torch.load(open(options.data_file + ".train", 'rb'), pickle_module=dill)
  dev_data, dev_postag, dev_action = torch.load(open(options.data_file + ".dev", 'rb'), pickle_module=dill)
  vocab, postags, actions = torch.load(open(options.data_file + ".dict", 'rb'), pickle_module=dill)
  singletons = set([vocab.stoi[w] for w in vocab.itos if vocab.freqs[w] == 1])
  use_pretrained_emb = os.path.isfile(options.data_file + ".pre")
  if use_pretrained_emb:
    train_data_pre, dev_data_pre, pre_vocab = torch.load(open(options.data_file + ".pre", 'rb'), pickle_module=dill)
  else:
    pre_vocab = None

  # create labeled to unlabeled action maps so we can report results without labels
  la2ua = {}
  u_actions = []
  for la_idx, action in enumerate(actions):
    if '|' in action:
      u_action = action.split('|')[0]
    else:
      u_action = action

    if u_action in u_actions:
      ua_idx = u_actions.index(u_action)
      la2ua[la_idx]= ua_idx
    else:
      ua_idx = len(u_actions)
      u_actions.append(u_action)
      la2ua[la_idx] = ua_idx

  # (num_batched_instances, seq_len, batch_size)
  batchized_train_data, batchized_train_data_mask, sort_index = utils.tensor.advanced_batchize(train_data, options.batch_size, vocab.stoi["<pad>"])
  batchized_train_postag, _, _ = utils.tensor.advanced_batchize(train_postag, options.batch_size, postags.index("<pad>"))
  batchized_train_action, batchized_train_action_mask = utils.tensor.advanced_batchize_no_sort(train_action, options.batch_size, actions.index("<pad>"), sort_index)
  if use_pretrained_emb:
    batchized_train_data_pre, _ = utils.tensor.advanced_batchize_no_sort(train_data_pre, options.batch_size, pre_vocab.stoi["<pad>"], sort_index)

  batchized_dev_data, batchized_dev_data_mask, sort_index = utils.tensor.advanced_batchize(dev_data, options.dev_batch_size, vocab.stoi["<pad>"])
  batchized_dev_postag, _, _ = utils.tensor.advanced_batchize(dev_postag, options.dev_batch_size, postags.index("<pad>"))
  batchized_dev_action, batchized_dev_action_mask = utils.tensor.advanced_batchize_no_sort(dev_action, options.dev_batch_size, actions.index("<pad>"), sort_index)
  if use_pretrained_emb:
    batchized_dev_data_pre, _ = utils.tensor.advanced_batchize_no_sort(dev_data_pre, options.dev_batch_size, pre_vocab.stoi["<pad>"], sort_index)

  parser = model.StackLSTMParser.StackLSTMParser(vocab, actions, options, pre_vocab=pre_vocab, postags=postags)

  if use_cuda:
    parser.cuda()
  else:
    parser.cpu()

  for param in parser.parameters():
    if len(param.size()) > 1:
      torch.nn.init.xavier_uniform_(param)

  # prepare optimizer and loss
  # loss = torch.nn.CrossEntropyLoss()
  # loss = model.Loss.NLLLoss(options.gpuid)
  loss = torch.nn.NLLLoss()
  if options.warmup_epochs == 0:
    optimizer = eval("torch.optim." + options.optimizer)(filter(lambda param: param.requires_grad, parser.parameters()), options.learning_rate, weight_decay=options.l2_norm)
  else:
    optimizer = eval("torch.optim." + options.optimizer)(filter(lambda param: param.requires_grad, parser.parameters()), options.warmup_init_lr, weight_decay=options.l2_norm)

  if options.lr_decay == "None":
    base_scheduler = None
  elif options.lr_decay == "Dyer":
    base_scheduler = LambdaLR(optimizer, lambda t: 1.0 / (1 + options.lr_decay_factor * t))
  else:
    base_scheduler = eval(options.lr_decay)(optimizer, 'min', options.lr_decay_factor, patience=0)

  if base_scheduler is None or options.warmup_epochs == 0:
    scheduler = base_scheduler
  else:
    scheduler = GradualWarmupScheduler(optimizer, multiplier=options.learning_rate / options.warmup_init_lr, total_epoch=options.warmup_epochs, after_scheduler=base_scheduler)


  def replace_singletons(batch, singletons, unk_idx):
    """
      assuming <pad> will be seen more than once

      :param batch:
      :param singletons:
      :param unk_idx:
      :return:
    """

    shape = batch.size()
    lin_batch = batch.clone().view(-1)  # shouldn't replace in-place

    for idx, w in enumerate(lin_batch):
      if int(w) in singletons and torch.rand(1)[0] < 0.5:
        lin_batch[idx] = unk_idx

    return lin_batch.view(shape)


  # main training loop
  for epoch_i in range(options.epochs):

    if optimizer.param_groups[0]['lr'] < options.min_lr and epoch_i > options.warmup_epochs:
      logging.info("Training finished at epoch {0} because the learning rate {1} is smaller than minimum {2}.".format( \
          epoch_i-1, optimizer.param_groups[0]['lr'], options.min_lr \
        ) \
      )
      break

    logging.info("At {0} epoch.".format(epoch_i))
    if type(base_scheduler) != ReduceLROnPlateau:
      scheduler.step()
      logging.info("Current learning rate {0}".format(optimizer.param_groups[0]['lr']))

    for i, batch_i in enumerate(utils.rand.srange(len(batchized_train_data))):
      logging.debug("{0} batch updates calculated, with batch {1}.".format(i, batch_i))
      train_data_batch = replace_singletons(batchized_train_data[batch_i], singletons, vocab.stoi["<unk>"])
      train_data_mask_batch = batchized_train_data_mask[batch_i] # (seq_len, batch_size) with dynamic seq_len
      train_postag_batch = batchized_train_postag[batch_i] # (seq_len, batch_size) with dynamic seq_len
      train_action_batch = batchized_train_action[batch_i] # (seq_len, batch_size) with dynamic seq_len
      train_action_mask_batch = batchized_train_action_mask[batch_i] # (seq_len, batch_size) with dynamic seq_len

      if use_pretrained_emb:
        train_data_pre_batch = batchized_train_data_pre[batch_i]
      else:
        train_data_pre_batch = None

      if use_cuda:
        train_data_batch = train_data_batch.cuda()
        train_data_mask_batch = train_data_mask_batch.cuda()
        train_postag_batch = train_postag_batch.cuda()
        train_action_batch = train_action_batch.cuda()
        train_action_mask_batch = train_action_mask_batch.cuda()
        if use_pretrained_emb:
          train_data_pre_batch = train_data_pre_batch.cuda()

      output_batch = parser(train_data_batch, train_data_mask_batch, train_data_pre_batch, train_postag_batch, train_action_batch) # (seq_len, batch_size, len(actions)) with dynamic seq_len
      output_batch = output_batch.view(-1, len(actions)) # (seq_len * batch_size, len(actions))
      train_action_batch = train_action_batch.view(-1) # (seq_len * batch_size)
      train_action_mask_batch = train_action_mask_batch.view(-1) # (seq_len * batch_size)
      output_batch = output_batch.masked_select(train_action_mask_batch.unsqueeze(1).expand(len(train_action_mask_batch), len(actions))).view(-1, len(actions))
      train_action_batch = train_action_batch.masked_select(train_action_mask_batch)
      loss_output = loss(output_batch, train_action_batch)

      optimizer.zero_grad()
      loss_output.backward()

      if options.grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm(parser.parameters(), options.grad_clip)
      optimizer.step()

      logging.debug(loss_output.item())
      _, pred = output_batch.max(dim=1)

      u_pred = map(lambda x: la2ua[x], pred.tolist())
      u_train_action_batch = map(lambda x: la2ua[x], train_action_batch.tolist())
      hit = sum(map(lambda x: 1 if x[0] == x[1] else 0, zip(u_pred, u_train_action_batch)))
      logging.debug("pred accuracy: {0}".format(hit / len(output_batch)))

      # minimize pytorch memory usage
      del train_data_batch
      del train_data_mask_batch
      del train_postag_batch
      del train_action_batch
      if use_pretrained_emb:
        del train_data_pre_batch

    dev_loss = 0.0
    for i, batch_i in enumerate(range(len(batchized_dev_data))):
      logging.debug("{0} dev batch calculated.".format(i))

      dev_data_batch = batchized_dev_data[batch_i] # (seq_len, dev_batch_size) with dynamic seq_len
      dev_data_mask_batch = batchized_dev_data_mask[batch_i] # (seq_len, dev_batch_size) with dynamic seq_len
      dev_postag_batch = batchized_dev_postag[batch_i] # (seq_len, dev_batch_size) with dynamic seq_len
      dev_action_batch = batchized_dev_action[batch_i] # (seq_len, dev_batch_size) with dynamic seq_len
      dev_action_mask_batch = batchized_dev_action_mask[batch_i] # (seq_len, dev_batch_size) with dynamic seq_len
      if use_pretrained_emb:
        dev_data_pre_batch = batchized_dev_data_pre[batch_i]
      else:
        dev_data_pre_batch = None

      if use_cuda:
        dev_data_batch = dev_data_batch.cuda()
        dev_data_mask_batch = dev_data_mask_batch.cuda()
        dev_postag_batch = dev_postag_batch.cuda()
        dev_action_batch = dev_action_batch.cuda()
        dev_action_mask_batch = dev_action_mask_batch.cuda()
        if use_pretrained_emb:
          dev_data_pre_batch = dev_data_pre_batch.cuda()

      output_batch = parser(dev_data_batch, dev_data_mask_batch, dev_data_pre_batch, dev_postag_batch, dev_action_batch) # (max_seq_len, dev_batch_size, len(actions))
      output_batch = output_batch[0:len(dev_action_batch), :].view(-1, len(actions))
      dev_action_batch = dev_action_batch[0:len(output_batch), :].view(-1)
      dev_action_mask_batch = dev_action_mask_batch.view(-1)
      output_batch = output_batch.masked_select(dev_action_mask_batch.unsqueeze(1).expand(len(dev_action_mask_batch), len(actions))).view(-1, len(actions))
      dev_action_batch = dev_action_batch.masked_select(dev_action_mask_batch) # (seq_len * batch_size)
      batch_dev_loss = loss(output_batch, dev_action_batch)
      dev_loss += batch_dev_loss.item()

    logging.info("End of {0} epoch: ".format(epoch_i))
    logging.info("Dev loss: {0}".format(dev_loss))

    logging.info("Saving model...")
    torch.save(parser, open(options.model_file + ".nll_{0:.2f}.epoch_{1}".format(dev_loss, epoch_i), 'wb'),
               pickle_module=dill)
    logging.info("Done.")

    if type(base_scheduler) == ReduceLROnPlateau:
      scheduler.step(metrics=dev_loss)
      logging.info("Current learning rate {0}".format(optimizer.param_groups[0]['lr']))


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
          opt_parser.parse_known_args()[1]))

  # several options that are pending:
  main(options)
