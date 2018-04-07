import torch
from torch import nn
from torch.autograd import Variable

import pdb

class StateStack(nn.Module):
  """
  Think of this as a "neutered" StackLSTMCell only capable of performing pop and hold operations.
  As we don't have to do any LSTM computations, (required only for push operations),
  a lot of computation time could've been saved.
  """

  def __init__(self, hidden_size, init_var=None):

    super(StateStack, self).__init__()
    self.hidden_size = hidden_size

    self.initial_hidden = self.init(init_var)

  def build_stack(self, batch_size, seq_len, hiddens=None, cells=None, hidden_masks=None, gpuid=[]):
    """
    :param hiddens: of type torch.autograd.Variable, (seq_len, batch_size, hidden_size)
    :param cells: of type torch.autograd.Variable, (seq_len, batch_size, hidden_size)
    :param hidden_masks: of type torch.autograd.Variable, (seq_len, batch_size)
    """
    if hiddens is not None:
      self.seq_len = hiddens.size(0)
      self.batch_size = hiddens.size(1)
    else:
      self.seq_len = seq_len
      self.batch_size = batch_size

    # create stacks
    if len(gpuid) >= 1:
      self.dtype = torch.cuda.FloatTensor
      self.long_dtype = torch.cuda.LongTensor
    else:
      self.dtype = torch.FloatTensor
      self.long_dtype = torch.LongTensor

    self.hidden_stack = Variable(torch.zeros(self.seq_len + 2, self.batch_size, self.hidden_size).type(self.dtype))

    # push start states to the stack
    self.hidden_stack[0, :, :] = self.initial_hidden.expand((self.batch_size, self.hidden_size))

    # initialize content
    if hiddens is not None:
      # pdb.set_trace()
      self.hidden_stack[1:self.seq_len+1, :, :] = hiddens

    # for padded sequences, the pos would be 0. That's why we need a padding state in the buffer.
    if hidden_masks is not None:
      self.pos = torch.sum(hidden_masks, dim=0).long() # note there is a initial state padding
    else:
      self.pos = Variable(torch.LongTensor([0] * self.batch_size).type(self.long_dtype))

  def forward(self, input, op):
    """

    :param input: (batch_size, input_size), input tensor, in batch.
    :param op: (batch_size,), stack operations, in batch (-1 means pop, 0 means hold, 1 means push).
    :return: hidden (batch_size, hidden_dim)
    """

    indexes = torch.arange(0, self.batch_size).type(self.long_dtype)
    self.hidden_stack[(self.pos + 1).data, indexes, :] = input
    self.pos += op
    hidden_ret = self.hidden_stack[self.pos.data, indexes, :].clone()
    return hidden_ret

  def init(self, init_var=None):
    if init_var is None:
      return Variable(torch.rand((self.hidden_size,)))
    else:
      return init_var

  def head(self):
    dtype = self.hidden_stack.long().data.type()
    ret = self.hidden_stack[self.pos.data, torch.arange(0, len(self.pos)).type(dtype), :].clone()
    return ret

  def size(self):
    return torch.min(self.pos + 1).data[0]
