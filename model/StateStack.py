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

  def build_stack(self, batch_size, seq_len, hiddens=None, hidden_masks=None, gpuid=[]):
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
      self.pos = torch.sum(hidden_masks, dim=0).type(self.long_dtype) # note there is a initial state padding
    else:
      self.pos = Variable(torch.LongTensor([0] * self.batch_size).type(self.long_dtype))

  def forward(self, input, op):
    """
    stack needs to be built before this is called.

    :param input: (batch_size, input_size), input vector, in batch.
    :param op: (batch_size,), stack operations, in batch (-1 means pop, 1 means push, 0 means hold).
    :return: (hidden, cell): both are (batch_size, hidden_dim)
    """

    batch_size = input.size(0)
    batch_indexes = torch.arange(0, batch_size).type(self.long_dtype)
    self.hidden_stack[self.pos + 1, batch_indexes, :] = input.clone()
    self.pos = self.pos + op  # XXX: should NOT use in-place assignment!
    hidden_ret = self.hidden_stack[self.pos, batch_indexes, :]
    return hidden_ret

  def init(self, init_var=None):
    if init_var is None:
      return Variable(torch.rand((self.hidden_size,)))
    else:
      return init_var

  def head(self):
    dtype = self.hidden_stack.long().type()
    ret = self.hidden_stack[self.pos, torch.arange(0, len(self.pos)).type(dtype), :].clone()
    return ret

  def size(self):
    return torch.min(self.pos + 1)[0]


class StateReducer(StateStack):

  def __init__(self, hidden_size, init_var=None):
    super(StateReducer, self).__init__(hidden_size, init_var)

    self.reducer = nn.Sequential(
      nn.Linear(self.hidden_size * 2, self.hidden_size),
      nn.Tanh())

  def forward(self, input, op, dir_):
    batch_size = input.size(0)
    batch_indexes = torch.arange(0, batch_size).type(self.long_dtype)

    # take care of the shifts
    self.hidden_stack[self.pos + 1, batch_indexes, :] = input.clone()

    # take care of the reduces
    cur_hidden = self.hidden_stack[self.pos, batch_indexes, :].clone()
    prev_hidden = self.hidden_stack[self.pos - 1, batch_indexes, :].clone()  # (batch_size, hidden_size)

    is_left_reduce = (op == -1) & (dir_ == 0)
    is_right_reduce = (op == -1) & (dir_ == 1)
    reducing_ret = torch.zeros(batch_size, self.hidden_size).type(self.dtype)
    if len(is_left_reduce) != 0:
      left_reduced_indexes = batch_indexes[is_left_reduce]
      left_reduced_pos = (self.pos - 1)[is_left_reduce]
      left_composed = self.reducer(torch.cat([cur_hidden, prev_hidden], dim=1))[is_left_reduce, :]
      self.hidden_stack[left_reduced_pos, left_reduced_indexes, :] = left_composed
      reducing_ret[is_left_reduce, :] = left_composed

    if len(is_right_reduce) != 0:
      right_reduced_indexes = batch_indexes[is_right_reduce]
      right_reduced_pos =  (self.pos - 1)[is_right_reduce]
      right_composed = self.reducer(torch.cat([prev_hidden, cur_hidden], dim=1))[is_right_reduce, :]
      self.hidden_stack[right_reduced_pos, right_reduced_indexes, :] = right_composed
      reducing_ret[is_right_reduce, :] = right_composed

    self.pos = self.pos + op  # XXX: should NOT use in-place assignment!
    hidden_ret = self.hidden_stack[self.pos, batch_indexes, :]

    return hidden_ret, reducing_ret

