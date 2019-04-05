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

    self.batch_indexes = torch.arange(0, batch_size).type(self.long_dtype)

  def forward(self, input, op):
    """
    stack needs to be built before this is called.

    :param input: (batch_size, input_size), input vector, in batch.
    :param op: (batch_size,), stack operations, in batch (-1 means pop, 1 means push, 0 means hold).
    :return: (hidden, cell): both are (batch_size, hidden_dim)
    """

    batch_size = input.size(0)
    self.hidden_stack[(self.pos + 1).data, self.batch_indexes, :] = input.clone()
    self.pos = self.pos + op  # XXX: should NOT use in-place assignment!
    pos = self.pos.unsqueeze(0).unsqueeze(2).expand(1, batch_size, self.hidden_size)
    hidden_ret = torch.gather(self.hidden_stack, 0, pos).squeeze()
    return hidden_ret

    """
    batch_size = input.size(0)
    push_indexes = torch.arange(0, batch_size).type(self.long_dtype)[(op == 1).data]
    pop_indexes = torch.arange(0, batch_size).type(self.long_dtype)[(op == -1).data]
    hold_indexes = torch.arange(0, batch_size).type(self.long_dtype)[(op == 0).data]

    if len(push_indexes) != 0:
      input = input[push_indexes, :]
      # cur_hidden = self.hidden_stack[self.pos[push_indexes].data, push_indexes, :, :].clone()  # (push_indexes, hidden_size, num_layers)
      # cur_cell = self.cell_stack[self.pos[push_indexes].data, push_indexes, :, :].clone()  # (push_indexes, hidden_size, num_layers)
      # next_hidden, next_cell = self.lstm(input, (cur_hidden, cur_cell))
      next_hidden = input.clone()
    if len(pop_indexes) != 0:
      prev_hidden = self.hidden_stack[((self.pos[pop_indexes] - 1) % (self.seq_len + 1)).data, pop_indexes, :].clone()
      # prev_cell = self.cell_stack[((self.pos[pop_indexes] - 1) % (self.stack_size + 1)).data, pop_indexes, :, :].clone()
      # next_hidden, next_cell = self.lstm(input, (prev_hidden, prev_cell))

    if len(push_indexes) != 0:
      self.hidden_stack[(self.pos[push_indexes] + 1).data, push_indexes, :] = next_hidden

    if len(hold_indexes) != 0:
      cur_hidden = self.hidden_stack[self.pos[hold_indexes].data, hold_indexes, :].clone()  # (hold_indexes, hidden_size, num_layers)
      # cur_cell = self.cell_stack[self.pos[hold_indexes].data, hold_indexes, :, :].clone()  # (hold_indexes, hidden_size, num_layers)

    # we only care about the hidden & cell states of the last layer for return values
    hidden_ret = Variable(torch.zeros(batch_size, self.hidden_size).type(self.dtype))
    # cell_ret = Variable(torch.zeros(batch_size, self.hidden_size).type(self.dtype))
    if len(hold_indexes) != 0:
      hidden_ret[hold_indexes] = cur_hidden
      # cell_ret[hold_indexes] = cur_cell[:, :, -1]
    if len(push_indexes) != 0:
      hidden_ret[push_indexes] = next_hidden
      # cell_ret[push_indexes] = next_cell[:, :, -1]
    if len(pop_indexes) != 0:
      hidden_ret[pop_indexes] = prev_hidden
      # cell_ret[pop_indexes] = prev_cell[:, :, -1]

    # position overflow/underflow protection should not be done here,
    # they may not be desirable depending on the application
    # self.pos += op
    self.pos = self.pos + op  # XXX: should NOT use in-place assignment!

    # return hidden_ret, cell_ret
    return hidden_ret
    """

  """
  def forward(self, input, op):

    :param input: (batch_size, input_size), input tensor, in batch.
    :param op: (batch_size,), stack operations, in batch (-1 means pop, 0 means hold, 1 means push).
    :return: hidden (batch_size, hidden_dim)

    indexes = torch.arange(0, self.batch_size).type(self.long_dtype)
    self.hidden_stack[(self.pos + 1).data, indexes, :] = input
    self.pos += op
    hidden_ret = self.hidden_stack[self.pos.data, indexes, :].clone()
    return hidden_ret
  """

  def init(self, init_var=None):
    if init_var is None:
      return Variable(torch.rand((self.hidden_size,)))
    else:
      return init_var

  def head(self):
    batch_size = len(self.pos)
    pos = self.pos.unsqueeze(0).unsqueeze(2).expand(1, batch_size, self.hidden_size)
    ret = torch.gather(self.hidden_stack, 0, pos).squeeze()
    return ret

  def size(self):
    return torch.min(self.pos + 1).data[0]
