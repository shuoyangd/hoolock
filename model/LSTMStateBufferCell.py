import torch
from torch import nn
from torch.autograd import Variable

import pdb

class LSTMStateBufferCell(nn.Module):
  """
  Think of this as a "neutered" StackLSTMCell only capable of performing pop and hold operations.
  As we don't have to do any LSTM computations, (required only for push operations),
  a lot of computation time could've been saved.
  """

  def __init__(self, hidden_size, init_hidden_var=None, init_cell_var=None):
    """

    :param stack_size:
    :param hiddens: of type torch.autograd.Variable, (seq_len, batch_size, hidden_size)
    :param cells: of type torch.autograd.Variable, (seq_len, batch_size, hidden_size)
    """
    super(LSTMStateBufferCell, self).__init__()
    self.hidden_size = hidden_size

    self.initial_hidden = self.init_hidden(init_hidden_var)
    self.initial_cell = self.init_cell(init_cell_var)

  def build_stack(self, hiddens, cells, hidden_masks, gpuid=[]):
    """
    :param hiddens: of type torch.autograd.Variable, (seq_len, batch_size, hidden_size)
    :param cells: of type torch.autograd.Variable, (seq_len, batch_size, hidden_size)
    :param hidden_masks: of type torch.autograd.Variable, (seq_len, batch_size)
    """
    self.seq_len = hiddens.size(0)
    batch_size = hiddens.size(1)

    # create stacks
    if len(gpuid) >= 1:
      dtype = torch.cuda.FloatTensor
      long_dtype = torch.cuda.LongTensor
    else:
      dtype = torch.FloatTensor
      long_dtype = torch.LongTensor

    self.hidden_stack = Variable(torch.zeros(self.seq_len + 1, batch_size, self.hidden_size).type(dtype))
    self.cell_stack = Variable(torch.zeros(self.seq_len + 1, batch_size, self.hidden_size).type(dtype))

    # push start states to the stack
    self.hidden_stack[0, :, :] = self.initial_hidden.expand((batch_size, self.hidden_size))
    self.cell_stack[0, :, :] = self.initial_cell.expand((batch_size, self.hidden_size))

    # initialize content
    self.hidden_stack[1:self.seq_len+1, :, :] = hiddens
    self.cell_stack[1:self.seq_len+1, :, :] = cells

    # for padded sequences, the pos would be 0. That's why we need a padding state in the buffer.
    self.pos = torch.sum(hidden_masks, dim=0).long() # note there is a initial state padding

  def forward(self, op):
    """

    :param op: (batch_size,), stack operations, in batch (-1 means pop, 0 means hold).
    :return: (hidden, cell): both are (batch_size, hidden_dim)
    """

    cur_hidden = self.hidden_stack[self.pos.data, range(len(self.pos)), :].clone()
    cur_cell = self.cell_stack[self.pos.data, range(len(self.pos)), :].clone()
    prev_hidden = self.hidden_stack[((self.pos - 1) % (self.seq_len + 1)).data, range(len(self.pos)), :].clone()
    prev_cell = self.cell_stack[((self.pos - 1) % (self.seq_len + 1)).data, range(len(self.pos)), :].clone()

    hidden_ret = torch.mm(prev_hidden.t(), torch.diag(torch.abs(op)).float()) + \
          torch.mm(cur_hidden.t(), torch.diag(1 + (-1) * torch.abs(op)).float())
    cell_ret = torch.mm(prev_cell.t(), torch.diag(torch.abs(op)).float()) + \
          torch.mm(cur_cell.t(), torch.diag(1 + (-1) * torch.abs(op)).float())

    self.pos += op
    return hidden_ret.t(), cell_ret.t()

  def init_hidden(self, init_var=None):
    if init_var is None:
      return Variable(torch.rand((self.hidden_size,)))
    else:
      return init_var

  def init_cell(self, init_var=None):
    if init_var is None:
      return Variable(torch.rand((self.hidden_size,)))
    else:
      return init_var

  def head(self):
    # print(self.pos.data.tolist())
    # print(len(self.pos))
    # print(self.hidden_stack.size())
    dtype = self.hidden_stack.long().data.type()
    return self.hidden_stack[self.pos.data, torch.arange(0, len(self.pos)).type(dtype), :]

  def size(self):
    return torch.min(self.pos + 1).data[0]
