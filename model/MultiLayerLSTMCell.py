import torch
from torch import nn
from torch.autograd import Variable

import pdb


class MultiLayerRNNCell(nn.Module):

  def __init__(self, input_size, hidden_size, num_layers, bias=True):
    super(MultiLayerRNNCell, self).__init__()
    self.rnn = nn.ModuleList()
    self.rnn.append(nn.RNNCell(input_size, hidden_size))
    for i in range(num_layers - 1):
      self.rnn.append(nn.RNNCell(hidden_size, hidden_size))
    self.num_layers = num_layers

  def forward(self, input, prev):
    """

    :param input: (batch_size, input_size)
    :param prev: tuple of (h0, c0), each has size (batch, hidden_size, num_layers)
    """

    next_hidden = []

    for i in range(self.num_layers):
      prev_hidden_i = prev[0][:, :, i]
      if i == 0:
        next_hidden_i = self.rnn[i](input, prev_hidden_i)
      else:
        next_hidden_i = self.rnn[i](input_im1, prev_hidden_i)
      next_hidden += [next_hidden_i]
      input_im1 = next_hidden_i

    next_hidden = torch.stack(next_hidden).permute(1, 2, 0)

    return next_hidden, torch.zeros_like(next_hidden)


class MultiLayerLSTMCell(nn.Module):

  def __init__(self, input_size, hidden_size, num_layers, bias=True):
    super(MultiLayerLSTMCell, self).__init__()
    self.lstm = nn.ModuleList()
    self.lstm.append(nn.LSTMCell(input_size, hidden_size))
    for i in range(num_layers - 1):
      self.lstm.append(nn.LSTMCell(hidden_size, hidden_size))
    self.num_layers = num_layers

  def forward(self, input, prev):
    """

    :param input: (batch_size, input_size)
    :param prev: tuple of (h0, c0), each has size (batch, hidden_size, num_layers)
    """

    next_hidden = []
    next_cell = []

    for i in range(self.num_layers):
      prev_hidden_i = prev[0][:, :, i]
      prev_cell_i = prev[1][:, :, i]
      if i == 0:
        next_hidden_i, next_cell_i = self.lstm[i](input, (prev_hidden_i, prev_cell_i))
      else:
        next_hidden_i, next_cell_i = self.lstm[i](input_im1, (prev_hidden_i, prev_cell_i))
      next_hidden += [next_hidden_i]
      next_cell += [next_cell_i]
      input_im1 = next_hidden_i

    next_hidden = torch.stack(next_hidden).permute(1, 2, 0)
    next_cell = torch.stack(next_cell).permute(1, 2, 0)

    return next_hidden, next_cell
