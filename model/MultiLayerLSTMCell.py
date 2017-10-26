import torch
from torch import nn
from torch.autograd import Variable

import pdb

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

    dtype = torch.cuda.FloatTensor if next(self.lstm.parameters()).is_cuda else torch.FloatTensor
    next_hidden = Variable(torch.zeros(prev[0].size()).type(dtype))
    next_cell = Variable(torch.zeros(prev[1].size()).type(dtype))

    for i in range(self.num_layers):
      prev_hidden_i = prev[0][:, :, i]
      prev_cell_i = prev[1][:, :, i]
      if i == 0:
        next_hidden_i, next_cell_i = self.lstm[i](input, (prev_hidden_i, prev_cell_i))
      else:
        next_hidden_i, next_cell_i = self.lstm[i](input_im1, (prev_hidden_i, prev_cell_i))
      next_hidden[:, :, i] = next_hidden_i
      next_cell[:, :, i] = next_cell_i
      input_im1 = next_hidden_i

    return next_hidden, next_cell
