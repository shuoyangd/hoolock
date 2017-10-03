import torch
from torch import nn

class MaxProbLoss(nn.Module):
  def __init__(self):
    super(MaxProbLoss, self).__init__()

  def forward(self, input, target):
    logprob = torch.sum(torch.log(input[torch.arange(0, len(target)).long(), target.data]))
    return -logprob / len(target)

