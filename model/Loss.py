import torch
from torch import nn

class NLLLoss(nn.Module):
  def __init__(self, gpuid):
    super(NLLLoss, self).__init__()
    if len(gpuid) >= 1:
      self.dtype = torch.cuda.LongTensor
    else:
      self.dtype = torch.LongTensor

  def forward(self, input, target):
    logprob = torch.sum(torch.log(input[torch.arange(0, len(target)).type(self.dtype), target.data]))
    return -logprob / len(target)

