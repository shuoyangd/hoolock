import torch
from torch import nn
from torch.autograd import Variable

import pdb

class StackLSTM(nn.Module):

  def __init__(self, input_size, hidden_size, dropout_rate, stack_size):
    super(StackLSTM, self).__init__()
    # self.input_size = input_size
    self.hidden_size = hidden_size
    # self.dropout = nn.Dropout(dropout_rate)
    self.stack_size = stack_size

    self.x2i = nn.Linear(input_size, hidden_size, True)
    self.h2i = nn.Linear(hidden_size, hidden_size, False)
    self.c2i = nn.Linear(hidden_size, hidden_size, False)

    self.x2f = nn.Linear(input_size, hidden_size, True)
    self.h2f = nn.Linear(hidden_size, hidden_size, False)
    self.c2f = nn.Linear(hidden_size, hidden_size, False)

    self.x2c = nn.Linear(input_size, hidden_size, True)
    self.h2c = nn.Linear(hidden_size, hidden_size, False)

    self.x2o = nn.Linear(input_size, hidden_size, True)
    self.h2o = nn.Linear(hidden_size, hidden_size, False)
    self.c2o = nn.Linear(hidden_size, hidden_size, False)

    self.nl = nn.Sigmoid()
    self.tan = nn.Tanh()

  # inputs: (seq_len, input_size)
  # ops: (seq_len, )
  # masks: (seq_len, ) TODO reserved for batch
  # hiddens: (seq_len, hidden_size)
  # cells: (seq_len, hidden_size)
  #
  def forward(self, inputs, ops):

    # assert(ops < self.stack_size)
    pts = torch.cumsum(ops, dim=0)
    bi_ops = torch.clamp(ops[1:], 0, 1)

    hiddens = []

    # the hidden and cell state stack maintained for future computation
    # note that they are different from the memory bank that's returned
    # by the forward function!
    hidden_stack = Variable(torch.zeros(self.stack_size + 1, self.hidden_size))
    cell_stack = Variable(torch.zeros(self.stack_size + 1, self.hidden_size))

    # push start states to the stack
    hidden_stack[0] = self.initHidden()
    cell_stack[0] = self.initCell()

    for (ip, pt, op) in zip(inputs, pts, bi_ops):

      # remove pytorch madness
      ip = ip.view(1, -1)
      pt = pt.data
      op = op.data[0]

      cur_hidden = hidden_stack[pt].clone()
      cur_cell = cell_stack[pt].clone()

      ig = self.nl(self.x2i(ip) + self.h2i(cur_hidden) + self.c2i(cur_cell))
      fg = self.nl(self.x2f(ip) + self.h2f(cur_hidden) + self.c2f(cur_cell))
      cg = fg * cur_cell + ig * self.tan(self.x2c(ip) + self.h2f(cur_hidden))
      og = self.nl(self.x2o(ip) + self.h2o(cur_hidden) + self.c2o(cur_cell))

      next_hidden = og * self.tan(cg)
      prev_hidden = hidden_stack[pt[0] - 1].clone()

      # update stack content
      hidden_stack[pt + 1] = next_hidden
      cell_stack[pt + 1] = cg

      # decide which to return
      ret = op * next_hidden + (1 - op) * prev_hidden

      # update memory bank
      hiddens.append(ret[0]) # want a vector

    return hiddens

  def initHidden(self):
    return Variable(torch.rand((self.hidden_size,)))

  def initCell(self):
    return Variable(torch.rand((self.hidden_size,)))

class ToyStackModel(nn.Module):

  def __init__(self, emb_dim, hid_dim, stack_size, vocab_size):
    super(ToyStackModel, self).__init__()

    self.emb = nn.Embedding(vocab_size, emb_dim) # 50 dimension embedding of 0 and 1
    self.srnn = StackLSTM(emb_dim, hid_dim, 0, stack_size)
    self.final = nn.Linear(hid_dim, vocab_size)
    self.sfmax = nn.LogSoftmax()

  # input: (seq_len, )
  # pointer: (seq_len, )
  def forward(self, inputs, ops):
    word_emb = self.emb(inputs)
    hidden_seq = self.srnn(word_emb, ops)
    hiddens = torch.stack(hidden_seq, dim = 1)
    hiddens = torch.t(hiddens)
    output = self.sfmax(self.final(hiddens))
    return output

def gen_data(vocab_size, seq_n, seq_len):
  X = torch.round((vocab_size - 1) * torch.rand(seq_n, seq_len)).long()
  P = torch.Tensor([[0] * (seq_len + 1)] * seq_n).long()
  Y = torch.Tensor([[0] * seq_len] * seq_n).long()
  for i in range(seq_n):
    S = []
    for j in range(seq_len):
      dice = torch.rand(1)
      # push
      if dice[0] < 0.5 or len(S) <= 1:
        P[i, j + 1] = 1
        S.append(X[i, j])
      # pop
      else:
        P[i, j + 1] = -1
        S.pop()
      Y[i, j] = S[-1]
  X = Variable(X)
  P = Variable(P)
  Y = Variable(Y)
  return X, P, Y

if __name__ == "__main__":
  SEQN = 100
  SEQLEN = 50
  EMB_DIM = 50
  HID_DIM = 150
  STACK_SIZE = 30
  VOCAB_SIZE = 500

  MAX_EPOCH = 10

  # create random push/pop training data
  X, P, Y = gen_data(VOCAB_SIZE, SEQN, SEQLEN)

  # create test
  SEQN_t = 10
  SEQLEN_t = 30

  X_t, P_t, Y_t = gen_data(VOCAB_SIZE, SEQN_t, SEQLEN_t)

  model = ToyStackModel(EMB_DIM, HID_DIM, STACK_SIZE, VOCAB_SIZE)
  criterion = nn.NLLLoss()
  optimizer = torch.optim.Adadelta(model.parameters())

  for n in range(MAX_EPOCH):
    print("epoch {0}".format(n))
    avg_loss = 0
    for i in range(SEQN):
      if i % 100 == 0:
        print(".", end="", flush=True)

      output = model(X[i], P[i])
      loss = criterion(output, Y[i])
      avg_loss += loss

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    print("avg loss: {0}".format(avg_loss.data[0] / SEQN))

    correct = 0
    correct0 = 0
    correct1 = 0
    total = SEQN_t * SEQLEN_t
    P_tp = P_t[:, 1:]
    total0 = torch.sum(P_tp == 1).data[0]
    total1 = torch.sum(P_tp == -1).data[0]
    for i in range(SEQN_t):
      output = model(X_t[i], P_t[i])
      _, argmax = torch.max(output, 1)
      correct += torch.sum(Y_t[i] == argmax).data[0]
      correct0 += torch.sum((P_tp[i] == 1).data & (Y_t[i] == argmax).data)
      correct1 += torch.sum((P_tp[i] == -1).data & (Y_t[i] == argmax).data)

    print("prediction accuracy: {0} / {1} = {2}\namong which:".format(correct, total, correct / total))
    print("correct push: {0} / {1} = {2}".format(correct0, total0, correct0 / total0))
    print("correct pop: {0} / {1} = {2}".format(correct1, total1, correct1 / total1))
