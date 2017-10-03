import torch
from torch import nn
from torch.autograd import Variable

import pdb

class StackLSTM(nn.Module):

  def __init__(self, input_size, hidden_size, dropout_rate, stack_size):
    super(StackLSTM, self).__init__()
    self.hidden_size = hidden_size
    self.stack_size = stack_size

    self.lstm = nn.LSTMCell(input_size, hidden_size)
    self.nl = nn.Sigmoid()
    self.tan = nn.Tanh()

    self.initial_hidden = self.initHidden()
    self.initial_cell = self.initCell()

  def forward(self, inputs, ops,
              preload_hidden = None,
              preload_cell = None,
              preload_pos = 0):
    """
    All inputs except for preload_pos should be torch tensors.

    :param inputs: (seq_len, batch_size, input_size), input vector, in batch.
    :param ops: (seq_len, batch_size), stack operations, in batch (-1 means pop, 1 means push, 0 means hold).
    :param preload_hidden: (self.stack_size, batch_size, self.hidden_size), hidden states that is preloaded before
           timestep 0.
    :param preload_cell: (self.stack_size + 1, batch_size, self.hidden_size), cell states that is preloaded before
           timestep 0.
    :param preload_pos: scalar, the pointer position that needs to initialized if preloading is used.
    :return: hiddens: (seq_len, batch_size, hidden_size)
    """
    # assert(ops < self.stack_size)
    pts = preload_pos + torch.cumsum(ops, dim=0)
    ops = ops[1:, :]

    hiddens = []

    batch_size = len(inputs[0]) # dynamic! viva pytorch!

    # the hidden and cell state stack maintained for future computation
    # note that they are different from the memory bank that's returned
    # by the forward function!
    hidden_stack = Variable(torch.zeros(self.stack_size + 1, batch_size, self.hidden_size))
    cell_stack = Variable(torch.zeros(self.stack_size + 1, batch_size, self.hidden_size))

    if preload_hidden is not None:
      hidden_stack[1:, :, :] = preload_hidden

    if preload_cell is not None:
      cell_stack[1:, :, :] = preload_cell

    # push start states to the stack
    hidden_stack[0, :, :] = self.initial_hidden.expand((batch_size, self.hidden_size))
    cell_stack[0, :, :] = self.initial_cell.expand((batch_size, self.hidden_size))

    for (ip, pt, op) in zip(inputs, pts, ops):

      cur_hidden = hidden_stack[pt.data.tolist(), range(len(pt)), :].clone()
      cur_cell = cell_stack[pt.data.tolist(), range(len(pt)), :].clone()
      prev_hidden = hidden_stack[((pt - 1) % (self.stack_size + 1)).data.tolist(), range(len(pt)), :].clone()

      next_hidden, next_cell = self.lstm(ip, (cur_hidden, cur_cell))

      hidden_stack[(pt + 1).data.tolist(), range(len(pt)), :] = next_hidden
      cell_stack[(pt + 1).data.tolist(), range(len(pt)), :] = next_cell.clone()

      # all return decision are here
      # push/pop
      ret = torch.mm(next_hidden.t(), torch.diag(1 + op).float()) + \
            torch.mm(prev_hidden.t(), torch.diag(1 + (-1) * op).float())
      # move/hold
      ret = torch.mm(ret, torch.diag(torch.abs(op)).float()) + \
            torch.mm(cur_hidden.t(), torch.diag(1 + (-1) * torch.abs(op)).float())
      ret = ret.t()

      # update memory bank
      hiddens.append(ret)

    return hiddens

  def initHidden(self):
    return Variable(torch.rand((self.hidden_size,)))

  def initCell(self):
    return Variable(torch.rand((self.hidden_size,)))

class BatchedToyStackModel(nn.Module):

  def __init__(self, emb_dim, hid_dim, stack_size, vocab_size):
    super(BatchedToyStackModel, self).__init__()

    self.emb = nn.Embedding(vocab_size, emb_dim) # 50 dimension embedding of 0 and 1
    self.srnn = StackLSTM(emb_dim, hid_dim, 0, stack_size)
    self.final = nn.Linear(hid_dim, vocab_size)
    self.sfmax = nn.LogSoftmax()

  # inputs: (seq_len, batch_size)
  # ops: (seq_len, bath_size)
  def forward(self, inputs, ops):
    seq_len = inputs.size()[0]
    batch_size = inputs.size()[1]

    word_emb = self.emb(inputs) # (seq_len, batch_size, self.input_size)
    hidden_seq = self.srnn(word_emb, ops) # (batch_size, self.hidden_size)
    hiddens = torch.stack(hidden_seq, dim = 1) # (batch_size, seq_len, self.hidden_size)
    hiddens = hiddens.view(batch_size * seq_len, -1) # (batch_size * seq_len, self.hidden_size)
    output = self.sfmax(self.final(hiddens))
    output = output.view(seq_len * batch_size, -1)
    return output

def gen_data(vocab_size, seq_n, seq_len, batch_size = None):
  X = torch.round((vocab_size - 1) * torch.rand(seq_n, seq_len)).long()
  P = torch.Tensor([[0] * (seq_len + 1)] * seq_n).long()
  Y = torch.Tensor([[0] * seq_len] * seq_n).long()
  for i in range(seq_n):
    S = []
    for j in range(seq_len):
      dice = torch.rand(1)
      # push
      if dice[0] < 0.3 or len(S) <= 1:
        P[i, j + 1] = 1
        S.append(X[i, j])
      # pop
      elif dice[0] < 0.7:
        P[i, j + 1] = -1
        S.pop()
      # hold
      else:
        P[i, j + 1] = 0
      Y[i, j] = S[-1]
  if batch_size:
    (X, P, Y) = batchize_data(X, P, Y, batch_size)

  X = Variable(X)
  P = Variable(P)
  Y = Variable(Y)
  return X, P, Y

def batchize_data(X, P, Y, batch_size):
  X_batched = []
  P_batched = []
  Y_batched = []

  for start in range(0, len(X), batch_size):
    X_batch = X[start: start + batch_size, :].t() # (seq_len, batch_size)
    P_batch = P[start: start + batch_size, :].t() # (seq_len, batch_size)
    Y_batch = Y[start: start + batch_size, :].t() # (seq_len, batch_size)

    X_batched.append(X_batch.tolist())
    P_batched.append(P_batch.tolist())
    Y_batched.append(Y_batch.tolist())

  X_batched = torch.Tensor(X_batched).long()
  P_batched = torch.Tensor(P_batched).long()
  Y_batched = torch.Tensor(Y_batched).long()

  return X_batched, P_batched, Y_batched

if __name__ == "__main__":
  SEQN = 100
  SEQLEN = 50
  EMB_DIM = 50
  HID_DIM = 150
  STACK_SIZE = 30
  VOCAB_SIZE = 250
  BATCH_SIZE = 5

  MAX_EPOCH = 10

  # reproducibility
  torch.manual_seed(1)

  # create random push/pop training data
  X, P, Y = gen_data(VOCAB_SIZE, SEQN, SEQLEN, BATCH_SIZE)
  Y = Y.permute(0, 2, 1).contiguous() # make sure the flattened golden labels are correct

  # create test
  SEQN_t = 10
  SEQLEN_t = 30
  assert(SEQLEN_t <= SEQLEN)

  X_t, P_t, Y_t = gen_data(VOCAB_SIZE, SEQN_t, SEQLEN_t, 2)
  Y_t = Y_t.permute(0, 2, 1).contiguous() # make sure the flattened golden labels are correct

  model = BatchedToyStackModel(EMB_DIM, HID_DIM, STACK_SIZE, VOCAB_SIZE)
  criterion = nn.NLLLoss()
  optimizer = torch.optim.Adadelta(model.parameters())

  for n in range(MAX_EPOCH):
    print("epoch {0}".format(n))
    avg_loss = 0
    for i in range(len(X)):
      if i % 100 == 0:
        print(".", end="", flush=True)

      output = model(X[i], P[i])
      loss = criterion(output, Y[i].view(BATCH_SIZE * SEQLEN,))
      avg_loss += loss

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    print("avg loss: {0}".format(avg_loss.data[0] / SEQN))

    correct = 0
    correct_push = 0
    correct_hold = 0
    correct_pop = 0
    total = SEQN_t * SEQLEN_t
    total_push = torch.sum(P_t[:, 1:] == 1).data[0]
    total_hold = torch.sum(P_t[:, 1:] == 0).data[0]
    total_pop = torch.sum(P_t[:, 1:] == -1).data[0]
    for i in range(len(X_t)):
      output = model(X_t[i], P_t[i])
      _, argmax = torch.max(output, 1)
      gold = Y_t[i].view(-1)
      op = P_t[:, 1:][i].view(-1)
      correct += torch.sum(gold == argmax).data[0]
      correct_push += torch.sum((op == 1).data & (gold == argmax).data)
      correct_hold += torch.sum((op == 0).data & (gold == argmax).data)
      correct_pop += torch.sum((op == -1).data & (gold == argmax).data)

    print("prediction accuracy: {0} / {1} = {2}\namong which:".format(correct, total, correct / total))
    print("correct push: {0} / {1} = {2}".format(correct_push, total_push, correct_push / total_push))
    print("correct hold: {0} / {1} = {2}".format(correct_hold, total_hold, correct_hold / total_hold))
    print("correct pop: {0} / {1} = {2}".format(correct_pop, total_pop, correct_pop / total_pop))
