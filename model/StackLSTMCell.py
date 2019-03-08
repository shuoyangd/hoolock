from model.MultiLayerLSTMCell import MultiLayerLSTMCell
import torch
from torch import nn
# from torch.autograd import Variable

class StackLSTMCell(nn.Module):

  def __init__(self, input_size, hidden_size, dropout_rate, stack_size, num_layers=1,
               init_hidden_var=None, init_cell_var=None):
    """

    :param input_size:
    :param hidden_size:
    :param dropout_rate:
    :param stack_size:
    """
    super(StackLSTMCell, self).__init__()
    self.hidden_size = hidden_size
    self.stack_size = stack_size
    self.num_layers = num_layers

    # self.lstm = nn.LSTMCell(input_size, hidden_size)
    self.lstm = MultiLayerLSTMCell(input_size, hidden_size, num_layers)

    self.initial_hidden = self.init_hidden(init_hidden_var)
    self.initial_cell = self.init_cell(init_cell_var)

  def build_stack(self, batch_size, gpuid=[]):
    """
    Provide a way to initialize stack with minimal overhead

    :param batch_size: needed to initialize stack, cannot be dynamic
    """
    # the hidden and cell state stack maintained for future computation
    # note that they are different from the memory bank that's returned
    # by the forward function!
    if len(gpuid) >= 1:
      self.dtype = torch.cuda.FloatTensor
      self.long_dtype = torch.cuda.LongTensor
    else:
      self.dtype = torch.FloatTensor
      self.long_dtype = torch.LongTensor

    self.pos = torch.LongTensor([0] * batch_size).type(self.long_dtype)

    self.hidden_stack = torch.zeros(self.stack_size + 1, batch_size, self.hidden_size, self.num_layers).type(self.dtype)
    self.cell_stack = torch.zeros(self.stack_size + 1, batch_size, self.hidden_size, self.num_layers).type(self.dtype)

    # seq_len = preload_hidden.size()[0]
    # if preload_hidden is not None:
    #   self.hidden_stack[1:seq_len+1, :, :] = preload_hidden

    # if preload_cell is not None:
    #   self.cell_stack[1:seq_len+1, :, :] = preload_cell

    # push start states to the stack
    self.hidden_stack[0, :, :, :] = self.initial_hidden.expand((batch_size, self.num_layers, self.hidden_size)).transpose(2, 1)
    self.cell_stack[0, :, :, :] = self.initial_cell.expand((batch_size, self.num_layers, self.hidden_size)).transpose(2, 1)

  def forward(self, input, op):
    """
    stack needs to be built before this is called.

    :param input: (batch_size, input_size), input vector, in batch.
    :param op: (batch_size,), stack operations, in batch (-1 means pop, 1 means push, 0 means hold).
    :return: (hidden, cell): both are (batch_size, hidden_dim)
    """

    batch_size = input.size(0)
    batch_indexes = torch.arange(0, batch_size).type(self.long_dtype)
    pos = self.pos.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(1, batch_size, self.hidden_size, self.num_layers)
    cur_hidden = torch.gather(self.hidden_stack, 0, pos)
    cur_cell = torch.gather(self.cell_stack, 0, pos)
    # cur_hidden, cur_cell = self.hidden_stack[self.pos, batch_indexes, :, :], \
    #                        self.cell_stack[self.pos, batch_indexes, :, :]
    next_hidden, next_cell = self.lstm(input, (cur_hidden, cur_cell))
    self.hidden_stack[self.pos + 1, batch_indexes, :, :] = next_hidden.clone()
    self.cell_stack[self.pos + 1, batch_indexes, :, :] = next_cell.clone()
    self.pos = self.pos + op  # XXX: should NOT use in-place assignment!
    
    # hidden_ret = self.hidden_stack[self.pos, batch_indexes, :, :]
    # cell_ret = self.cell_stack[self.pos, batch_indexes, :, :]
    pos = self.pos.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(1, batch_size, self.hidden_size, self.num_layers)
    hidden_ret = torch.gather(self.hidden_stack, 0, pos)
    cell_ret = torch.gather(self.cell_stack, 0, pos)
    return hidden_ret[:, :, -1], cell_ret[:, :, -1]

  def init_hidden(self, init_var=None):
    if init_var is None:
      return torch.rand((self.hidden_size,))
    else:
      return init_var

  def init_cell(self, init_var=None):
    if init_var is None:
      return torch.rand((self.hidden_size,))
    else:
      return init_var

  def head(self):
    return self.hidden_stack[self.pos, torch.arange(0, len(self.pos)).type(self.long_dtype), :][:, :, -1] ,\
           self.cell_stack[self.pos, torch.arange(0, len(self.pos)).type(self.long_dtype), :][:, :, -1]

  def size(self):
    return torch.max(self.pos + 1)[0].item()

class BatchedToyStackModel(nn.Module):

  def __init__(self, emb_dim, hid_dim, stack_size, vocab_size):
    super(BatchedToyStackModel, self).__init__()

    self.emb = nn.Embedding(vocab_size, emb_dim) # 50 dimension embedding of 0 and 1
    self.srnn = StackLSTMCell(emb_dim, hid_dim, 0, stack_size, 2)
    self.final = nn.Linear(hid_dim, vocab_size)
    self.sfmax = nn.LogSoftmax()

  # inputs: (seq_len, batch_size)
  # ops: (seq_len, bath_size)
  def forward(self, inputs, ops):
    seq_len = inputs.size()[0]
    batch_size = inputs.size()[1]

    word_emb = self.emb(inputs) # (seq_len, batch_size, self.input_size)

    self.srnn.build_stack(len(inputs[0]))
    hidden_seq = []
    for (ip, op) in zip(word_emb, ops[1:]):
      hidden, cell = self.srnn(ip, op) # (batch_size, self.hidden_size)
      hidden_seq.append(hidden)
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

  # X = Variable(X)
  # P = Variable(P)
  # Y = Variable(Y)
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

  MAX_EPOCH = 15

  # reproducibility
  torch.manual_seed(1)

  # create random push/pop training data
  X, P, Y = gen_data(VOCAB_SIZE, SEQN, SEQLEN, BATCH_SIZE)
  Y = Y.permute(0, 2, 1).contiguous() # make sure the flattened golden labels are correct

  # create test
  SEQN_t = 10
  SEQLEN_t = 30
  assert(SEQLEN_t <= SEQLEN)

  X_t, P_t, Y_t = gen_data(VOCAB_SIZE, SEQN_t, SEQLEN_t, 1)
  Y_t = Y_t.permute(0, 2, 1).contiguous() # make sure the flattened golden labels are correct

  model = BatchedToyStackModel(EMB_DIM, HID_DIM, STACK_SIZE, VOCAB_SIZE)
  criterion = nn.NLLLoss()
  # optimizer = torch.optim.Adadelta(model.parameters())
  optimizer = torch.optim.Adam(model.parameters())

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

    print("avg loss: {0}".format(avg_loss.item() / SEQN))

    correct = 0
    correct_push = 0
    correct_hold = 0
    correct_pop = 0
    total = SEQN_t * SEQLEN_t
    total_push = torch.sum(P_t[:, 1:] == 1).item()
    total_hold = torch.sum(P_t[:, 1:] == 0).item()
    total_pop = torch.sum(P_t[:, 1:] == -1).item()
    for i in range(len(X_t)):
      output = model(X_t[i], P_t[i])
      _, argmax = torch.max(output, 1)
      gold = Y_t[i].view(-1)
      op = P_t[:, 1:][i].view(-1)
      correct += torch.sum(gold == argmax).item()
      correct_push += torch.sum((op == 1) & (gold == argmax)).item()
      correct_hold += torch.sum((op == 0) & (gold == argmax)).item()
      correct_pop += torch.sum((op == -1) & (gold == argmax)).item()

    print("prediction accuracy: {0} / {1} = {2}\namong which:".format(correct, total, correct / total))
    print("correct push: {0} / {1} = {2}".format(correct_push, total_push, 0 if total_push == 0 else correct_push / total_push))
    print("correct hold: {0} / {1} = {2}".format(correct_hold, total_hold, 0 if total_hold == 0 else correct_hold / total_hold))
    print("correct pop: {0} / {1} = {2}".format(correct_pop, total_pop, 0 if total_pop == 0 else correct_pop / total_pop))
