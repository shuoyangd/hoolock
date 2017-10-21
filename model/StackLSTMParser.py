from enum import Enum
import logging

import torch
from torch import nn
from torch.autograd import Variable
from model.StackLSTMCell import StackLSTMCell
from model.BufferLSTMCell import BufferLSTMCell

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

class TransitionSystems(Enum):
  ASd = 0
  AER = 1
  AES = 2
  AH = 3
  ASw = 4

  @classmethod
  def has_value(cls, value):
    return any(value == item for item in cls)

AER_map = {"Shift": (1, -1), "Reduce": (-1, 0), "Left-Arc": (-1, 0), "Right-Arc": (1, -1)}
AH_map = {"Shift": (1, -1), "Left-Arc": (-1, 0), "Right-Arc": (1, 0)}

class StackLSTMParser(nn.Module):

  def __init__(self, vocab, actions, options, pre_vocab=None, postags=None):
    """
    :param vocab: vocabulary, with type torchtext.vocab.Vocab
    :param actions: a python list of actions, denoted as "transition" (unlabled) or "transition|label" (labeled).
           e.g. "Left-Arc", "Left-Arc|nsubj"
    :param pre_vocab: vocabulary with pre-trained word embedding, with type torchtext.vocab.Vocab.
           This vocabulary should contain the word embedding itself as well, see torchtext documents for details.
    :param postags: a python list of pos, in strings (optional)
    """
    super(StackLSTMParser, self).__init__()
    self.stack_size = options.stack_size
    self.hid_dim = options.hid_dim
    self.max_step_length = options.max_step_length
    self.transSys = options.transSys
    self.exposure_eps = options.exposure_eps

    # gpu
    self.gpuid = options.gpuid
    self.dtype = torch.cuda.FloatTensor if len(options.gpuid) >= 1 else torch.FloatTensor
    self.long_dtype = torch.cuda.LongTensor if len(options.gpuid) >= 1 else torch.LongTensor

    # vocabularies
    self.vocab = vocab
    self.pre_vocab = pre_vocab
    self.actions = actions
    self.postags = postags

    # action mappings
    self.stack_action_mapping = Variable(torch.zeros(len(actions),).type(self.long_dtype))
    self.buffer_action_mapping = Variable(torch.zeros(len(actions),).type(self.long_dtype))
    self.set_action_mappings()

    # embeddings
    self.word_emb = nn.Embedding(len(vocab), options.word_emb_dim)
    self.action_emb = nn.Embedding(len(actions), options.action_emb_dim)

    if pre_vocab is not None:
      self.pre_word_emb = nn.Embedding(pre_vocab.vectors.size(0), pre_vocab.dim)
      # initialzie and fixed
      self.pre_word_emb.weight.data = pre_vocab.vectors
      self.pre_word_emb.weight.requires_grad = False
    else:
      self.pre_word_emb = None

    if postags is not None:
      self.postag_emb = nn.Embedding(len(postags), options.postag_emb_dim)
    else:
      self.postag_emb = None

    # token embdding composition
    token_comp_in_features = options.word_emb_dim # vocab only
    if pre_vocab is not None:
      # token_comp_in_features += options.pre_word_emb_dim
      token_comp_in_features += pre_vocab.dim
    if postags is not None:
      token_comp_in_features += options.postag_emb_dim
    self.compose_tokens = nn.Sequential(
        nn.Linear(token_comp_in_features, options.input_dim),
        nn.ReLU()
    )

    # recurrent components
    # the only reason to have unified h0 and c0 is because pre_buffer and buffer should have the same initial states
    # but due to simplicity we just borrowed this for all three recurrent components (stack, buffer, action)
    self.h0 = nn.Parameter(torch.rand(options.hid_dim,).type(self.dtype))
    self.c0 = nn.Parameter(torch.rand(options.hid_dim,).type(self.dtype))
    # FIXME: there is no dropout in StackLSTMCell at this moment
    # BufferLSTM could have 0 or 2 parameters, depending on what is passed for initial hidden and cell state
    self.stack = StackLSTMCell(options.input_dim, options.hid_dim, options.dropout_rate, options.stack_size, self.h0, self.c0)
    self.buffer = BufferLSTMCell(options.hid_dim, self.h0, self.c0)
    self.token_buffer = BufferLSTMCell(options.input_dim) # elememtns in this buffer has size input_dim so h0 and c0 won't fit
    self.pre_buffer = nn.LSTMCell(input_size=options.input_dim, hidden_size=options.hid_dim) # FIXME: dropout needs to be implemented manually
    self.history = nn.LSTMCell(input_size=options.action_emb_dim, hidden_size=options.hid_dim) # FIXME: dropout needs to be implemented manually

    # parser state and softmax
    self.summarize_states = nn.Sequential(
        nn.Linear(3 * options.hid_dim, options.state_dim),
        nn.ReLU()
    )
    self.state_to_actions = nn.Linear(options.state_dim, len(actions))

    self.softmax = nn.LogSoftmax() # LogSoftmax works on final dimension only for two dimension tensors. Be careful.

  def cpu(self):
    super(StackLSTMParser, self).cpu()
    self.dtype = torch.FloatTensor
    self.long_dtype = torch.LongTensor

  def gpu(self):
    super(StackLSTMParser, self).gpu()
    self.dtype = torch.cuda.FloatTensor
    self.long_dtype = torch.cuda.LongTensor

  def forward(self, tokens, pre_tokens=None, postags=None, actions=None):
    """

    :param tokens: (seq_len, batch_size) tokens of the input sentence, preprocessed as vocabulary indexes
    :param postags: (seq_len, batch_size) optional POS-tag of the tokens of the input sentence, preprocessed as indexes
    :param actions: (seq_len, batch_size) optional golden action sequence, preprocessed as indexes
    :return: output: (output_seq_len, batch_size, len(actions)), or when actions = None, (output_seq_len, batch_size, max_step_length)
    """

    seq_len = tokens.size()[0]
    batch_size = tokens.size()[1]
    word_emb = self.word_emb(tokens.t()) # (batch_size, seq_len, word_emb_dim)
    token_comp_input = word_emb
    if self.pre_word_emb is not None:
      pre_word_emb = self.pre_word_emb(pre_tokens.t()) # (batch_size, seq_len, self.pre_vocab.dim)
      token_comp_input = torch.cat((token_comp_input, pre_word_emb), dim=-1)
    if self.postag_emb is not None and postags is not None:
      pos_tag_emb = self.postag_emb(postags.t()) # (batch_size, seq_len, word_emb_dim)
      token_comp_input = torch.cat((token_comp_input, pos_tag_emb), dim=-1)
    token_comp_output = self.compose_tokens(token_comp_input) # (batch_size, seq_len, input_dim)
    token_comp_output = torch.transpose(token_comp_output, 0, 1) # (seq_len, batch_size, input_dim)
    rev_idx = Variable(torch.arange(seq_len - 1, -1, -1).type(self.long_dtype))
    token_comp_output_rev = token_comp_output.index_select(0, rev_idx)

    # initialize stack
    self.stack.build_stack(batch_size, self.gpuid)

    # initialize and preload buffer
    buffer_hiddens = Variable(torch.zeros((seq_len, batch_size, self.hid_dim)).type(self.dtype))
    buffer_cells = Variable(torch.zeros((seq_len, batch_size, self.hid_dim)).type(self.dtype))
    bh = self.h0.unsqueeze(0).expand(batch_size, self.hid_dim)
    bc = self.c0.unsqueeze(0).expand(batch_size, self.hid_dim)
    for t_i in range(seq_len):
      bh, bc = self.pre_buffer(token_comp_output_rev[t_i], (bh, bc)) # (batch_size, self.hid_dim)
      buffer_hiddens[t_i, :, :] = bh
      buffer_cells[t_i, :, :] = bc
    # buffer_hidden and buffer_cell need to be reversed because early words needs to be popped first
    # inv_idx = Variable(torch.arange(buffer_hiddens.size(0) - 1, -1, -1).type(self.long_dtype)) # ugly reverse on dim0
    # buffer_hiddens = Variable(buffer_hiddens.data.index_select(0, inv_idx.data))
    # buffer_cells = Variable(buffer_cells.data.index_select(0, inv_idx.data))
    self.buffer.build_stack(buffer_hiddens, buffer_cells, self.gpuid)
    self.token_buffer.build_stack(token_comp_output, Variable(torch.zeros(token_comp_output.size())), self.gpuid) # don't need cell for this

    stack_state, _ = self.stack.head() # (batch_size, hid_dim)
    buffer_state = self.buffer.head() # (batch_size, hid_dim)
    stack_input = self.token_buffer.head() # (batch_size, input_size)
    action_state = self.h0.unsqueeze(0).expand(batch_size, self.hid_dim) # (batch_size, hid_dim)
    action_cell = self.c0.unsqueeze(0).expand(batch_size, self.hid_dim) # (batch_size, hid_dim)

    # prepare bernoulli probability for exposure indicator
    batch_exposure_prob = Variable(torch.Tensor([self.exposure_eps] * batch_size).type(self.dtype))

    # main loop
    if actions is None:
      outputs = Variable(torch.zeros((self.max_step_length, batch_size, len(self.actions))).type(self.dtype))
      step_length = self.max_step_length
    else:
      outputs = Variable(torch.zeros((actions.size()[0], batch_size, len(self.actions))).type(self.dtype))
      step_length = actions.size()[0]
    for step_i in range(step_length):
      # get action decisions
      summary = self.summarize_states(torch.cat((stack_state, buffer_state, action_state), dim=1)) # (batch_size, self.state_dim)
      action_dist = self.softmax(self.state_to_actions(summary)) # (batch_size, len(actions))
      outputs[step_i, :, :] = action_dist.clone()
      _, action_i = torch.max(action_dist, dim=1) # (batch_size,)
      # control exposure (only for training)
      if actions is not None:
        oracle_action_i = actions[step_i, :] # (batch_size,)
        batch_exposure = torch.bernoulli(batch_exposure_prob) # (batch_size,)
        action_i = Variable((action_i.data.float() * (1 - batch_exposure.data) + oracle_action_i.data.float() * batch_exposure.data).long()) # (batch_size,)
      # translate action into stack & buffer ops
      # stack_op, buffer_op = self.map_action(action_i.data)
      stack_op = self.stack_action_mapping.index_select(0, action_i)
      buffer_op = self.buffer_action_mapping.index_select(0, action_i)

      # check boundary conditions, assuming buffer won't be pushed anymore
      stack_pop_boundary = Variable((self.stack.pos == 0).data & (stack_op == -1).data)
      stack_push_boundary = Variable((self.stack.pos >= self.stack_size).data & (stack_op == 1).data)
      buffer_pop_boundary = Variable((self.buffer.pos ==0).data & (buffer_op == -1).data)

      # if stack/buffer is at boundary, hold at current position rather than push/pop
      stack_op = stack_op.masked_fill(stack_pop_boundary, 0)
      stack_op = stack_op.masked_fill(stack_push_boundary, 0)
      buffer_op = buffer_op.masked_fill(buffer_pop_boundary, 0)

      # update stack, buffer and action state
      stack_state, _ = self.stack(stack_input, stack_op)
      buffer_state, _ = self.buffer(buffer_op)
      stack_input, _ = self.token_buffer(buffer_op)
      action_input = self.action_emb(action_i) # (batch_size, hid_dim)
      action_state, action_cell = self.history(action_input, (action_state, action_cell))

    return outputs

  def set_action_mappings(self):
    for idx, astr in enumerate(self.actions):
      transition = astr.split('|')[0]
      if self.transSys == TransitionSystems.AER:
        self.stack_action_mapping[idx], self.buffer_action_mapping[idx] = AER_map.get(transition, (0, 0))
      elif self.transSys == TransitionSystems.AH:
        self.stack_action_mapping[idx], self.buffer_action_mapping[idx] = AH_map.get(transition, (0, 0))
      else:
        logging.fatal("Unimplemented transition system.")
        raise NotImplementedError
