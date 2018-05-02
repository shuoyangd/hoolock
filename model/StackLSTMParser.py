from enum import Enum
import logging
import pdb
import utils

import torch
from torch import nn
from torch.autograd import Variable
from model.StackLSTMCell import StackLSTMCell
from model.StateStack import StateStack
from model.MultiLayerLSTMCell import MultiLayerLSTMCell

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
AH_map = {"Shift": (1, -1), "Left-Arc": (-1, 0), "Right-Arc": (-1, 0)}


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
    self.input_dim = options.input_dim
    self.max_step_length = options.max_step_length
    self.transSys = TransitionSystems(options.transSys)
    self.exposure_eps = options.exposure_eps
    self.num_lstm_layers = options.num_lstm_layers

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
    self.root_input = nn.Parameter(torch.rand(options.input_dim))

    # recurrent components
    # the only reason to have unified h0 and c0 is because pre_buffer and buffer should have the same initial states
    # but due to simplicity we just borrowed this for all three recurrent components (stack, buffer, action)
    self.h0 = nn.Parameter(torch.rand(options.hid_dim,).type(self.dtype))
    self.c0 = nn.Parameter(torch.rand(options.hid_dim,).type(self.dtype))
    # FIXME: there is no dropout in StackLSTMCell at this moment
    # BufferLSTM could have 0 or 2 parameters, depending on what is passed for initial hidden and cell state
    self.stack = StackLSTMCell(options.input_dim, options.hid_dim, options.dropout_rate, options.stack_size, options.num_lstm_layers, self.h0, self.c0)
    self.buffer = StateStack(options.hid_dim, self.h0)
    self.token_stack = StateStack(options.input_dim)
    self.token_buffer = StateStack(options.input_dim) # elememtns in this buffer has size input_dim so h0 and c0 won't fit
    self.pre_buffer = MultiLayerLSTMCell(input_size=options.input_dim, hidden_size=options.hid_dim, num_layers=options.num_lstm_layers) # FIXME: dropout needs to be implemented manually
    self.history = nn.LSTMCell(input_size=options.action_emb_dim, hidden_size=options.hid_dim) # FIXME: dropout needs to be implemented manually

    # parser state and softmax
    self.use_token_highway = options.use_token_highway
    if options.use_token_highway:
      self.summarize_states = nn.Sequential(
          nn.Linear(3 * options.hid_dim + 2 * options.input_dim, options.state_dim),
          nn.ReLU()
      )
    else:
      self.summarize_states = nn.Sequential(
          nn.Linear(3 * options.hid_dim, options.state_dim),
          nn.ReLU()
      )
    self.state_to_actions = nn.Linear(options.state_dim, len(actions))

    # composition
    self.composition_k = options.composition_k
    if self.composition_k > 0:
      self.W_h = nn.Bilinear(options.action_emb_dim, self.input_dim, 1, bias=False)
      self.W_d = nn.Bilinear(options.action_emb_dim, self.input_dim, 1, bias=False)
      self.W_b = nn.Bilinear(options.action_emb_dim, self.input_dim, 1, bias=False)
      self.composition_func = nn.Sequential(
          nn.Linear(self.input_dim * 2 + options.rel_emb_dim, self.input_dim),
          nn.Tanh()
      )

      self.relations = [ "NONE" ]
      self.a2r = torch.LongTensor(len(self.actions)).type(self.long_dtype)
      for a_id, action in enumerate(actions):
        if '|' in action:
          rel = action.split('|')[1]
          if rel not in self.relations:
           rel_id = len(self.relations)
           self.relations.append(rel)
          else:
           rel_id = self.relations.index(rel)
          self.a2r[a_id] = rel_id
        else:
          self.a2r[a_id] = 0

      self.relations.append("NONE")  # reserved for SHIFT or REDUCE
      self.rel_emb = nn.Embedding(len(self.relations), options.rel_emb_dim)

    self.softmax = nn.LogSoftmax() # LogSoftmax works on final dimension only for two dimension tensors. Be careful.

    self.composition_logging_factor = 1000
    self.composition_logging_count = 0


  def forward(self, tokens, tokens_mask, pre_tokens=None, postags=None, actions=None):
    """

    :param tokens: (seq_len, batch_size) tokens of the input sentence, preprocessed as vocabulary indexes
    :param tokens_mask: (seq_len, batch_size) indication of whether this is a "real" token or a padding
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
    token_comp_output = self.compose_tokens(token_comp_input) # (batch_size, seq_len, input_dim), pad at end
    token_comp_output = torch.transpose(token_comp_output, 0, 1) # (seq_len, batch_size, input_dim), pad at end
    token_comp_output_rev = utils.tensor.masked_revert(token_comp_output, tokens_mask.unsqueeze(2).expand(seq_len, batch_size, self.input_dim), 0) # (seq_len, batch_size, input_dim), pad at end

    # initialize stack
    self.stack.build_stack(batch_size, self.gpuid)

    # initialize and preload buffer
    buffer_hiddens = Variable(torch.zeros((seq_len, batch_size, self.hid_dim)).type(self.dtype))
    buffer_cells = Variable(torch.zeros((seq_len, batch_size, self.hid_dim)).type(self.dtype))
    bh = self.h0.unsqueeze(0).unsqueeze(2).expand(batch_size, self.hid_dim, self.num_lstm_layers)
    bc = self.c0.unsqueeze(0).unsqueeze(2).expand(batch_size, self.hid_dim, self.num_lstm_layers)
    for t_i in range(seq_len):
      bh, bc = self.pre_buffer(token_comp_output_rev[t_i], (bh, bc)) # (batch_size, self.hid_dim, self.num_lstm_layers)
      buffer_hiddens[t_i, :, :] = bh[:, :, -1]
      buffer_cells[t_i, :, :] = bc[:, :, -1]

    self.buffer.build_stack(batch_size, seq_len, buffer_hiddens, tokens_mask, self.gpuid)
    self.token_stack.build_stack(batch_size, self.stack_size, gpuid=self.gpuid)
    self.token_buffer.build_stack(batch_size, seq_len, token_comp_output_rev, tokens_mask, self.gpuid)
    self.stack(self.root_input.unsqueeze(0).expand(batch_size, self.input_dim), Variable(torch.ones(batch_size).type(self.long_dtype))) # push root

    stack_state, _ = self.stack.head() # (batch_size, hid_dim)
    buffer_state = self.buffer.head() # (batch_size, hid_dim)
    token_stack_state = self.token_stack.head()
    token_buffer_state = self.token_buffer.head()
    stack_input =  token_buffer_state # (batch_size, input_size)
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
    step_i = 0
    # during decoding, some instances in the batch may terminate earlier than others
    # and we cannot terminate the decoding because one instance terminated
    while step_i < step_length:
      # get action decisions
      if self.use_token_highway:
        summary = self.summarize_states(torch.cat((stack_state, token_stack_state, buffer_state, token_buffer_state, action_state), dim=1))
      else:
        summary = self.summarize_states(torch.cat((stack_state, buffer_state, action_state), dim=1)) # (batch_size, self.state_dim)
      action_dist = self.softmax(self.state_to_actions(summary)) # (batch_size, len(actions))
      outputs[step_i, :, :] = action_dist.clone()

      # get rid of forbidden actions (only for decoding)
      if actions is None or self.exposure_eps < 1.0:
        forbidden_actions = self.get_valid_actions(batch_size) ^ 1
        # print("stack size = {0}".format(self.stack.size()))
        # print("buffer size = {0}".format(self.buffer.size()))
        # print("forbidden actions = {0}".format(torch.sum(forbidden_actions, dim=1).tolist()))
        num_forbidden_actions = torch.sum(forbidden_actions, dim=1)
        # if all actions except <pad> and <unk> are forbidden for all sentences in the batch,
        # there is no sense continue decoding.
        if (num_forbidden_actions == (len(self.actions) - 2)).all():
          break
        action_dist.masked_fill_(Variable(forbidden_actions), -999)

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
      # this should be controlled by the valid action above
      # stack_pop_boundary = Variable((self.stack.pos == 0).data & (stack_op == -1).data)
      # stack_push_boundary = Variable((self.stack.pos >= self.stack_size).data & (stack_op == 1).data)
      # buffer_pop_boundary = Variable((self.buffer.pos ==0).data & (buffer_op == -1).data)

      # if stack/buffer is at boundary, hold at current position rather than push/pop
      # stack_op = stack_op.masked_fill(stack_pop_boundary, 0)
      # stack_op = stack_op.masked_fill(stack_push_boundary, 0)
      # buffer_op = buffer_op.masked_fill(buffer_pop_boundary, 0)

      # update stack, buffer and action state
      if self.composition_k > 0:
        self.token_composition(action_i, self.composition_k)  # not sure if token composition should happen before or after stack/buffer op, but let's try this
      stack_state, _ = self.stack(stack_input, stack_op)
      buffer_state = self.buffer(stack_state, buffer_op) # actually nothing will be pushed
      token_stack_state = self.token_stack(stack_input, stack_op)
      token_buffer_state = self.token_buffer(stack_input, buffer_op)
      stack_input = self.token_buffer.head()
      action_input = self.action_emb(action_i) # (batch_size, hid_dim)
      action_state, action_cell = self.history(action_input, (action_state, action_cell))

      step_i += 1

    # print("loop terminated.")
    # print("stack size = {0}".format(self.stack.size()))
    # print("buffer size = {0}".format(self.buffer.size()))
    # print("forbidden actions = {0}".format(torch.sum(forbidden_actions, dim=1).tolist()))

    return outputs


  def token_composition(self, action_ids, k):
    rel_ids = Variable(self.a2r[action_ids.data])  # XXX: gradient won't propagate through this
    batch_size = len(action_ids)
    emb_size = self.action_emb.embedding_dim

    action_emb = self.action_emb(action_ids).unsqueeze(1).expand(-1, k * 2, -1)  # (batch_size, k * 2, action_emb_size)
    active_token_emb = Variable(torch.Tensor(batch_size, k * 2, self.input_dim)).type(self.dtype)  # (batch_size, k * 2, input_dim)
    stack_pos = self.token_stack.pos.unsqueeze(0).expand(k, -1).data.clone()  # (k, batch_size)
    buffer_pos = self.token_buffer.pos.unsqueeze(0).expand(k, -1).data.clone()  # (k, batch_size)
    for kk in range(k):
      stack_pos[kk] -= kk
      buffer_pos[kk] -= kk
    stack_pos[(stack_pos < 0)] = 0
    buffer_pos[(buffer_pos < 0)] = 0
    active_token_emb[:, 0:k, :] = \
      self.token_stack.hidden_stack[ stack_pos, torch.arange(0, batch_size).unsqueeze(0).expand(k, -1).type(self.long_dtype), :]  # (batch_size, k, input_dim)
    active_token_emb[:, k:k*2, :] = \
      self.token_buffer.hidden_stack[ buffer_pos, torch.arange(0, batch_size).unsqueeze(0).expand(k, -1).type(self.long_dtype), :]  # (batch_size, k, input_dim)

    # calculate attention weights
    alpha_h = self.W_h(action_emb.contiguous().view(-1, emb_size), active_token_emb.contiguous().view(-1, self.input_dim))  # (batch_size * k * 2)
    alpha_d = self.W_d(action_emb.contiguous().view(-1, emb_size), active_token_emb.contiguous().view(-1, self.input_dim))
    alpha_b = self.W_b(action_emb.contiguous().view(-1, emb_size), active_token_emb.contiguous().view(-1, self.input_dim))

    alpha_h = alpha_h.view(batch_size, -1)  # (batch_size, k * 2)
    alpha_d = alpha_d.view(batch_size, -1)  # (batch_size, k * 2)
    alpha_b = alpha_b.view(batch_size, -1)  # (batch_size, k * 2)

    alpha_h = torch.nn.functional.softmax(alpha_h)
    alpha_d = torch.nn.functional.softmax(alpha_d)
    alpha_b = torch.nn.functional.sigmoid(alpha_b)

    if torch.sum(alpha_h).data[0] != torch.sum(alpha_h).data[0]:
      pdb.set_trace()

    if torch.sum(alpha_d).data[0] != torch.sum(alpha_d).data[0]:
      pdb.set_trace()

    if torch.sum(alpha_b).data[0] != torch.sum(alpha_b).data[0]:
      pdb.set_trace()

    self.composition_logging_count += 1
    if self.composition_logging_count % self.composition_logging_factor == 0:
      logging.info("sample action: {0}".format(self.actions[action_ids.data[0]]))
      logging.info("head attention value sample: {0}".format(alpha_h[0, :].data.tolist()))
      logging.info("dependency attention value sample: {0}".format(alpha_d[0, :].data.tolist()))
      logging.info("writing head weights value sample; {0}".format(alpha_b[0, :].data.tolist()))

    # use weighted sum to get attentional h, d, and r
    h = torch.sum(active_token_emb.permute(2, 0, 1) * alpha_h, dim=2).t()  # (input_dim, batch_size, k*2) * (batch_size, k * 2) -> sum dim2 -> transpose -> (batch_size, input_dim)
    d = torch.sum(active_token_emb.permute(2, 0, 1) * alpha_d, dim=2).t()  # (input_dim, batch_size, k*2) * (batch_size, k * 2) -> sum dim2 -> transpose -> (batch_size, input_dim)
    r = self.rel_emb(rel_ids)  # (batch_size, rel_dim)

    # compose
    c = self.composition_func(torch.cat([h, d, r], dim=1))  # (batch_size, input_dim)
    c = c.unsqueeze(0).expand(k, -1, -1)  # (k, batch_size, input_dim)

    # c does not need to be summed over the k dimension because they'll be written to different units of stack / buffer
    stack_elements = self.token_stack.hidden_stack[stack_pos, torch.arange(0, stack_pos.size(1)).type(self.long_dtype), :]
    buffer_elements = self.token_buffer.hidden_stack[buffer_pos, torch.arange(0, buffer_pos.size(1)).type(self.long_dtype), :]
    self.token_stack.hidden_stack[stack_pos, torch.arange(0, stack_pos.size(1)).type(self.long_dtype), :] = (stack_elements.permute(2, 1, 0) * (1 - alpha_b[:, 0:k])).permute(2, 1, 0) + (c.permute(2, 1, 0) * alpha_b[:, 0:k]).permute(2, 1, 0)  # (input_dim, batch_size, k) * (batch_size, k) -> permute -> (k, batch_size, input_dim)
    self.token_buffer.hidden_stack[buffer_pos, torch.arange(0, buffer_pos.size(1)).type(self.long_dtype), :] = (buffer_elements.permute(2, 1, 0) * (1 - alpha_b[:, k:2*k])).permute(2, 1, 0) + (c.permute(2, 1, 0) * alpha_b[:, k:2*k]).permute(2, 1, 0)  # (input_dim, batch_size, k) * (batch_size, k) -> permute -> (k, batch_size, input_dim)


  def get_valid_actions(self, batch_size):
    action_mask = torch.ones(batch_size, len(self.actions)).type(self.long_dtype).byte()  # (batch_size, len(actions))

    for idx, action_str in enumerate(self.actions):
      # general popping safety
      sa = self.stack_action_mapping[idx].repeat(batch_size)
      ba = self.buffer_action_mapping[idx].repeat(batch_size)
      stack_forbid = ((sa == -1).data & (self.stack.pos <= 1).data)
      buffer_forbid = ((ba == -1).data & (self.buffer.pos == 0).data)
      action_mask[:, idx] = (action_mask[:, idx] & ((stack_forbid | buffer_forbid) ^ 1))

      # transition-system specific operation safety
      if self.transSys == TransitionSystems.AER:
        la_forbid = (((self.buffer.pos == 0).data | (self.stack.pos <= 1).data) & action_str.startswith("Left-Arc"))
        ra_forbid = (((self.stack.pos == 0).data | (self.stack.pos == 0).data) & action_str.startswith("Right-Arc"))
        action_mask[:, idx] = (action_mask[:, idx] & ((la_forbid | ra_forbid) ^ 1))

      if self.transSys == TransitionSystems.AH:
        la_forbid = (((self.buffer.pos == 0).data | (self.stack.pos <= 1).data) & action_str.startswith("Left-Arc"))
        ra_forbid = (self.stack.pos <= 1).data & action_str.startswith("Right-Arc")
        action_mask[:, idx] = (action_mask[:, idx] & ((la_forbid | ra_forbid) ^ 1))

    return action_mask


  def set_action_mappings(self):
    for idx, astr in enumerate(self.actions):
      transition = astr.split('|')[0]

      # If the action is unknown, leave the stack and buffer state as-is
      if self.transSys == TransitionSystems.AER:
        self.stack_action_mapping[idx], self.buffer_action_mapping[idx] = AER_map.get(transition, (0, 0))
      elif self.transSys == TransitionSystems.AH:
        self.stack_action_mapping[idx], self.buffer_action_mapping[idx] = AH_map.get(transition, (0, 0))
      else:
        logging.fatal("Unimplemented transition system.")
        raise NotImplementedError
