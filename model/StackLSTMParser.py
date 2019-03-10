from enum import Enum
import logging
import pdb
import time
import utils
import utils.rand

import torch
from torch import nn
import torch.nn.functional as F
# from torch.autograd import Variable
from model.StackLSTMCell import StackLSTMCell, ReducerLSTMCell
from model.StateStack import StateStack, StateReducer
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


ASd_map = {"Shift": (1, -1, 0), "Left-Arc": (-1, 0, 0), "Right-Arc": (-1, 0, 1)}
AER_map = {"Shift": (1, -1), "Reduce": (-1, 0), "Left-Arc": (-1, 0), "Right-Arc": (1, -1)}
AH_map = {"Shift": (1, -1), "Left-Arc": (-1, 0), "Right-Arc": (-1, 0)}

AER_hard_composition_map = {"Shift": ([0, 0], [0, 0], [0, 0]), "Reduce": ([0, 0], [0, 0], [0, 0]),
                            "Left-Arc": ([0, 1], [1, 0], [0, 1]), "Right-Arc": ([1, 0], [0, 1], [0, 1])}
AH_hard_composition_map = {"Shift": ([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]),
                           "Left-Arc": ([0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]),
                           "Right-Arc": ([0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0])}


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
    self.dropout_rate = options.dropout_rate

    # gpu
    self.gpuid = options.gpuid
    self.dtype = torch.cuda.FloatTensor if len(options.gpuid) >= 1 else torch.FloatTensor
    self.long_dtype = torch.cuda.LongTensor if len(options.gpuid) >= 1 else torch.LongTensor

    # vocabularies
    self.vocab = vocab
    self.pre_vocab = pre_vocab
    self.actions = actions
    self.postags = postags
    self.la = torch.LongTensor([ 1 if action_str.startswith("Left-Arc") else 0 for action_str in self.actions ]).type(self.long_dtype).byte()
    self.ra = torch.LongTensor([ 1 if action_str.startswith("Right-Arc") else 0 for action_str in self.actions ]).type(self.long_dtype).byte()

    # action mappings
    self.stack_action_mapping = torch.zeros(len(actions),).type(self.long_dtype)
    self.buffer_action_mapping = torch.zeros(len(actions),).type(self.long_dtype)
    self.dir_mapping = torch.zeros(len(actions),).type(self.long_dtype)
    self.set_action_mappings()

    # embeddings
    self.word_emb = nn.Embedding(len(vocab), options.word_emb_dim)
    self.action_emb = nn.Embedding(len(actions), options.action_emb_dim)

    if pre_vocab is not None:
      self.pre_word_emb = nn.Embedding(pre_vocab.vectors.size(0), pre_vocab.dim)
      # initialzie and fixed
      self.pre_word_emb.weight = torch.nn.Parameter(pre_vocab.vectors)
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

    self.relations = []
    self.hard_composition = options.hard_composition
    self.composition_k = options.composition_k
    if self.hard_composition or self.composition_k > 0 or \
       (hasattr(options, "use_relations") and options.use_relations):
      self.relations.append("NONE")
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

    # recurrent components
    # the only reason to have unified h0 and c0 is because pre_buffer and buffer should have the same initial states
    # but due to simplicity we just borrowed this for all three recurrent components (stack, buffer, action)
    self.h0 = nn.Parameter(torch.rand(options.hid_dim,).type(self.dtype))
    self.c0 = nn.Parameter(torch.rand(options.hid_dim,).type(self.dtype))
    # BufferLSTM could have 0 or 2 parameters, depending on what is passed for initial hidden and cell state
    if self.transSys == TransitionSystems.ASd:
      self.stack = ReducerLSTMCell(options.input_dim, options.hid_dim, options.stack_size, options.num_lstm_layers, self.h0, self.c0)
      self.buffer = StateStack(options.hid_dim, self.h0)
      if hasattr(options, "use_relations") and options.use_relations:
        self.token_stack = StateReducer(options.input_dim, relations=self.relations, rel_emb_dim=options.rel_emb_dim)
      else:
        self.token_stack = StateReducer(options.input_dim)
      self.token_buffer = StateStack(options.input_dim) # elememtns in this buffer has size input_dim so h0 and c0 won't fit
    else:
      self.stack = StackLSTMCell(options.input_dim, options.hid_dim, options.stack_size, options.num_lstm_layers, self.h0, self.c0)
      self.buffer = StateStack(options.hid_dim, self.h0)
      self.token_stack = StateStack(options.input_dim)
      self.token_buffer = StateStack(options.input_dim) # elememtns in this buffer has size input_dim so h0 and c0 won't fit

    # self.pre_buffer = MultiLayerLSTMCell(input_size=options.input_dim, hidden_size=options.hid_dim, num_layers=options.num_lstm_layers)
    self.pre_buffer = nn.RNN(options.input_dim, options.hid_dim, options.num_lstm_layers)
    self.history = MultiLayerLSTMCell(input_size=options.action_emb_dim, hidden_size=options.hid_dim, num_layers=options.num_lstm_layers)

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
    if self.hard_composition or self.composition_k > 0:
      self.composition_func = nn.Sequential(
          nn.Linear(self.input_dim * 2 + options.rel_emb_dim, self.input_dim),
          nn.Tanh()
      )

      self.rel_emb = nn.Embedding(len(self.relations), options.rel_emb_dim)

    if self.hard_composition:
      if self.transSys == TransitionSystems.AER:
        self.composition_k = 1
      elif self.transSys == TransitionSystems.AH:
        self.composition_k = 2
      else:
        raise NotImplementedError

      self.composition_attn_h = torch.zeros(len(self.actions), self.composition_k * 2).tolist()
      self.composition_attn_d = torch.zeros(len(self.actions), self.composition_k * 2).tolist()
      self.composition_attn_b = torch.zeros(len(self.actions), self.composition_k * 2).tolist()

      self.set_hard_composition_mappings()

      self.composition_attn_h = torch.FloatTensor(self.composition_attn_h).type(self.dtype)
      self.composition_attn_d = torch.FloatTensor(self.composition_attn_d).type(self.dtype)
      self.composition_attn_b = torch.FloatTensor(self.composition_attn_b).type(self.dtype)

    elif self.composition_k > 0:
      self.W_h = nn.Bilinear(options.action_emb_dim, self.input_dim, 1, bias=False)
      self.W_d = nn.Bilinear(options.action_emb_dim, self.input_dim, 1, bias=False)
      self.W_b = nn.Bilinear(options.action_emb_dim, self.input_dim, 1, bias=False)

    self.softmax = nn.LogSoftmax(dim=-1) # LogSoftmax works on final dimension only for two dimension tensors. Be careful.

    self.composition_logging_factor = 1000
    self.composition_logging_count = 0

    # use gumbel-softmax
    self.st_gumbel_softmax = options.st_gumbel_softmax


  def forward(self, tokens, tokens_mask, pre_tokens=None, postags=None, actions=None):
    """

    :param tokens: (seq_len, batch_size) tokens of the input sentence, preprocessed as vocabulary indexes
    :param tokens_mask: (seq_len, batch_size) indication of whether this is a "real" token or a padding
    :param postags: (seq_len, batch_size) optional POS-tag of the tokens of the input sentence, preprocessed as indexes
    :param actions: (seq_len, batch_size) optional golden action sequence, preprocessed as indexes
    :return: output: (output_seq_len, batch_size, len(actions)), or when actions = None, (output_seq_len, batch_size, max_step_length)
    """

    start_time = time.time()
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
    time1 = time.time()
    logging.debug("duration 1 is {0}".format(time1 - start_time))

    # initialize and preload buffer
    b0 = self.h0.unsqueeze(0).unsqueeze(0).expand(self.num_lstm_layers, batch_size, self.hid_dim).contiguous()
    buffer_hiddens, _ = self.pre_buffer(token_comp_output_rev, b0)

    time2 = time.time()
    logging.debug("duration 2 is {0}".format(time2 - time1))

    self.buffer.build_stack(batch_size, seq_len, buffer_hiddens, tokens_mask, self.gpuid)
    self.token_stack.build_stack(batch_size, self.stack_size, gpuid=self.gpuid)
    self.token_buffer.build_stack(batch_size, seq_len, token_comp_output_rev, tokens_mask, self.gpuid)
    self.stack(self.root_input.unsqueeze(0).expand(batch_size, self.input_dim), torch.ones(batch_size).type(self.long_dtype)) # push root

    time3 = time.time()
    logging.debug("duration 3 is {0}".format(time3 - time2))

    stack_state, _ = self.stack.head() # (batch_size, hid_dim)
    buffer_state = self.buffer.head() # (batch_size, hid_dim)
    token_stack_state = self.token_stack.head()
    token_buffer_state = self.token_buffer.head()
    stack_input = token_buffer_state # (batch_size, input_size)
    action_state = self.h0.unsqueeze(0).unsqueeze(2).expand(batch_size, self.hid_dim, self.num_lstm_layers) # (batch_size, hid_dim, num_lstm_layers)
    action_cell = self.c0.unsqueeze(0).unsqueeze(2).expand(batch_size, self.hid_dim, self.num_lstm_layers) # (batch_size, hid_dim, num_lstm_layers)

    # apply dropout when necessary
    stack_state = F.dropout(stack_state, p=self.dropout_rate, training=self.training)
    buffer_state = F.dropout(buffer_state, p=self.dropout_rate, training=self.training)
    token_buffer_state = F.dropout(token_stack_state, p=self.dropout_rate, training=self.training)
    action_state = F.dropout(action_state, p=self.dropout_rate, training=self.training)
    action_cell = F.dropout(action_state, p=self.dropout_rate, training=self.training)

    # prepare bernoulli probability for exposure indicator
    batch_exposure_prob = torch.Tensor([self.exposure_eps] * batch_size).type(self.dtype)

    # main loop
    if actions is None:
      outputs = torch.zeros((self.max_step_length, batch_size, len(self.actions))).type(self.dtype)
      step_length = self.max_step_length
    else:
      outputs = torch.zeros((actions.size()[0], batch_size, len(self.actions))).type(self.dtype)
      step_length = actions.size()[0]

    time4 = time.time()
    logging.debug("duration 4 is {0}".format(time4 - time3))

    step_i = 0
    # during decoding, some instances in the batch may terminate earlier than others
    # and we cannot terminate the decoding because one instance terminated
    while step_i < step_length:
      # get action decisions
      if self.use_token_highway:
        summary = self.summarize_states(torch.cat((stack_state, token_stack_state, buffer_state, token_buffer_state, action_state[:, :, -1]), dim=1))
      else:
        summary = self.summarize_states(torch.cat((stack_state, buffer_state, action_state[:, :, -1]), dim=1)) # (batch_size, self.state_dim)
      if self.st_gumbel_softmax:
        action_dist = utils.rand.gumbel_softmax_sample(self.state_to_actions(summary)) # (batch_size, len(actions))
      else:
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
        action_dist.masked_fill_(forbidden_actions, -999)

      _, action_i = torch.max(action_dist, dim=1) # (batch_size,)

      # control exposure (only for training)
      if actions is not None:
        oracle_action_i = actions[step_i, :] # (batch_size,)
        batch_exposure = torch.bernoulli(batch_exposure_prob) # (batch_size,)
        action_i = (action_i.float() * (1 - batch_exposure) + oracle_action_i.float() * batch_exposure).long() # (batch_size,)
      # translate action into stack & buffer ops
      # stack_op, buffer_op = self.map_action(action_i.data)
      stack_op = self.stack_action_mapping.index_select(0, action_i)
      buffer_op = self.buffer_action_mapping.index_select(0, action_i)
      if self.transSys == TransitionSystems.ASd:
          dir_ = self.dir_mapping.index_select(0, action_i)

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
      if self.hard_composition or self.composition_k > 0:
        self.token_composition(action_i, self.composition_k)  # not sure if token composition should happen before or after stack/buffer op, but let's try this

      if self.transSys == TransitionSystems.ASd:
          # for reduce operations, reduced word embeddings should act as input
          # before this, stack_input is the buffer head from the previous timestep (ln 329)
          if self.relations:
            token_stack_state, reduced_stack_state = self.token_stack(stack_input, stack_op, dir_, self.a2r[action_i])
          else:
            token_stack_state, reduced_stack_state = self.token_stack(stack_input, stack_op, dir_)
            stack_input[(stack_op == -1), :] = reduced_stack_state[(stack_op == -1), :]
      else:
          token_stack_state = self.token_stack(stack_input, stack_op)

      stack_state, _ = self.stack(stack_input, stack_op)
      buffer_state = self.buffer(stack_state, buffer_op) # actually nothing will be pushed
      token_buffer_state = self.token_buffer(stack_input, buffer_op) # push might happen
      stack_input = self.token_buffer.head()
      action_input = self.action_emb(action_i) # (batch_size, action_emb_dim)
      action_state, action_cell = self.history(action_input, (action_state, action_cell)) # (batch_size, hid_dim, num_lstm_layers)

      # apply dropout when necessary
      stack_state = F.dropout(stack_state, p=self.dropout_rate, training=self.training)
      buffer_state = F.dropout(buffer_state, p=self.dropout_rate, training=self.training)
      token_buffer_state = F.dropout(token_stack_state, p=self.dropout_rate, training=self.training)
      action_state = F.dropout(action_state, p=self.dropout_rate, training=self.training)
      action_cell = F.dropout(action_state, p=self.dropout_rate, training=self.training)

      step_i += 1

    time5 = time.time()
    logging.debug("average loop duration is {0}, executed {1} times".format((time5 - time4) / step_i, step_i))

    # print("loop terminated.")
    # print("stack size = {0}".format(self.stack.size()))
    # print("buffer size = {0}".format(self.buffer.size()))
    # print("forbidden actions = {0}".format(torch.sum(forbidden_actions, dim=1).tolist()))

    return outputs


  def token_composition(self, action_ids, k):
    rel_ids = self.a2r[action_ids]  # XXX: gradient won't propagate through this
    batch_size = len(action_ids)
    emb_size = self.action_emb.embedding_dim

    action_emb = self.action_emb(action_ids).unsqueeze(1).expand(-1, k * 2, -1)  # (batch_size, k * 2, action_emb_size)
    active_token_emb = torch.Tensor(batch_size, k * 2, self.input_dim).type(self.dtype)  # (batch_size, k * 2, input_dim)
    stack_pos = self.token_stack.pos.unsqueeze(0).expand(k, -1).clone()  # (k, batch_size)
    buffer_pos = self.token_buffer.pos.unsqueeze(0).expand(k, -1).clone()  # (k, batch_size)
    for kk in range(k):
      stack_pos[kk] -= kk
      buffer_pos[kk] -= kk
    stack_pos_mask = (stack_pos < 0)  # (k, batch_size)
    buffer_pos_mask = (buffer_pos < 0)  # (k, batch_size)
    pos_mask = torch.cat((stack_pos_mask, buffer_pos_mask), dim=0).t()  # (batch_size, k * 2)

    stack_pos[(stack_pos < 0)] = 0
    buffer_pos[(buffer_pos < 0)] = 0
    active_token_emb[:, 0:k, :] = \
      self.token_stack.hidden_stack[ stack_pos, torch.arange(0, batch_size).unsqueeze(0).expand(k, -1).type(self.long_dtype), :].permute(1, 0, 2)  # (batch_size, k, input_dim)
    active_token_emb[:, k:k*2, :] = \
      self.token_buffer.hidden_stack[ buffer_pos, torch.arange(0, batch_size).unsqueeze(0).expand(k, -1).type(self.long_dtype), :].permute(1, 0, 2)  # (batch_size, k, input_dim)

    # calculate attention weights
    if self.hard_composition:
      alpha_h = self.composition_attn_h[action_ids, :]
      alpha_d = self.composition_attn_d[action_ids, :]
      alpha_b = self.composition_attn_b[action_ids, :]
    else:
      alpha_h = self.W_h(action_emb.contiguous().view(-1, emb_size), active_token_emb.contiguous().view(-1, self.input_dim))  # (batch_size * k * 2)
      alpha_d = self.W_d(action_emb.contiguous().view(-1, emb_size), active_token_emb.contiguous().view(-1, self.input_dim))
      alpha_b = self.W_b(action_emb.contiguous().view(-1, emb_size), active_token_emb.contiguous().view(-1, self.input_dim))

      alpha_h = alpha_h.view(batch_size, -1)  # (batch_size, k * 2)
      alpha_d = alpha_d.view(batch_size, -1)  # (batch_size, k * 2)
      alpha_b = alpha_b.view(batch_size, -1)  # (batch_size, k * 2)

      alpha_h = torch.nn.functional.softmax(alpha_h)
      alpha_d = torch.nn.functional.softmax(alpha_d)
      alpha_b = torch.nn.functional.sigmoid(alpha_b)

    # the masked positions should not be considered as valid
    alpha_h[pos_mask] = 0
    alpha_d[pos_mask] = 0
    alpha_b[pos_mask] = 0

    if torch.sum(alpha_h).item() != torch.sum(alpha_h).item():
      pdb.set_trace()

    if torch.sum(alpha_d).item() != torch.sum(alpha_d).item():
      pdb.set_trace()

    if torch.sum(alpha_b).item() != torch.sum(alpha_b).item():
      pdb.set_trace()

    self.composition_logging_count += 1
    if self.composition_logging_count % self.composition_logging_factor == 0:
      logging.info("sample action: {0}".format(self.actions[action_ids[0].item()]))
      logging.info("head attention value sample: {0}".format(alpha_h[0, :].tolist()))
      logging.info("dependency attention value sample: {0}".format(alpha_d[0, :].tolist()))
      logging.info("writing head weights value sample; {0}".format(alpha_b[0, :].tolist()))

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

    # general popping safety
    sa = self.stack_action_mapping.unsqueeze(0).expand(batch_size, -1)  # (batch_size, len(actions))
    ba = self.buffer_action_mapping.unsqueeze(0).expand(batch_size, -1)  # (batch_size, len(actions))
    stack_forbid = ((sa == -1) & (self.stack.pos <= 1).unsqueeze(1).expand(-1, len(self.actions)))  # (batch_size, len(actions))
    buffer_forbid = ((ba == -1) & (self.buffer.pos <= 0).unsqueeze(1).expand(-1, len(self.actions)))  # (batch_size, len(actions))
    action_mask = (action_mask & ((stack_forbid | buffer_forbid) ^ 1))

    # transition-system specific operation safety
    if self.transSys == TransitionSystems.ASd:
      la_forbid = (self.stack.pos <= 2).unsqueeze(1).expand(-1, len(self.actions)) & \
                  self.la.unsqueeze(0).expand(batch_size, -1)  # root
      # ra_forbid = (self.stack.pos <= 1) & action_str.startswith("Right-Arc")  # same as stack_forbid
      action_mask = (action_mask & (la_forbid ^ 1))

    elif self.transSys == TransitionSystems.AER:
      la_forbid = (((self.buffer.pos <= 0).unsqueeze(1).expand(-1, len(self.actions)) |
                    (self.stack.pos <= 1).unsqueeze(1).expand(-1, len(self.actions))) &
                    self.la.unsqueeze(0).expand(batch_size, -1))
      ra_forbid = (((self.stack.pos <= 0).unsqueeze(1).expand(-1, len(self.actions)) |
                    (self.stack.pos <= 0).unsqueeze(1).expand(-1, len(self.actions))) &
                    self.ra.unsqueeze(0).expand(batch_size, -1))
      action_mask = (action_mask & ((la_forbid | ra_forbid) ^ 1))

    elif self.transSys == TransitionSystems.AH:
      la_forbid = (((self.buffer.pos <= 0).unsqueeze(1).expand(-1, len(self.actions)) |
                    (self.stack.pos <= 1).unsqueeze(1).expand(-1, len(self.actions))) &
                    self.la.unsqueeze(0).expand(batch_size, -1))
      ra_forbid = ((self.stack.pos <= 1).unsqueeze(1).expand(-1, len(self.actions)) &
                  self.ra.unsqueeze(0).expand(batch_size, -1))
      action_mask = (action_mask & ((la_forbid | ra_forbid) ^ 1))

    else:
      logging.fatal("Unimplemented transition system.")
      raise NotImplementedError

    return action_mask


  def set_action_mappings(self):
    for idx, astr in enumerate(self.actions):
      transition = astr.split('|')[0]

      # If the action is unknown, leave the stack and buffer state as-is
      if self.transSys == TransitionSystems.AER:
        self.stack_action_mapping[idx], self.buffer_action_mapping[idx] = AER_map.get(transition, (0, 0))
      elif self.transSys == TransitionSystems.AH:
        self.stack_action_mapping[idx], self.buffer_action_mapping[idx] = AH_map.get(transition, (0, 0))
      elif self.transSys == TransitionSystems.ASd:
        self.stack_action_mapping[idx], self.buffer_action_mapping[idx], self.dir_mapping[idx] = ASd_map.get(transition, (0, 0, 0))
      else:
        logging.fatal("Unimplemented transition system.")
        raise NotImplementedError


  def set_hard_composition_mappings(self):
    for idx, astr in enumerate(self.actions):
      transition = astr.split('|')[0]

      # If the action is unknown, read/write attention weights should all be 0
      if self.transSys == TransitionSystems.AER:
        self.composition_attn_h[idx], self.composition_attn_d[idx], self.composition_attn_b[idx] = \
          AER_hard_composition_map.get(transition, ([0] * self.composition_k * 2, [0] * self.composition_k * 2, [0] * self.composition_k * 2))
      if self.transSys == TransitionSystems.AH:
        self.composition_attn_h[idx], self.composition_attn_d[idx], self.composition_attn_b[idx] = \
          AH_hard_composition_map.get(transition, ([0] * self.composition_k * 2, [0] * self.composition_k * 2, [0] * self.composition_k * 2))
      else:
        logging.fatal("Unimplemented transition system.")
        raise NotImplementedError

