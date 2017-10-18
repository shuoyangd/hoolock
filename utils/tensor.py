import math
import torch

def truncate_or_pad(tensor, dim, length, pad_index=0):

  """
  truncate or pad the tensor on a given dimension to the given length.

  :param tensor:
  :param dim:
  :param length:
  :param pad_index:
  """
  orig_length = tensor.size()[dim]
  # truncate
  if orig_length > length:
    return tensor.index_select(dim, torch.arange(0, length).long())
  # pad
  else:
    # make pad
    pad_length = length - orig_length
    pad_size = list(tensor.size())
    pad_size[dim] = pad_length
    pad_size = tuple(pad_size)
    pad = (torch.ones(pad_size) * pad_index).long()

    return torch.cat((tensor, pad), dim=dim)

def advanced_batchize(data, batch_size, pad_index):
  """

  :param data: [(sent_len,)]
  :param batch_size:
  :param pad_index:
  :return [(seq_len, batch_size)], order of sorted data
  """
  # rank by data length
  sorted_data = sorted(data, key=lambda sent: len(sent))
  sort_index = sorted(range(len(data)), key=lambda k: len(data[k]))
  batchized_data = []
  batchized_mask = []
  # except for last batch
  for start_i in range(0, len(sorted_data) - batch_size, batch_size):
    batch_data = sorted_data[start_i: start_i + batch_size]
    seq_len = len(batch_data[-1])
    batch_tensor = (torch.ones((seq_len, batch_size)) * pad_index).long()
    mask_tensor = torch.zeros((seq_len, batch_size)).byte()
    for idx, sent_data in enumerate(batch_data):
      # batch_tensor[:, idx] = truncate_or_pad(sent_data, 0, seq_len, pad_index=pad_index)
      batch_tensor[0:len(sent_data), idx] = sent_data
      mask_tensor[0:len(sent_data), idx] = 1
    batchized_data.append(batch_tensor)
    batchized_mask.append(mask_tensor)

  # last batch
  if len(sorted_data) % batch_size != 0:
    batch_data = sorted_data[len(sorted_data) // batch_size * batch_size:]
    seq_len = len(batch_data[-1])
    batch_tensor = (torch.ones((seq_len, batch_size)) * pad_index).long()
    mask_tensor = torch.zeros((seq_len, batch_size)).byte()
    for idx, sent_data in enumerate(batch_data):
      batch_tensor[0:len(sent_data), idx] = sent_data
      mask_tensor[0:len(sent_data), idx] = 1
    batchized_data.append(batch_tensor)
    batchized_mask.append(mask_tensor)
  return batchized_data, batchized_mask, sort_index

def advanced_batchize_no_sort(data, batch_size, pad_index, order=None):
  """

  :param data: [(sent_len,)]
  :param batch_size:
  :param pad_index:
  :param order: (optional) desired order of data
  :return [(seq_len, batch_size)]
  """
  if order is not None:
    data = [data[i] for i in order]
  batchized_data = []
  batchized_mask = []
  # except for last batch
  for start_i in range(0, len(data) - batch_size, batch_size):
    batch_data = data[start_i: start_i + batch_size]
    seq_len = max([len(batch_data[i]) for i in range(len(batch_data))])  # find longest seq
    batch_tensor = (torch.ones((seq_len, batch_size)) * pad_index).long()
    mask_tensor = torch.zeros((seq_len, batch_size)).byte()
    for idx, sent_data in enumerate(batch_data):
      # batch_tensor[:, idx] = truncate_or_pad(sent_data, 0, seq_len, pad_index=pad_index)
      batch_tensor[0:len(sent_data), idx] = sent_data
      mask_tensor[0:len(sent_data), idx] = 1
    batchized_data.append(batch_tensor)
    batchized_mask.append(mask_tensor)

  # last batch
  if len(data) % batch_size != 0:
    batch_data = data[len(data) // batch_size * batch_size:]
    seq_len = max([len(batch_data[i]) for i in range(len(batch_data))])  # find longest seq
    batch_tensor = (torch.ones((seq_len, batch_size)) * pad_index).long()
    mask_tensor = torch.zeros((seq_len, batch_size)).byte()
    for idx, sent_data in enumerate(batch_data):
      batch_tensor[0:len(sent_data), idx] = sent_data
      mask_tensor[0:len(sent_data), idx] = 1
    batchized_data.append(batch_tensor)
    batchized_mask.append(mask_tensor)
  return batchized_data, batchized_mask

def batchize(tensor, batch_size, pad_index):
  """
  Prepare a data with shape (num_instances, seq_len, *)
  into (ceil(num_instances / batch_size), seq_len, batch_size, *),
  and pad dummy instances when applicable.

  :param tensor:
  :param batch_size:
  :param pad_index:
  """
  batched_size = list(tensor.size())
  num_instances = tensor.size()[0]
  num_batched_instances = math.ceil(num_instances / batch_size)
  batched_size.insert(2, batch_size)
  batched_size[0] = num_batched_instances
  padded_tensor = truncate_or_pad(tensor, 0, num_batched_instances * batch_size, pad_index)
  return padded_tensor.view(batched_size)
