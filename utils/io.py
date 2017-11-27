from model.StackLSTMParser import TransitionSystems
import logging

import pdb

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


class CoNLLReader:
  def __init__(self, file):
    """

    :param file: FileIO object
    """
    self.file = file

  def __iter__(self):
    return self

  def __next__(self):
    sent = self.readsent()
    if sent == []:
      raise StopIteration()
    else:
      return sent

  def readsent(self):
    """
    Assuming CoNLL-U format, where the columns are:
    ID FORM LEMMA UPOSTAG XPOSTAG FEATS HEAD DEPREL DEPS MISC
    """
    sent = []
    row_str = self.file.readline().strip()
    while row_str != "":
      row = {}
      columns = row_str.split()
      row["ID"] = int(columns[0])
      row["FORM"] = columns[1]
      row["LEMMA"] = columns[2] if len(columns) > 2 else "_"
      row["UPOSTAG"] = columns[3] if len(columns) > 3 else "_"
      row["XPOSTAG"] = columns[4] if len(columns) > 4 else "_"
      row["FEATS"] = columns[5] if len(columns) > 5 else "_"
      row["HEAD"] = columns[6] if len(columns) > 6 else "_"
      row["DEPREL"] = columns[7] if len(columns) > 7 else "_"
      row["DEPS"] = columns[8] if len(columns) > 8 else "_"
      row["MISC"] = columns[9] if len(columns) > 9 else "_"
      sent.append(row)
      row_str = self.file.readline().strip()
    return sent

  def close(self):
    self.file.close()


class OracleReader:
  def __init__(self, file):
    self.file = file

  def __iter__(self):
    return self

  def __next__(self):
    sent = self.readsent()
    if sent == []:
      raise StopIteration()
    else:
      return sent

  def readsent(self):
    """
    Using oracle format from Peng Qi's code at https://github.com/qipeng/arc-swift,
    but currently do not support transition system ASw (arc-swift)
    """
    sent = []
    row_str = self.file.readline().strip()
    while row_str != "":
      row = {}
      columns = row_str.split()
      row["OP"] = columns[0]
      if row["OP"] == "Reduce":
        pass
      elif row["OP"] == "Left-Arc" or row["OP"] == "Right-Arc":
        row["DEPREL"] = columns[1]
        if len(columns) >= 4:
          row["UPOSTAG"] = columns[2]
          row["XPOSTAG"] = columns[3]
      else:
        row["UPOSTAG"] = columns[1]
        row["XPOSTAG"] = columns[2]
      sent.append(row)
      row_str = self.file.readline().strip()
    return sent

  def close(self):
    self.file.close()

class OracleWriter:
  def __init__(self, file, actions, pad_idx, total_sent_n):
    self.file = file
    self.actions = actions
    self.pad_idx = pad_idx
    self.total_sent_n = total_sent_n
    self.output_sent_n = 0

  def writesent(self, sent_actions):
    for sent_i in zip(range(sent_actions.size()[1])):
      for action_i in range(sent_actions.size()[0]):
        action_idx = sent_actions.data[action_i][sent_i]
        if action_idx == self.pad_idx:
          break
        action_str = self.actions[action_idx]
        action_fields = action_str.split('|')

        # pseudo POSTAG
        # if action_fields[0] == "Left-Arc" \
        #     or action_fields[0] == "Right-Arc" \
        #     or action_fields[0] == "Shift":
        #   action_fields.append("POSX")
        #   action_fields.append("POSX")

        self.file.write("\t".join(action_fields) + "\n")
      self.file.write("\n")

      self.output_sent_n += 1
      if self.output_sent_n >= self.total_sent_n:
        break

  def close(self):
    self.file.close()

class Oracle2CoNLLWriter:
  def __init__(self, file, transSys):
    self.file = file
    self.transSys = transSys

  def terminated(self, j, t, stack, buf):
    if self.transSys == TransitionSystems.ASd:
      raise NotImplementedError
    elif self.transSys == TransitionSystems.AES or \
        self.transSys == TransitionSystems.AER:
      if not j:
        return 1
      elif len(stack) < 2 and t["OP"] == "Left-Arc":
        return 2
      else:
        return 0
    elif self.transSys == TransitionSystems.ASw:
      raise NotImplementedError
    elif self.transSys == TransitionSystems.AH:
      if (not j) and (t["OP"] == "Shift" or t["OP"] == "Left-Arc"):
        return 1
      elif len(stack) < 2 and t["OP"] == "Right-Arc":
        return 2
      elif len(stack) < 1 and t["OP"] == "Left-Arc":
        return 2
      else:
        return 0
    else:
      raise NotImplementedError

  def writesent(self, lines, ref_lines=None):
    """
    This function is (almost) blindly copied from Peng Qi's code, with some modifications:
    https://github.com/qipeng/arc-swift/blob/master/test/test_oracle.py

    :param lines: the oracle of a single sentence
    :param ref_lines: optional CoNLL reference of the same sentence
    """
    stack = [0]
    Nwords = 0
    if not ref_lines:
      for t in lines:
        if self.transSys in [TransitionSystems.ASw, TransitionSystems.AER, TransitionSystems.AES]:
          if t["OP"] == 'Shift' or t["OP"] == 'Right-Arc':
            Nwords += 1
        elif self.transSys in [TransitionSystems.ASd, TransitionSystems.AH]:
          if t["OP"] == 'Shift':
            Nwords += 1
    else:
      Nwords = len(ref_lines)

    buf = [i + 1 for i in range(Nwords)]

    parent = [-1 for i in range(Nwords + 1)]
    output = ["" for i in range(Nwords + 1)]

    pos = ["" for i in range(Nwords + 1)]

    def output_str(idx, head_idx, relation, ref_lines=None):
      if ref_lines:
        ref_line = ref_lines[idx - 1] # output is 1-based, ref_lines is 0-based
        ret = str(ref_line["ID"])
        ret += ("\t" + ref_line["FORM"])
        ret += ("\t" + ref_line["LEMMA"])
        ret += ("\t" + ref_line["UPOSTAG"])
        ret += ("\t" + ref_line["XPOSTAG"])
        ret += ("\t" + ref_line["FEATS"])
        ret += ("\t" + str(head_idx))
        ret += ("\t" + str(relation))
        ret += ("\t" + ref_line["DEPS"])
        ret += ("\t" + ref_line["MISC"])
      else:
        ret = "%d\t%s" % (head_idx, relation)

      return ret

    # ret = 0 -> peacefully return
    # ret = 1 -> premature buffer emptiness
    # ret = 2 -> premature stack emptiness
    ret = 0
    for t in lines:
      j = None if len(buf) == 0 else buf[0]
      ret = self.terminated(j, t, stack, buf)
      if ret != 0:
        break
      if self.transSys == TransitionSystems.ASw:
        if t["OP"] == 'Left-Arc':
          n = int(t[1]) - 1
          relation = t[2]
          parent[stack[n]] = j
          # output[stack[n]] = "%d\t%s" % (j, relation)
          output[stack[n]] = output_str(stack[n], j, relation, ref_lines)
          stack = stack[(n + 1):]
        elif t["OP"] == 'Right-Arc':
          pos[j] = t[3]
          n = int(t[1]) - 1
          relation = t[2]
          parent[j] = stack[n]
          # output[j] = "%d\t%s" % (stack[n], relation)
          output[j] = output_str(j, stack[n], relation, ref_lines)
          buf = buf[1:]
          stack = [j] + stack[n:]
        else:
          pos[j] = t[1]
          stack = [j] + stack
          buf = buf[1:]
      elif self.transSys in [TransitionSystems.AES, TransitionSystems.AER]:
        if t["OP"] == 'Left-Arc':
          relation = t["DEPREL"]
          parent[stack[0]] = j
          # output[stack[0]] = "%d\t%s" % (j, relation)
          output[stack[0]] = output_str(stack[0], j, relation, ref_lines)
          stack = stack[1:]
        elif t["OP"] == 'Right-Arc':
          pos[j] = t["UPOSTAG"]
          relation = t["DEPREL"]
          parent[j] = stack[0]
          # output[j] = "%d\t%s" % (stack[0], relation)
          output[j] = output_str(j, stack[0], relation, ref_lines)
          buf = buf[1:]
          stack = [j] + stack
        elif t["OP"] == 'Shift':
          pos[j] = t["UPOSTAG"]
          stack = [j] + stack
          buf = buf[1:]
        else:
          stack = stack[1:]
      elif self.transSys == TransitionSystems.ASd:
        if t["OP"] == 'Left-Arc':
          relation = t[1]
          parent[stack[1]] = stack[0]
          # output[stack[1]] = "%d\t%s" % (stack[0], relation)
          output[stack[1]] = output_str(stack[1], stack[0], relation, ref_lines)
          stack = [stack[0]] + stack[2:]
        elif t["OP"] == 'Right-Arc':
          relation = t[1]
          parent[stack[0]] = stack[1]
          # output[stack[0]] = "%d\t%s" % (stack[1], relation)
          output[stack[0]] = output_str(stack[0], stack[1], relation, ref_lines)
          stack = stack[1:]
        elif t["OP"] == 'Shift':
          pos[j] = t[1]
          stack = [j] + stack
          buf = buf[1:]
      elif self.transSys == TransitionSystems.AH:
        if t["OP"] == 'Left-Arc':
          relation = t["DEPREL"]
          parent[stack[0]] = j
          # output[stack[0]] = "%d\t%s" % (j, relation)
          output[stack[0]] = output_str(stack[0], j, relation, ref_lines)
          stack = stack[1:]
        elif t["OP"] == 'Right-Arc':
          relation = t["DEPREL"]
          parent[stack[0]] = stack[1]
          # output[stack[0]] = "%d\t%s" % (stack[1], relation)
          output[stack[0]] = output_str(stack[0], stack[1], relation, ref_lines)
          stack = stack[1:]
        elif t["OP"] == 'Shift':
          pos[j] = t["UPOSTAG"]
          stack = [j] + stack
          buf = buf[1:]

    for idx, line in enumerate(output):
      if output[idx] == "":
        output[idx] = output_str(idx, -1, "_", ref_lines)

    self.file.write("%s\n\n" % ("\n".join(output[1:])))

    return ret

  def close(self):
    self.file.close()

if __name__ == "__main__":

  with open("/Users/shuoyang/Workspace/arc-swift/src/ptb3-wsj-train.conllx") as reg_file:
    conll_file = CoNLLReader(reg_file)
    for sent in conll_file:
      pass
    conll_file.close()

  with open("/Users/shuoyang/Workspace/arc-swift/src/train.AER.seq") as reg_file:
    oracle_file = OracleReader(reg_file)
    for sent in oracle_file:
      pass
    oracle_file.close()
