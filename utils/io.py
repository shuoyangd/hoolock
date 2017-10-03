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
    sent = self.readline()
    if sent == []:
      raise StopIteration()
    else:
      return sent

  def readline(self):
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
    sent = self.readline()
    if sent == []:
      raise StopIteration()
    else:
      return sent

  def readline(self):
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
      else:
        row["UPOSTAG"] = columns[1]
        row["XPOSTAG"] = columns[2]
      sent.append(row)
      row_str = self.file.readline().strip()
    return sent

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
