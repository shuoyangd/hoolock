import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import utils.io

if len(sys.argv) < 3:
  sys.stderr.write("usage: python substituteConllxPOS.py conllx_file pos_file\n")
  sys.exit(1)

def write_row(row):
  sys.stdout.write(str(row["ID"]) + "\t")
  sys.stdout.write(row["FORM"] + "\t")
  sys.stdout.write(row["LEMMA"] + "\t")
  sys.stdout.write(row["UPOSTAG"] + "\t")
  sys.stdout.write(row["XPOSTAG"] + "\t")
  sys.stdout.write(row["FEATS"] + "\t")
  sys.stdout.write(str(row["HEAD"]) + "\t")
  sys.stdout.write(row["DEPREL"] + "\t")
  sys.stdout.write(row["DEPS"] + "\t")
  sys.stdout.write(row["MISC"] + "\n")


conll_reader = utils.io.CoNLLReader(open(sys.argv[1]))
pos_reader = open(sys.argv[2])
for conll_sent, pos_sent in zip(conll_reader, pos_reader):
  tokens = pos_sent.strip().split(" ")
  postags = [ token.split("_")[1] if "_" in token else "" for token in tokens ]
  for conll_row, pos_row in zip(conll_sent, postags):
    if pos_row != "":
      conll_row["UPOSTAG"] = pos_row
      conll_row["XPOSTAG"] = pos_row
    write_row(conll_row)
  sys.stdout.write("\n")
pos_reader.close()
conll_reader.close()
