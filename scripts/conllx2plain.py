import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import utils.io

if len(sys.argv) < 2:
  sys.stderr.write("usage: python conllx2plain.py conllx_file\n")
  sys.exit(1)

conll_reader = utils.io.CoNLLReader(open(sys.argv[1]))
for sent in conll_reader:
  tokens = []
  for row in sent:
    tokens.append(row["FORM"])
  sys.stdout.write(" ".join(tokens) + "\n")
conll_reader.close()
