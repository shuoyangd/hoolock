from model.StackLSTMParser import TransitionSystems
import utils.io

import argparse
import pdb

parser = argparse.ArgumentParser()

parser.add_argument("--fin", required=True, type=str, help="Input oracle file.")
parser.add_argument("--fout", required=True, type=str, help="Output parse file.")
parser.add_argument("--transSys", default="AER", type=str, choices=['ASw', 'AER', 'AES', 'ASd', 'AH'],
                    help="Transition system to use. (default=AER)")
parser.add_argument("--conllin", default="", type=str, help="Optional conll input file.")
parser.add_argument("--text", default="", type=str, help="Optional raw texts.")

options = parser.parse_args()

writer = utils.io.Oracle2CoNLLWriter(open(options.fout, 'w'), eval("TransitionSystems." + options.transSys))
reader = utils.io.OracleReader(open(options.fin))
if options.conllin != "":
  conll_reader = utils.io.CoNLLReader(open(options.conllin))
  for oracle_sent, conll_sent in zip(reader, conll_reader):
    writer.writesent(oracle_sent, conll_sent)
  conll_reader.close()
elif options.text != "":
  text_reader = open(options.text)
  for oracle_sent, text_sent in zip(reader, text_reader):
    text_toks = text_sent.strip().split()
    writer.writesent(oracle_sent, ref_toks=text_toks)
else:
  for oracle_sent in reader:
    writer.writesent(oracle_sent)
reader.close()
writer.close()
