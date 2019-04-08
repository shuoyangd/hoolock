# hoolock

hoolock is a Pytorch-based, GPU-friendly StackLSTM implementation that makes it much easier and faster to build a StackLSTM parser. To give you an idea, for Penn Treebank, the training takes about 30 minutes with batch size 256 on a single GTX-1080Ti GPU.

### Dependencies

You can install all python dependencies by calling `pip install -r requirements.txt`. For others, you only have to download them.

- Python 3.6
- [dill](https://pypi.org/project/dill)
- [PyTorch v1.0.0](https://pytorch.org): any version after v0.4 should work, but we enforce version to eliminate randomness
- [pytorch-gradual-warmup-lr](https://github.com/shuoyangd/pytorch-gradual-warmup-lr)
- [six](https://pypi.org/project/six/)
- [torchtext v0.2.1](https://github.com/pytorch/text/releases/tag/v0.2.1): later versions may also work, but not tested
- [arc-swift](https://github.com/qipeng/arc-swift): only for data pre and post-processing
- [Stanford POS Tagger](https://nlp.stanford.edu/software/tagger.shtml): optional, just for reproducing the results in the paper


*NOTE*: Unfortunately, [arc-swift](https://github.com/qipeng/arc-swift) seems to assume Python 2. So you might want to set up some conda/virtualenv environments or run alternative PTB data preparation script.

### Usage

This short tutorial aims to help you reproduce result in the paper, which is on Penn Treebank (PTB), but the procedure to training and testing on other data should be similar.

#### Preprocessing

Follow the instructions in [arc-swift](https://github.com/qipeng/arc-swift) to run the initial preprocessing on Penn Treebank. Their script will create standard data split on PTB, filter non-projective trees in training data, and generate oracle transition sequence. We will assume your preprocessed data (preprocessed conllx and oracle sequences) are all stored in a single directory referred to as `${data_dir}`.

Setup [Stanford POS Tagger](https://nlp.stanford.edu/software/tagger.shtml) and run `scripts/postag.sh ${data_dir}/ptb3-wsj-[train|dev|dev.proj|test].conllx` to generate data with Stanford POSTags. Your data is going to be stored in `$PWD/data/postags`

You'll also need a special word embedding used in Dyer et al. 2015 which you can download [here](https://drive.google.com/file/d/0B8nESzOdPhLsdWF2S1Ayb1RkTXc/view?usp=sharing).

The last step is to build vocabulary, integerize everything and dump them as a binary object:

```
mkdir -p binarized
python preprocess.py --train_conll_file $PWD/data/postags/ptb3-wsj-train.conllx.pos --train_oracle_file ${data_dir}/data/train.AH.seq --dev_conll_file $PWD/data/postags/ptb3-wsj-dev.proj.conllx.pos --dev_oracle_file ${data_dir}/data/dev.AH.seq --save_data $PWD/binarized/data+pre+inferpos.en.AH --pre_word_emb_file ${data_dir}/sskip.100.vectors --sent_len 150 --action_seq_len 300
```

#### Training

You can explore all the available options by running `python train.py --help`, but to reproduce the paper the default is sufficient:

```
python train.py --data_file $PWD/binarized/data+pre+inferpos.en.AH --model_file model --gpuid [gpu id]
```

#### Parsing

Here is what we use to generate our output. You are free to adjust the batch size as you wish, which does not change the output. The stack size, on the other hand, may need to be changed from corpus to corpus, although it doesn't need to be the same as the one you used for training as it doesn't change the number of parameters. We find 150 to be enough for PTB.

```
python parse.py --input_file $PWD/data/postags/[conllx input]  --model_file [model file] --output_file out.seq --batch_size 80 --stack_size 150 --data_file $PWD/binarized/data+pre+inferpos.en.AH --pre_emb_file ${data_dir}/sskip.100.vectors
```

The parser will output a transition sequence, which is not very helpful. To convert this back to conllx format, run the following script:

```
python oracle2conll.py --fin out.seq --fout out.conllx --transSys AH --conllin $PWD/data/postags/[conllx input]
```

Finally, to evaluate arc F1 score, run the script from [arc-swift](https://github.com/qipeng/arc-swift) (remember that you need to switch back to Python 2).

```
python $arc_swift/src/eval.py -g [reference] -s out.conllx
```

Note that while the preprocesed conllx input will have the parse of sentence, we are not looking at the parse during parsing and postprocessing. If you are parsing plain text file, you'll want to convert them into conllx format and put a placeholder for these fields.

### Citation

TBD

### Naming

According to [Wikipedia](https://en.wikipedia.org/wiki/Hoolock_gibbon), hoolock gibbons are generally found in Eastern Bangladesh, Northeast India and Southwest China. Benefiting from their special brachiating skills, they can travel up to 35mph between the **trees**, making them the **fastest** and **most agile** of all tree-dwelling, non-flying mammals.

Because of shrinking habitat and hunting, most species of the hoolock genus have been classified by IUCN as "Endangered" or "Vulnerable" (see [here](https://www.iucnredlist.org/species/39876/10278553) and [here](https://www.iucnredlist.org/species/118355453/17968300)). You can find more information about this [IUCN classification](https://www.iucnredlist.org) and [donate to them](https://www.iucn.org/donate) to help combat the declining global biodiversity.
