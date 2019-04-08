HOOLOCK_SCRIPTS_DIR=
STANFORD_POSTAGGER_DIR=
N_THREAD=1

mkdir -p $PWD/data/postags
file_name=`basename $1`
python $HOOLOCK_SCRIPTS_DIR/conllx2plain.py $1 > $PWD/data/postags/${file_name}.plain
exec_dir=$PWD
cd $STANFORD_POSTAGGER_DIR 
./stanford-postagger.sh models/english-bidirectional-distsim.tagger ${exec_dir}/data/postags/${file_name}.plain $N_THREAD > ${exec_dir}/data/postags/${file_name}.pos 
cd ${exec_dir}
python $HOOLOCK_SCRIPTS_DIR/substituteConllxPOS.py $1 $PWD/data/postags/${file_name}.pos > $PWD/data/postags/${file_name}
