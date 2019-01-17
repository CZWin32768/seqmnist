import torch
import torchtext
import torch.nn.init as init
import argparse


from torch.optim.lr_scheduler import StepLR

#from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
#from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

from seqmnist.img_encoder import EncoderCNN2D
from seqmnist.seqmnist_dataset import SeqMnistDataset
from seqmnist.trainer import SupervisedTrainer
from seqmnist.field import SrcField, TgtField


def build_model(tgt_field, max_len=50, hidden_size=100, bidirectional=False):
  print("building model...")
  vocab:torchtext.vocab.Vocab = tgt_field.vocab
  print("vocab: ", vocab.stoi)

  encoder = EncoderCNN2D()
  decoder = DecoderRNN(
    vocab_size=len(vocab),
    max_len=max_len,
    hidden_size=hidden_size * 2 if bidirectional else hidden_size,
    dropout_p=0.2,
    use_attention=True,
    bidirectional=bidirectional,
    eos_id=tgt_field.eos_id,
    sos_id=tgt_field.sos_id,
    rnn_cell='lstm'
  )
  model_obj = Seq2seq(encoder, decoder)
  # if torch.cuda.is_available():
  #   model_obj.cuda()
  # for param in model_obj.parameters():
  #   init.xavier_uniform(param.data)
  for param in model_obj.parameters():
    param.data.uniform_(-0.08, 0.08)
  
  return model_obj


def build_dataset(args):
  print("loading dataset...")
  src = SrcField()
  tgt = TgtField()
  train_ds = SeqMnistDataset(args.train_path, [('src', src), ('tgt', tgt)])
  dev_ds = SeqMnistDataset(args.dev_path, [('src', src), ('tgt', tgt)])
  tgt.build_vocab(train_ds)
  return train_ds, dev_ds, src, tgt


def train(args):
  train_ds, dev_ds, src_field, tgt_field = build_dataset(args)
  model = build_model(tgt_field,bidirectional=True)
  trainer = SupervisedTrainer(
    loss = Perplexity(),
    batch_size=args.batch_size,
    checkpoint_every=50,
    expt_dir=args.expt_dir,
    print_every=args.print_every
  )
  model = trainer.train(
    model=model,
    data=train_ds,
    num_epochs=args.num_epochs,
    optimizer=None,
    dev_data=dev_ds
  )


def conf():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", default=128)
  parser.add_argument(
    "--train_path", default="./multi_mnist/train")
  parser.add_argument(
    "--dev_path", default="./multi_mnist/test")
  parser.add_argument(
    "--expt_dir", default="./expt")
  parser.add_argument(
    "--print_every", default=5)
  parser.add_argument("--num_epochs", default=20)

  return parser.parse_args()


if __name__ == "__main__":
  train(conf())
