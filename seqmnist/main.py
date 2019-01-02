import torch
import torchtext
import argparse


from torch.optim.lr_scheduler import StepLR

from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

from seqmnist.img_encoder import EncoderCNN2D
from seqmnist.seqmnist_dataset import SeqMnistDataset


def build_model(max_len=50, hidden_size=200, bidirectional=True):
  print("building model...")
  vocab = {str(i):i for i in range(10)}
  vocab["<EOS>"] = 10
  vocab["<BOS>"] = 11

  encoder = EncoderCNN2D()
  decoder = DecoderRNN(
    vocab_size=len(vocab),
    max_len=max_len,
    hidden_size=hidden_size * 2 if bidirectional else hidden_size,
    dropout_p=0.2,
    use_attention=True,
    bidirectional=bidirectional,
    eos_id=vocab["<EOS>"],
    sos_id=vocab["<BOS>"]
  )
  model_obj = Seq2seq(encoder, decoder)
  if torch.cuda.is_available():
    model_obj.cuda()
  
  for param in model_obj.parameters():
    param.data.uniform_(-0.08, 0.08)
  
  return model_obj, vocab


def build_dataset(args):
  print("loading dataset...")
  train_ds = SeqMnistDataset(args.train_path)
  dev_ds = SeqMnistDataset(args.dev_path)
  return train_ds, dev_ds


def train(args):
  train_ds, dev_ds = build_dataset(args)
  model, vocab = build_model()
  trainer = SupervisedTrainer(
    loss = Perplexity(),
    batch_size=args.batch_size,
    checkpoint_every=50,
    print_every=10,
    expt_dir=args.expt_dir
  )
  model = trainer.train(
    model=model,
    data=train_ds,
    num_epochs=args.num_epochs,
    optimizer=None,
  )

def conf():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", default=32)
  parser.add_argument(
    "--train_path", default="/home/czwin32768/res/mnist/seq-mnist/train")
  parser.add_argument(
    "--dev_path", default="/home/czwin32768/res/mnist/seq-mnist/test")
  parser.add_argument(
    "--expt_dir", default="./expt")
  parser.add_argument("--num_epochs", default=6)
  return parser.parse_args()


if __name__ == "__main__":
  train(conf())
