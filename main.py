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

from seqmnist.img_encoder import EncoderCNN2D, CustomedEncoderCNN2D
from seqmnist.seqmnist_dataset import SeqMnistDataset
from seqmnist.trainer import SupervisedTrainer
from seqmnist.field import SrcField, TgtField
from PSO_Opt.PSO_Opt import pso

def build_model(featruemap_nums, kernel_sizes, tgt_field, max_len=50, hidden_size=175, bidirectional=False):
  print("building model...")
  vocab:torchtext.vocab.Vocab = tgt_field.vocab
  print("vocab: ", vocab.stoi)

  encoder = CustomedEncoderCNN2D(featruemap_nums=featruemap_nums,kernel_sizes=kernel_sizes)
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


def build_dataset(train_path, dev_path):
  print("loading dataset...")
  src = SrcField()
  tgt = TgtField()
  train_ds = SeqMnistDataset(train_path, [('src', src), ('tgt', tgt)])
  dev_ds = SeqMnistDataset(dev_path, [('src', src), ('tgt', tgt)])
  tgt.build_vocab(train_ds)
  return train_ds, dev_ds, src, tgt


def train(args):
  train_ds, dev_ds, src_field, tgt_field = build_dataset(args.train_path, args.dev_path)
  model = build_model([22,76],[5,7],tgt_field,hidden_size = 76*7,bidirectional=False)
  trainer = SupervisedTrainer(
    loss = Perplexity(),
    batch_size=args.batch_size,
    checkpoint_every=50,
    expt_dir=args.expt_dir,
    print_every=args.print_every
  )
  torch.cuda.set_device(0)
  model = trainer.train(
    model=model.cuda(),
    data=train_ds,
    num_epochs=args.num_epochs,
    optimizer=None,
    dev_data=dev_ds,
    device=0
  )
  torch.save(model.state_dict(), '/home/mcis105/yuhongfei/AI_Course/seqmnist/models/best_model')

def caculateLoss(x, *args):
  train_path, dev_path, batch_size, num_epochs, print_every = args
  train_ds, dev_ds, src_field, tgt_field = build_dataset(train_path=train_path,dev_path=dev_path)

  ##get feature_nums and kernel_sizes
  feature_nums = []
  feature_nums.append(int(round(x[0])))
  feature_nums.append(int(round(x[1])))

  kernel_sizes = []
  kernel_sizes.append(x[2])
  kernel_sizes.append(x[3])
  #print(kernel_sizes)
  for i in range(2):
    if int(kernel_sizes[i])%2 != 0:
      kernel_sizes[i] = int(kernel_sizes[i])
    else:
      kernel_sizes[i] = int(kernel_sizes[i]) + 1
  #print(feature_nums[0])
  #print(kernel_sizes)
  #nvmlInit()
  #handle = nvmlDeviceGetHandleByIndex(0)
  #info = nvmlDeviceGetMemoryInfo(handle)
  device = int(x[-1])
  model = build_model(feature_nums,kernel_sizes,tgt_field,hidden_size=feature_nums[1]*7,bidirectional=False).cuda(device)
  trainer = SupervisedTrainer(
    loss = Perplexity(),
    batch_size=batch_size,
    checkpoint_every=50,
    print_every=print_every
  )
  acc = trainer.train(
    model=model,
    data=train_ds,
    num_epochs=num_epochs,
    optimizer=None,
    dev_data=dev_ds,
    device=device
  )
  torch.save(model.state_dict(), '/home/mcis105/yuhongfei/AI_Course/seqmnist/models/' + str(x))
  del trainer
  del model
  del train_ds, dev_ds, src_field, tgt_field
  return acc


def PSOtrain(args):
  lb = [20, 50, 2, 2]
  ub = [50, 80, 8, 8]
  vmax = [10, 10, 1, 1]

  xopt1, fopt1 = pso(caculateLoss,(args.train_path, args.dev_path, args.batch_size, args.num_epochs, args.print_every),
      lb,ub,vmax,swarmsize=8,omega=0.792,phip=1.494,phig=1.494,maxiter=30,
      minstep=0.001,debug=True,processes=args.num_workers)
  print(xopt1)
  print('max_acc = ' + str(fopt1))

def conf():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", default=90)
  parser.add_argument(
    "--train_path", default="./data/multi_mnist/train")
  parser.add_argument(
    "--dev_path", default="./data/multi_mnist/test")
  parser.add_argument(
    "--expt_dir", default="./expt")
  parser.add_argument(
    "--print_every", default=50)
  parser.add_argument("--num_epochs", default=15)
  parser.add_argument("--num_workers", default=1)

  return parser.parse_args()


if __name__ == "__main__":
  #train(conf())
  PSOtrain(conf())
