# Sequence MNIST


## Dataset

https://drive.google.com/open?id=1I8NbuUc0vF3igpCihhryVuiNd31nlwkh

## Requirements

For `seq2seq` package, refer to https://github.com/CZWin32768/pytorch-seq2seq.

```
imageio
torchtext
torch
tqdm
```

## Component

- `main.py` the entry of the code.
- `img_encoder.py` a CNN, encode the image to a vector sequence.
- `seqmnist_dataset.py` define how to load the dataset.

## TODO

- 目前只写完了模型结构和数据输入的部分，需要把训练的代码加上去
- 需要添加AutoEncoder的Loss，即重建图片（作业要求：无监督）
- 需要在解码过程中应用REINFORCE（作业要求：强化学习） https://arxiv.org/pdf/1511.06732.pdf
- 添加群智和进化算法（作业要求）

