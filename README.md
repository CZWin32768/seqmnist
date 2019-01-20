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
dill
six
torchvision
```

## Component

- `main.py` the entry of the code.
- `img_encoder.py` a CNN, encode the image to a vector sequence.
- `seqmnist_dataset.py` define how to load the dataset.


## Usage

- Install the required packages. `pip install -r requirements.txt`
- Install the seqmnist package. `pip install .`
- Prepare the seq-mnist dataset.
- Modify the conf in main.py
- `python seqmnist/main.py`

效果如下：

```
(base) [czwin32768@czwsxiaobawang seq-mnist]$ python seqmnist/main.py 
/home/czwin32768/prog/miniconda3/lib/python3.7/site-packages/torch/nn/functional.py:52: UserWarning: size_average and reduce args will be deprecated, please use reduction='elementwise_mean' instead.
  warnings.warn(warning.format(ret))
/home/czwin32768/prog/miniconda3/lib/python3.7/site-packages/torch/nn/functional.py:52: UserWarning: size_average and reduce args will be deprecated, please use reduction='elementwise_mean' instead.
  warnings.warn(warning.format(ret))
loading dataset...
100%|██████████████████████████████████████████████████████| 50000/50000 [00:16<00:00, 3122.44it/s]
100%|████████████████████████████████████████████████████████| 2000/2000 [00:00<00:00, 3130.70it/s]
building model...
vocab:  defaultdict(<function _default_unk_index at 0x7f166deed598>, {'<unk>': 0, '<pad>': 1, '7': 2, '5': 3, '0': 4, '1': 5, '9': 6, '2': 7, '3': 8, '6': 9, '8': 10, '4': 11, '<sos>': 0, '<eos>': 0})
/home/czwin32768/prog/miniconda3/lib/python3.7/site-packages/torch/nn/modules/rnn.py:38: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1
  "num_layers={}".format(dropout, num_layers))
/home/czwin32768/prog/miniconda3/lib/python3.7/site-packages/torch/nn/functional.py:52: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))
The `device` argument should be set by using `torch.device` or passing a string as an argument. This behavior will be deprecated soon and currently defaults to cpu.
Epoch: 1, Step: 0
/home/czwin32768/prog/miniconda3/lib/python3.7/site-packages/torch/nn/functional.py:995: UserWarning: nn.functional.tanh is deprecated. Use torch.tanh instead.
  warnings.warn("nn.functional.tanh is deprecated. Use torch.tanh instead.")
Progress: 0%, Train Perplexity: 20.5263
Progress: 0%, Train Perplexity: 7.9059
Progress: 0%, Train Perplexity: 7.8818
Progress: 0%, Train Perplexity: 7.2626
Progress: 0%, Train Perplexity: 6.2033
Progress: 0%, Train Perplexity: 7.0596
Progress: 0%, Train Perplexity: 6.7849
Progress: 0%, Train Perplexity: 5.7193
Progress: 0%, Train Perplexity: 6.1770
Progress: 0%, Train Perplexity: 5.7955

```


## 日志与TODO

12月21日：

- 使用VAE生成了额外的数据（YHF 作业要求：无监督学习）
- 造了seq-mnist数据集（CZW）

1月2日更新：

- 写完了seq2seq的模型代码, encoder使用两层CNN，再通过pooling转化为序列，decoder还是用的GRU （CZW）
- 数据输入的部分（Dataset类），定义了seqmnist_dataset.SeqMnistDataset 和 seqmnist_dataset.SeqMnistExample 类型 （CZW）
- 需要在解码过程中应用REINFORCE（作业要求：强化学习） https://arxiv.org/pdf/1511.06732.pdf 
- 添加群智和进化算法（作业要求） （TODO）

1月7日更新：

- 重写了field类（field.py）以适应图片类型的输入（因为seq2seq库是建立在torchtext基础上的，因此对mnist这种图片输入无能为力，这里我通过重写Filed类process函数来实现非文本类型预处理）（CZW）
- 重写了trainer (trainer.py)，因为原有的库的trainer也仅仅针对文本序列写的，这个针对图片输入稍作改动 （CZW）
- 经过调试与debug，已经可以在cpu上运行，但是GPU上还没有尝试过（CZW）
- 需要写模型的 Evaluation ，现在只能显示困惑度，但是这个指标不够清晰(TODO)

1月15日更新：

- 解码部分添加了policy_gradient部分代码，实现了强化学习部分（ZHY）
- 修改了seq2seq的evaluate代码，现在可以得到准确率（ZHY）
- 目前模型准确率为25%左右

1月17日更新：

- encoder最后一层改为了全连接（ZHY）
- decoder改为双向LSTM（ZHY）
- 目前训练20个epoch模型准确率为81%

1月18日更新：

- 更改了encoder使其能够自定义卷积核数量以及大小以方便PSO算法来更新（YHF）

- 添加了使用PSO来优化encoder网络卷积核数量以及大小的代码（YHF）

  注：本分支，该代码为GPU代码，如需使用CPU运行请把相关cuda代码注释掉，请运行根目录下的main.py，不要运行seqminist目录下的mian.py

  运行方法 `python main.py --batch_size <batch_size> --train_path <训练集路径> --dev_path <测试集路径> --num_epochs <每个模型训练的epoch数> --num_workers <并行运行的进程数量>`

- 测试了原网络训练30个epoch的准确率为84%，其encoder的两层卷积核的数量及大小为[20,50],[5,5]

1月20日更新：

- 使用8个粒子，迭代10次得到的最优的模型参数为[22,76],[5,7]，改模型训练23个epoch后准确率为91%，有明显提升

##Author

- Zewen Chi
- Hongyu Zang
- Hongfei Yu