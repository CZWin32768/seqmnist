import os
import imageio

from tqdm import tqdm
from torchtext.data import Dataset
from torchtext.data.example import Example

IMG_H, IMG_W = 28, 600

class SeqMnistExample(Example):

  @classmethod
  def fromImgFile(cls, filename):
    ex = cls()
    # image (28, 600)
    image = imageio.imread(filename)
    fn = os.path.basename(filename)
    label_str = fn[fn.find("_")+1:fn.find(".")]
    label = list(label_str)
    setattr(ex, "src", image)
    setattr(ex, "trg", label)

class SeqMnistDataset(Dataset):

  def __init__(self, path):
    examples = []
    for fn in tqdm(os.listdir(path)):
      examples.append(SeqMnistExample.fromImgFile(
        os.path.join(path, fn)
      ))
    super(SeqMnistDataset, self).__init__(examples, {})