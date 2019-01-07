import torch
import numpy as np

from seq2seq.dataset.fields import SourceField, TargetField


class SrcField(SourceField):

  def __init__(self, **kwargs):
    super(SrcField, self).__init__(**kwargs)

  def process(self, batch, device=None):
    """ Process a list of examples to create a torch.Tensor.

    (batch, 1, 28, 600)

    Args:
        batch (list(object)): A list of object from a batch of examples.
    Returns:
        torch.autograd.Variable: Processed object given the input
        and custom postprocessing Pipeline.
    """
    mat = np.stack(batch)
    ret = torch.FloatTensor(mat) / 255.0
    ret = ret.unsqueeze(1)
    assert ret.shape[1:] == (1, 28, 600)
    return ret


class TgtField(TargetField):

  def __init__(self, **kwargs):
    super(TgtField, self).__init__(**kwargs)