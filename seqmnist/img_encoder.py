from torch import nn
from torch.nn import functional as F

class EncoderCNN2D(nn.Module):
  """
  Encode input image tensor to a code sequence.
  """

  def __init__(self):
    super(EncoderCNN2D, self).__init__()

    self.conv1 = nn.Conv2d(
      in_channels=1,
      out_channels=20,
      kernel_size=5,
      stride=1)
    self.conv2 = nn.Conv2d(
      in_channels=20,
      out_channels=50,
      kernel_size=5,
      stride=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.pool2seq = nn.MaxPool1d(4, 4)

  
  def forward(self, input_var, input_lengths=None):
    """
    Applies a multi-layer CNN to an input sequence.

    Args:
        input_var (batch, channel, height, width): the input image.

    Returns: output, hidden
        - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
    """

    # (batch, 1, 28, 600) -> (batch, 50, 4, 147)
    x = input_var
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)


    # (batch, 50, 4, 147) -> (batch, 200, 147)
    batch_size, channel_num, height, width = x.shape
    hidden_size = channel_num * height
    x = x.view(batch_size, hidden_size, width)

    # (batch, 200, 147) -> (batch, 200, 36)
    x = self.pool2seq(x)

    # (batch, 200, 36) -> (batch, 36, 200)
    output = x.transpose(1,2)
    hidden = None
    return output, hidden

