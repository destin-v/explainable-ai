import torch
import numpy as np

# from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter

import tensorboard
import tensorflow

tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile

writer = SummaryWriter("./logs")
writer.add_image("image", np.ones((3, 3, 3)), 0)
writer.add_embedding(torch.randn(100, 5), global_step=0)
