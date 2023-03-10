"""WARNING:  Must display using Chrome!"""

import torch
import numpy as np
from tensorboardX import SummaryWriter

writer = SummaryWriter()
writer.add_image("image", np.ones((3, 3, 3)), 0)
writer.add_embedding(torch.randn(100, 5), global_step=0)
