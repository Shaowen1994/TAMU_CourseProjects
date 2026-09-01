import Model_helper

import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

import math
import time

from torchtext.data import BucketIterator

########################################################### Load the Data ############################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (torch.ones([100,20,20]), torch.ones([100,20,20]), torch.ones([100,20,20])),
    batch_size = BATCH_SIZE,
    device = device)

