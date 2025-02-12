import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BertTokenizer


INPUT_SIZE = 1
HIDDEN_SIZE = 20
OUTPUT_SIZE = 1
NUM_EPOCHS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
