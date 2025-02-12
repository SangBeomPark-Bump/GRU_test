import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import BertTokenizer
from torch.optim import Adam


from model.gru_cell import MyGRUCell
from model.my_model import MyModel
from train import my_train
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, DEVICE, NUM_EPOCHS
from data_loader import train_dataloader, test_dataloader

def is_google_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

if __name__ == "__main__":

    if is_google_colab():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    model = MyModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    model.to(DEVICE)
    # 옵티마이저 설정
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    my_train(model, NUM_EPOCHS, DEVICE, train_dataloader, test_dataloader, optimizer, criterion, is_google_colab())

    torch.save(model.state_dict(), 'model.pt')

    print("성공!")