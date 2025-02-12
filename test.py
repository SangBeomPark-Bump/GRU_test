from model.my_model import MyModel

if __name__ == "__main__":


    model = MyModel(1, 20, 1)
    print(MyModel)
    # model.to(DEVICE)
    # # 옵티마이저 설정
    # optimizer = Adam(model.parameters(), lr=0.001)
    # criterion = nn.BCELoss()
    # my_train(model, NUM_EPOCHS, DEVICE, train_dataloader, test_dataloader, optimizer, criterion, is_google_colab())

    # torch.save(model.state_dict(), 'model.pt')

    print("성공!")