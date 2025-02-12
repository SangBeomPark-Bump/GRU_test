
import torch

def my_train(model, num_epochs, device, train_dataloader, test_dataloader, optimizer, criterion, is_google_colab):
    if is_google_colab:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False):
            input_ids = batch['input_ids'].unsqueeze(dim=-1).to(device)
            labels = batch['labels'].float().to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)[0]
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
        
        # 평가
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].unsqueeze(dim = -1).to(device)
                labels = batch['labels'].float().to(device)
                
                outputs = model(input_ids)[0]
                predicted = (outputs.squeeze() > 0.5).long()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.4f}")