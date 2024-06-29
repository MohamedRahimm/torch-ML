import torch


def calc_loss_and_acc(model, optimizer, data_loader, criterion, device, train=True):
    if (train == True):
        model.train()
    else:
        model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for batch, label in data_loader:
        batch, label = batch.to(device), label.to(device)
        outputs = model(batch)
        loss = criterion(outputs, label)
        if (train == True):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, predicted_labels = torch.max(outputs, 1)
        correct_predictions = (predicted_labels == label).sum().item()
        total_correct += correct_predictions
        total_samples += label.size(0)
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy
