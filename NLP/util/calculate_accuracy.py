import torch


def calculate_accuracy(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch, label in data_loader:
            batch, label = batch.to(device), label.to(device)
            outputs = model(batch)
            loss = criterion(outputs, label)
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions = (predicted_labels == label).sum().item()
            total_correct += correct_predictions
            total_samples += label.size(0)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy
