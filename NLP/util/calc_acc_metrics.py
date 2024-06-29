from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
import torch


def calc_acc_metrics(model, data_loader, device):
    all_labels = []
    all_predictions = []

    for batch, label in data_loader:
        batch, label = batch.to(device), label.to(device)
        outputs = model(batch)
        _, predicted_labels = torch.max(outputs, 1)
        all_labels.extend(label.cpu().numpy())
        all_predictions.extend(predicted_labels.cpu().numpy())

    f1 = f1_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    precision = precision_score(all_labels, all_predictions, average='binary')
    confusion_mat = confusion_matrix(all_labels, all_predictions)

    return f1, recall, precision, confusion_mat
