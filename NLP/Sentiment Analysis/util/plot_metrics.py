import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_metrics(train_losses, valid_losses, train_accs, valid_accs, confusion_mat):
    epochs = range(1, len(train_losses) + 1)

    # Plot train and validation losses
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, valid_losses, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot train and validation accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accs, 'b', label='Training accuracy')
    plt.plot(epochs, valid_accs, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot confusion matrix
    plt.subplot(1, 3, 3)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=[
                                  'Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), colorbar=False)
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.show()
