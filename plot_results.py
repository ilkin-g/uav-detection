import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_training_history(csv_file='training_history.csv'):
    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []

    print(f"Reading data from {csv_file}...")
    try:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                epochs.append(int(row[0]))
                train_loss.append(float(row[1]))
                train_acc.append(float(row[2]))
                val_loss.append(float(row[3]))
                val_acc.append(float(row[4]))
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Run the training pipeline first.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, label='Train Loss', color='blue', marker='o')
    ax1.plot(epochs, val_loss, label='Validation Loss', color='red', marker='x')
    ax1.set_title('Model Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, train_acc, label='Train Accuracy', color='blue', marker='o')
    ax2.plot(epochs, val_acc, label='Validation Accuracy', color='red', marker='x')
    ax2.set_title('Model Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.axhline(y=33.3, color='gray', linestyle='--', label='Random Guessing (33%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('training_graphs.png', dpi=300)
    print("Graphs successfully saved as 'training_graphs.png'.")
    plt.show()

if __name__ == '__main__':
    plot_training_history()