import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from utils import KANLinear, InteractiveConvolutionBlock2D  # Import classes from utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

torch.set_default_dtype(torch.float64)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check available GPUs
print("Available GPUs:")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Select the GPU
device_index = 1  # Update to use GPU index 1
if torch.cuda.device_count() > device_index:
    device = torch.device(f'cuda:{device_index}')
else:
    raise RuntimeError(f"GPU device with index {device_index} is not available.")

print(f"Using device: {device}")

class KANICE(nn.Module):
    def __init__(self, grid_size=5, spline_order=3, dropout_rate=0.25):
        super(KANICE, self).__init__()
        
        self.interactive_conv1 = InteractiveConvolutionBlock2D(1, 32)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.interactive_conv2 = InteractiveConvolutionBlock2D(64, 64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.kan1 = KANLinear(128 * 7 * 7, 256, grid_size=grid_size, spline_order=spline_order)
        self.kan2 = KANLinear(256, 10, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x):
        x = self.interactive_conv1(x)
        x = self.pool(self.bn1(self.conv1(x)))
        x = self.interactive_conv2(x)
        x = self.pool(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.kan1(x))
        x = self.kan2(x)
        return x

# Data augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the FashionMNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model_name = 'KANICE'
model = KANICE(grid_size=5, spline_order=2, dropout_rate=0.3).to(device)

print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
num_epochs = 25
scaler = GradScaler()
train_losses = []
test_losses = []
metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': []
}

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Add regularization loss
            reg_loss = 0
            for module in model.modules():
                if isinstance(module, KANLinear):
                    reg_loss += module.regularization_loss()
            
            total_loss = loss + 0.01 * reg_loss  # You can adjust the regularization strength

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_train_loss += total_loss.item()
    
    scheduler.step()
    
    epoch_train_loss = running_train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)
    
    # Evaluation
    model.eval()
    running_test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    epoch_test_loss = running_test_loss / len(test_loader)
    test_losses.append(epoch_test_loss)
    
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    metrics['accuracy'].append(accuracy)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1_score'].append(f1)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

print('Training finished.')

# Final evaluation
model.eval()
correct = 0
total = 0
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

accuracy = 100 * correct / total
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f'Final Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# Create results directory if it does not exist
os.makedirs('results', exist_ok=True)

# Save the model
torch.save(model.state_dict(), f'results/{model_name}_fashion_mnist_model.pth')

# Save the training and test loss data
np.save(f'results/{model_name}_training_losses.npy', np.array(train_losses))
np.save(f'results/{model_name}_test_losses.npy', np.array(test_losses))

# Save the evaluation metrics
np.save(f'results/{model_name}_metrics.npy', metrics)

# Plot the training and test loss
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, marker='o', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and Test Loss vs Epochs ({model_name})')
plt.legend()
plt.savefig(f'results/{model_name}_loss_plot.png')
plt.show()

# Plot the evaluation metrics
plt.figure()
plt.plot(range(1, num_epochs + 1), metrics['accuracy'], marker='o', label='Accuracy')
plt.plot(range(1, num_epochs + 1), metrics['precision'], marker='o', label='Precision')
plt.plot(range(1, num_epochs + 1), metrics['recall'], marker='o', label='Recall')
plt.plot(range(1, num_epochs + 1), metrics['f1_score'], marker='o', label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title(f'Evaluation Metrics vs Epochs ({model_name})')
plt.legend()
plt.savefig(f'results/{model_name}_metrics_plot.png')
plt.show()
