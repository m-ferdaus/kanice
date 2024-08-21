
#### kanice-mini
    
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from utils import InteractiveConvolutionBlock2D  # Import classes from utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import math

torch.set_default_dtype(torch.float64)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check available GPUs
print("Available GPUs:")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Select the GPU
device_index = 2  # Update to use GPU index 1
if torch.cuda.device_count() > device_index:
    device = torch.device(f'cuda:{device_index}')
else:
    raise RuntimeError(f"GPU device with index {device_index} is not available.")

print(f"Using device: {device}")

class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, groups):
        super(GroupedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.group_in_features = in_features // groups
        self.group_out_features = out_features // groups

        self.weight = nn.Parameter(torch.Tensor(groups, self.group_out_features, self.group_in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1/math.sqrt(self.out_features), 1/math.sqrt(self.out_features))

    def forward(self, x):
        x = x.view(x.size(0), self.groups, self.group_in_features)
        out = torch.einsum('bgc,goc->bog', x, self.weight)
        return out.view(x.size(0), -1) + self.bias

class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        share_spline_weights=False,
        groups=1
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.share_spline_weights = share_spline_weights
        self.groups = groups

        self.base_weight = nn.Linear(in_features, out_features)
        
        spline_features = grid_size + spline_order
        if share_spline_weights:
            self.spline_weight = nn.Parameter(torch.Tensor(out_features, spline_features))
        else:
            self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, spline_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight.weight, a=math.sqrt(5))
        if self.base_weight.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_weight.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.base_weight.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

    def forward(self, x):
        base_output = self.base_weight(x)
        
        # Simple spline computation
        spline_input = torch.linspace(0, 1, self.grid_size + self.spline_order, device=x.device)
        if self.share_spline_weights:
            spline_output = F.linear(spline_input.repeat(x.size(0), 1), self.spline_weight)
        else:
            spline_output = torch.sum(F.linear(spline_input.repeat(x.size(0), self.in_features, 1), 
                                               self.spline_weight.view(-1, self.grid_size + self.spline_order)) 
                                      * x.unsqueeze(-1), dim=1)
        
        return base_output + spline_output

    def regularization_loss(self):
        return torch.sum(torch.abs(self.spline_weight))
    
    
    
class kanice_mini(nn.Module):
    def __init__(self, grid_size=5, spline_order=3, dropout_rate=0.25, share_spline_weights=False, groups=1):
        super(kanice_mini, self).__init__()
        
        self.interactive_conv1 = InteractiveConvolutionBlock2D(1, 32)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.interactive_conv2 = InteractiveConvolutionBlock2D(64, 64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate the output size after convolutions and pooling
        self.feature_size = self._get_conv_output((1, 28, 28))
        
        self.kan1 = KANLinear(self.feature_size, 256, grid_size=grid_size, spline_order=spline_order, share_spline_weights=share_spline_weights, groups=groups)
        self.kan2 = KANLinear(256, 10, grid_size=grid_size, spline_order=spline_order, share_spline_weights=share_spline_weights, groups=groups)

    def _get_conv_output(self, shape):
        input = torch.rand(1, *shape)
        output = self.interactive_conv1(input)
        output = self.pool(self.bn1(self.conv1(output)))
        output = self.interactive_conv2(output)
        output = self.pool(self.bn2(self.conv2(output)))
        return int(torch.prod(torch.tensor(output.size())))

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
model_name = 'kanice_mini'
model = kanice_mini(grid_size=5, spline_order=3, dropout_rate=0.3, share_spline_weights=True, groups=4).to(device)

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
