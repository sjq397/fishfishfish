import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from datareader import HeartSoundsDataset
from transformer_base import CNNTransformer
from tqdm import tqdm
from cnn import Conv1DModel

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 2. 数据加载和划分
def split_dataset(dataset, test_size=0.2):
    """
    将数据集划分为训练集和验证集。
    """
    train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=test_size, random_state=42)
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    return train_set, val_set

csv_file_path = "D:\\sjq\\heart_sounds\\code\\data\\clinical_data_expanded.csv"
hdf5_file_path = "D:\\sjq\\heart_sounds\\Clinical Trials\\clinical_study_2024_dataset.hdf5"
dataset = HeartSoundsDataset(csv_path=csv_file_path, hdf5_path=hdf5_file_path)

train_set, val_set = split_dataset(dataset)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

# 3. 定义模型
# model = CNNTransformer(input_dim=5000, num_classes=2).to(device)
model =  Conv1DModel(num_classes=2, audio_length=5000).to(device)
# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 二分类任务
# optimizer = optim.Adam(model.parameters(), lr=1e-5)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# 5. 训练和验证函数
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    loop = tqdm(loader, desc="Training", leave=False)
    for signals, labels in loop:
        signals, labels = signals.to(device).float(), labels.to(device)  # 转换为 float32 类型
        signals = signals.squeeze(-1)
        signals = signals.unsqueeze(1)
        # print(f"signals shape: {signals.shape}")
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        
       
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        loop = tqdm(loader, desc="Validating", leave=False)
        for signals, labels in loop:
            signals, labels = signals.to(device).float(), labels.to(device)  # 转换为 float32 类型
            signals = signals.squeeze(-1)
            signals = signals.unsqueeze(1)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            

            loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

# 6. 训练过程
num_epochs = 20
best_val_acc = 0.0


epoch_progress = tqdm(range(num_epochs), desc="Epochs")
for epoch in epoch_progress:
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
    

    epoch_progress.set_postfix(train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
    
    print(f"\nEpoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_cnn_transformer_model.pth")
        print("  Best model saved!")

print("Training complete!")

# 7. 加载和测试最佳模型
model.load_state_dict(torch.load("best_cnn_transformer_model.pth"))
model.eval()
print("Loaded the best model for testing.")
