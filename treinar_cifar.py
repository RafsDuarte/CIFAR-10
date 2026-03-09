import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. Configurar para usar a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Treinando na: {torch.cuda.get_device_name(0)}")

# 2. Preparar os dados (Normalização ajuda a IA a aprender mais rápido)
# Data Augmentation para melhorar performance
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Virar imagens horizontalmente
    transforms.RandomCrop(32, padding=4),  # Crop aleatório com padding
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# 3. Definir a arquitetura da Rede Neural (Avançada para o CIFAR)
class AdvancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3) # 30% de chance de ignorar neurônios (evita decoreba)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = AdvancedCNN().to(device)

# 4. Função de Perda (Loss) e Otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)  # Weight decay para regularização
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduzir LR a cada 10 épocas

# 5. Loop de Treinamento (Aumentado para 50 épocas para melhor performance)
print("\nIniciando Treinamento...")
for epoch in range(50): 
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device) # Joga para a GPU

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 500 == 499:
            print(f'Época {epoch + 1} | Progresso: {i + 1}/1563 | Perda: {running_loss / 500:.3f}')
            running_loss = 0.0
    
    scheduler.step()  # Atualizar learning rate

print("\nTreinamento concluído!")
torch.save(net.state_dict(), './cifar_model.pth')
print("Modelo salvo como cifar_model.pth")