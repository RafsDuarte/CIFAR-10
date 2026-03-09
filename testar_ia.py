import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. Configurar dispositivo e classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('avião', 'carro', 'pássaro', 'gato', 'cervo', 'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão')

# 2. Carregar dados de teste
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True)

# 3. Recriar a estrutura da rede (Exatamente como no treino)
class AdvancedCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 512)
        self.fc2 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 4. Carregar os pesos que você treinou na sua GPU
net = AdvancedCNN().to(device)
net.load_state_dict(torch.load('./cifar_model.pth'))
net.eval() # Modo de avaliação

# 5. Pegar algumas imagens e testar
dataiter = iter(testloader)
images, labels = next(dataiter)

# Mover para GPU e processar
outputs = net(images.to(device))
_, predicted = torch.max(outputs, 1)

# Mostrar resultados
print('Real: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
print('IA:   ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

# Visualizar
def imshow(img):
    img = img / 2 + 0.5     # desnormalizar
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

imshow(torchvision.utils.make_grid(images))