import torch
import torchvision
import torchvision.transforms as transforms

def main():
    # 1. Definir a transformação (converter imagem para número/tensor)
    transform = transforms.Compose([transforms.ToTensor()])

    print("Baixando o CIFAR-10... (aguarde)")

    try:
        # 2. Baixar o dataset de treino
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        
        # Adicionar download do dataset de teste se necessário
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
    except Exception as e:
        print(f"Erro ao baixar o dataset: {e}")
        return

    # 3. Informações do Dataset
    classes = ('avião', 'carro', 'pássaro', 'gato', 'cervo', 
               'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão')

    print(f"\nSucesso! O dataset de treino possui {len(trainset)} imagens.")
    print(f"O dataset de teste possui {len(testset)} imagens.")
    print(f"Classes identificadas: {classes}")

    # Testar se a GPU (NVIDIA) está disponível para o PyTorch
    print(f"\nGPU disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Usando: {torch.cuda.get_device_name(0)}")

    # Adicionar asserções básicas para teste
    assert len(trainset) == 50000, "Tamanho do dataset de treino incorreto"
    assert len(testset) == 10000, "Tamanho do dataset de teste incorreto"
    assert len(classes) == 10, "Número de classes incorreto"
    print("\nTodos os testes básicos passaram!")

if __name__ == "__main__":
    main()