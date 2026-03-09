# Classificador CIFAR-10 com PyTorch

Um projeto de classificação de imagens usando redes neurais convolucionais (CNN) para classificar imagens do dataset CIFAR-10 em 10 categorias. Implementado para fins educacionais enquanto estou estudando o assunto.

## 🎯 Objetivo

Treinar um modelo de deep learning capaz de classificar automaticamente imagens em uma das 10 classes:

- Avião, Carro, Pássaro, Gato, Cervo, Cachorro, Sapo, Cavalo, Navio, Caminhão

## 🏗️ Arquitetura do Modelo

O modelo utiliza uma **CNN (Convolutional Neural Network) avançada** com as seguintes camadas:

```
AdvancedCNN:
├── Conv2d(3→32, 3x3) + BatchNorm2d + ReLU + MaxPool2d
├── Conv2d(32→64, 3x3) + BatchNorm2d + ReLU + MaxPool2d
├── Dropout(0.3)
├── Flatten
├── Linear(64*8*8 → 512) + ReLU
└── Linear(512 → 10)
```

**Técnicas de otimização implementadas:**

- ✅ Batch Normalization (acelera convergência)
- ✅ Dropout (previne overfitting)
- ✅ Data Augmentation (flip horizontal, crop aleatório)
- ✅ Weight Decay (L2 regularization)
- ✅ Learning Rate Scheduler (reduz LR ao longo do treinamento)

## 📋 Requisitos

- Python 3.8+
- PyTorch
- Torchvision
- NumPy
- Matplotlib

## 🚀 Como Usar

### 1. Preparar o Ambiente

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Instalar dependências
pip install torch torchvision numpy matplotlib
```

### 2. Treinar o Modelo

```bash
python treinar_cifar.py
```

**O que acontece:**

- Baixa automaticamente o dataset CIFAR-10
- Treina o modelo por 50 épocas
- Aplica data augmentation nos dados
- Salva o modelo treinado em `cifar_model.pth`
- Exibe perda a cada 500 batches

### 3. Testar o Modelo

```bash
python testar_ia.py
```

**O que acontece:**

- Carrega o modelo treinado
- Testa em 4 imagens aleatórias do dataset de teste
- Exibe a classificação real vs predição da IA
- Mostra as imagens em uma janela gráfica

## 📁 Estrutura do Projeto

```
CIFAR-10/
├── treinar_cifar.py       # Script de treinamento
├── testar_ia.py           # Script de teste/inferência
├── test_cifar.py          # (Opcional) Testes adicionais
├── cifar_model.pth        # Pesos do modelo treinado (gerado após treinamento)
├── data/                  # Dataset CIFAR-10 (gerado automaticamente)
│   └── cifar-10-batches-py/
├── venv/                  # Ambiente virtual (ignorado no git)
├── .gitignore             # Arquivo de exclusão do git
└── README.md             # Este arquivo
```

## 💡 Dicas de Uso

**Adicionar mais épocas:**
Edite `treinar_cifar.py` e altere a linha:

```python
for epoch in range(50):  # Mude para maior número
```

**Usar CPU se não tiver GPU:**
O código detecta automaticamente. Se quiser forçar:

```python
device = torch.device("cpu")  # No início do arquivo
```

**Entender os dados:**

```python
import torchvision.datasets as datasets
# Cada imagem: 32x32 pixels, 3 canais (RGB)
# Total: 50,000 imagens de treino, 10,000 de teste
```

## 📚 Referências

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Documentation](https://pytorch.org/)
- [CNN Basics](https://cs231n.github.io/convolutional-networks/)

## 📝 Licença

Este projeto é de código aberto e livre para uso educacional.

---

**Autor:** Rafael Duarte  
**Data:** Março 2026  
**Status:** ✅ Funcional e Testado
