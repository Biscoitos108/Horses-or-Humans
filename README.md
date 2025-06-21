# 🐎🤖 Classificador de Imagens: Cavalo ou Humano com MobileNetV2

Este projeto implementa um modelo de deep learning para **classificar imagens entre cavalos e humanos** utilizando a arquitetura MobileNetV2 com transferência de aprendizado e inferência interativa via upload de imagens no Jupyter Notebook.

---

## 🔧 Justificativa das Escolhas Técnicas

- **Arquitetura Base**: MobileNetV2 foi escolhida por sua leveza, eficiência e ótimo desempenho em dispositivos com recursos limitados.
- **Transfer Learning**: A base do modelo foi congelada inicialmente para evitar overfitting precoce. Em seguida, as últimas camadas foram descongeladas para fine-tuning.
- **Funções de Ativação**:
  - `ReLU` nas camadas densas e convolucionais intermediárias por ser simples e eficaz.
  - `Sigmoid` na saída, adequada para classificação binária.
- **Função Objetivo**: `binary_crossentropy`, padrão para classificação com duas classes.
- **Regularização**:
  - `Dropout`, `BatchNormalization` e `l1_l2` foram usados para minimizar overfitting.
- **Validação**:
  - Divisão entre treino e validação foi feita com diretórios distintos.
  - Utilização de callbacks como `EarlyStopping`, `ReduceLROnPlateau` e `ModelCheckpoint` para estabilidade do treinamento.

---

## 🖥️ Setup do Ambiente

### 1. Clonar o repositório ou baixar os arquivos

```bash
git clone https://github.com/seu_usuario/projeto-horse-human.git
```
### 2. Criar e ativar ambiente virtual (opcional)
```bash

python -m venv venv
# Ativação no Windows:
venv\Scripts\activate
# Ativação no Linux/macOS:
source venv/bin/activate
```

### 3. Instalar dependências
```bash

pip install -r requirements.txt
# Alternativamente, instale manualmente:

pip install tensorflow matplotlib numpy pillow ipywidgets
```

### 4. Executar o notebook
```bash

jupyter notebook
# E abra o notebook principal do projeto.
````

### 📦 Módulo de Inferência
## Upload de imagens diretamente no Jupyter Notebook
O notebook permite ao usuário carregar imagens via widget FileUpload (ipywidgets). Após o upload, o modelo carrega as imagens, faz a predição e exibe o resultado visual com rótulo e confiança.

### ✔️ Formatos Aceitos
PNG, JPG, JPEG ou outros compatíveis com PIL.Image.

### 📐 Tamanho das imagens
As imagens são redimensionadas para (180, 180) antes da inferência.

### 📊 Exibição dos Resultados
As imagens são exibidas em um grid com no máximo 5 imagens por linha, todas com tamanho uniforme, e legendas com a classe prevista e a confiança.

### 📊 Análise de Performance
Acurácia final de validação: aproximadamente 97%.

O modelo apresentou val_loss estável após o fine-tuning e callbacks.

A performance foi avaliada com gráficos de loss e acurácia por época.

O classificador é adequado para uso inicial, experimentação ou como base para APIs.

### 🚀 Próximos Passos
Aumentar a base de dados com novas amostras.

Implementar uma API com FastAPI ou Flask para inferência em tempo real.

Testar novas arquiteturas (EfficientNet, ResNet50, etc.).

Exportar o modelo para .tflite visando uso em dispositivos móveis.

Implementar validação cruzada para avaliação mais robusta.

### 📁 Estrutura de Diretórios Esperada
```bash
dataset/
├── horse-or-human/
│   ├── horses/
│   └── humans/
├── validation-horse-or-human/
│   ├── horses/
│   └── humans/
```

### 🧠 Autor
Estêvão Santos Cavalcante
