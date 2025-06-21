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
