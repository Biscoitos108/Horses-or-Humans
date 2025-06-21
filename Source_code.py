#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# In[41]:


# Caminhos dos diretórios
train_dir = 'C:/Users/Estevao Cavalcante/Downloads/horse-or-human'
val_dir = 'C:/Users/Estevao Cavalcante/Downloads/validation-horse-or-human'


# In[42]:


# Augmentação para treino
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=(0.6, 1.4),
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(180, 180),
    batch_size=32,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(180, 180),
    batch_size=32,
    class_mode='binary'
)


# In[43]:


# Base do modelo
base_model = MobileNetV2(input_shape=(180, 180, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Congela a base no início


# In[44]:


# rodar se já tiver modelo pré-treinado:
loaded_model = tf.keras.models.load_model('C:/Users/Estevao Cavalcante/Downloads/HorseOrHuman/best_model_horse_human.keras')


# In[45]:


# Cabeça do modelo
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)


# In[46]:


# Compilação
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['acc']
)


# In[47]:


model.summary()


# In[48]:


# Descongela as últimas camadas para fine-tuning
base_model.trainable = True

# Treine só as últimas N camadas
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['acc']
)


# In[49]:


# Treine novamente com base descongelada
earlystop = EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model_horse_human.keras', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1)

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[earlystop, checkpoint, reduce_lr]
)


# In[50]:


print("Média do Loss de Treino:", np.mean(history_finetune.history['loss']))


# In[51]:


print("Média da Acurácia de Treino:", np.mean(history_finetune.history['acc']))


# In[52]:


print("Média do Loss de Validação:", np.mean(history_finetune.history['val_loss']))


# In[53]:


print("Média da Acurácia de Validação:", np.mean(history_finetune.history['val_acc']))


# In[54]:


plt.plot(history_finetune.history['acc'])
plt.plot(history_finetune.history['val_acc'])
plt.title('Modelo de Acurácia')
plt.ylabel('Acurácia')
plt.xlabel('Epochs')
plt.legend(['Treino', 'Validação'], loc='lower right')
plt.show()


# In[55]:


plt.plot(history_finetune.history['loss'])
plt.plot(history_finetune.history['val_loss'])
plt.title('Modelo de Perda')
plt.ylabel('Perda')
plt.xlabel('Epochs')
plt.legend(['Treino', 'Validação'], loc='lower right')
plt.show()


# In[56]:


model.save('C:/Users/Estevao Cavalcante/Downloads/HorseOrHuman/best_model_horse_human.keras') # salvando modelo


# In[ ]:





# In[ ]:





# In[98]:


from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from IPython.display import display
import io
from IPython.display import display
from ipywidgets import FileUpload
import math

# Criar o widget de upload
upload_widget = FileUpload(accept='image/*', multiple=True)
display(upload_widget)


# In[102]:


# Carregar modelo
model_path = 'C:/Users/Estevao Cavalcante/Downloads/HorseOrHuman/best_model_horse_human.keras'
model = tf.keras.models.load_model(model_path)


# In[103]:


# Parâmetros
img_width, img_height = 180, 180
class_names = {0: 'Is a horse', 1: 'Is a person'}


# In[104]:


# Fazer a previsão com o novo limiar
# Número total de imagens
n_images = len(upload_widget.value)
cols = 5
rows = math.ceil(n_images / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

# Garante que 'axes' seja sempre uma matriz 2D
axes = np.array(axes).reshape((rows, cols))

for idx, file_info in enumerate(upload_widget.value):
    row = idx // cols
    col = idx % cols

    image_data = file_info['content']
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img_resized = img.resize((img_width, img_height))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predicted_prob = model.predict(img_array)[0][0]
    predicted_class_index = int(predicted_prob > 0.5)
    predicted_class_name = class_names[predicted_class_index]
    confidence_percentage = predicted_prob * 100 if predicted_class_index == 1 else (1 - predicted_prob) * 100

    # Plot da imagem na célula correta
    axes[row, col].imshow(img)
    axes[row, col].set_title(f"{predicted_class_name}\n({confidence_percentage:.2f}%)", fontsize=10)
    axes[row, col].axis('off')

# Esconde os eixos restantes
for idx in range(n_images, rows * cols):
    row = idx // cols
    col = idx % cols
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()


# In[ ]:




