# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflowjs as tfjs

# Descargar el dataset "cats_vs_dogs" con sus metadatos; se utiliza 'as_supervised=True' para obtener pares (imagen, etiqueta)
datos, metadatos = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)

import matplotlib.pyplot as plt
import cv2

TAMANO_IMG = 100  # Definimos el tamaño al que redimensionaremos las imágenes

# Visualización preliminar: se configuran las dimensiones de la figura para mostrar algunas imágenes
plt.figure(figsize=(20,20))
for i, (imagen, etiqueta) in enumerate(datos['train'].take(25)):
    # Convertimos la imagen a array de NumPy y la redimensionamos
    imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
    # Convertimos la imagen a escala de grises
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # Aquí se podría usar plt.imshow(imagen, cmap='gray') para visualizar la imagen

# Preprocesamiento de datos: convertir todas las imágenes a un formato adecuado para el entrenamiento
datos_entrenamiento = []
for i, (imagen, etiqueta) in enumerate(datos['train']):
    imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))  # Redimensiona la imagen
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)  # Convierte la imagen a escala de grises
    imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG, 1)  # Añade la dimensión de canal (1 canal para escala de grises)
    datos_entrenamiento.append([imagen, etiqueta])

# Separar imágenes (X) y etiquetas (y) en listas individuales
X = []
y = []
for imagen, etiqueta in datos_entrenamiento:
    X.append(imagen)
    y.append(etiqueta)

import numpy as np
# Convertir las listas a arrays de NumPy y normalizar las imágenes (valores entre 0 y 1)
X = np.array(X).astype(float) / 255
y = np.array(y)

# Función que construye y compila el modelo, parametrizado para ajustar hiperparámetros con Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(100, 100, 1)))  # Capa de entrada para imágenes 100x100 con 1 canal

    # Hiperparámetro: número de bloques convolucionales (entre 2 y 4)
    num_blocks = hp.Int("num_conv_blocks", min_value=2, max_value=4)

    # Opciones para el número de filtros en las capas convolucionales
    filter_options = [32, 64, 128, 256]

    for i in range(num_blocks):
        # Selección dinámica de filtros para cada bloque convolucional
        filters = hp.Choice(f"conv_{i+1}_filters", filter_options[:len(filter_options) - (3 - i)])
        model.add(Conv2D(filters=filters, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))  # Capa de pooling para reducir dimensiones

    model.add(Flatten())  # Aplanamiento de la salida para conectarla a las capas densas
    model.add(Dense(100, activation='relu'))  # Capa densa intermedia con 100 neuronas
    model.add(Dense(1, activation='sigmoid'))  # Capa de salida para clasificación binaria (gato vs. perro)

    # Hiperparámetros para la optimización: tasa de aprendizaje y elección del optimizador
    lr = hp.Choice('lr', values=[1e-2, 1e-3, 1e-4])
    optimizer_choice = hp.Choice("optimizer", values=["SGD", "Adam", "Adagrad"])
    optimizers_dict = {
        "Adam": tf.keras.optimizers.Adam(learning_rate=lr),
        "SGD": tf.keras.optimizers.SGD(learning_rate=lr),
        "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=lr)
    }

    # Compilar el modelo utilizando la función de pérdida 'binary_crossentropy' y la métrica 'accuracy'
    model.compile(optimizer=optimizers_dict[optimizer_choice],
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# Configuración del tuner Hyperband para la búsqueda de hiperparámetros
tuner = kt.Hyperband(
    build_model,  # Función para construir el modelo
    objective=kt.Objective("val_accuracy", "max"),  # Objetivo: maximizar la precisión en validación
    executions_per_trial=1,
    max_epochs=10,  # Número máximo de épocas en cada prueba de hiperparámetros
    factor=3,
    directory='salida',  # Directorio para guardar resultados del tuner
    project_name='intro_to_HP',
    overwrite=True,
)

# Configuración del ImageDataGenerator para aumentar los datos mediante transformaciones aleatorias
datagen = ImageDataGenerator(
    rotation_range=30,          # Rotación aleatoria de hasta 30 grados
    width_shift_range=0.2,      # Desplazamiento horizontal de hasta el 20%
    height_shift_range=0.2,     # Desplazamiento vertical de hasta el 20%
    shear_range=15,             # Ángulo de cizallamiento
    zoom_range=[0.7, 1.4],        # Rango de zoom
    horizontal_flip=True,       # Volteo horizontal
    vertical_flip=True          # Volteo vertical
)
datagen.fit(X)  # Ajusta el generador a los datos

# División de los datos en entrenamiento y validación
X_entrenamiento = X[:19700]
X_validacion = X[19700:]
y_entrenamiento = y[:19700]
y_validacion = y[19700:]

# Creación del generador de datos para el entrenamiento con batch size de 32
data_gen_entrenamiento = datagen.flow(X_entrenamiento, y_entrenamiento, batch_size=32)
steps_per_epoch = X_entrenamiento.shape[0] // 32  # Número de pasos por cada época

# Definición de callbacks para optimizar el entrenamiento:
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, verbose=1),  # Detiene el entrenamiento si la pérdida de validación no mejora en 3 épocas
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)  # Reduce la tasa de aprendizaje si la pérdida se estanca
]

# Búsqueda de los mejores hiperparámetros usando el tuner y el generador de datos
hist = tuner.search(
    data_gen_entrenamiento,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=(X_validacion, y_validacion),
    callbacks=callbacks
)

# Obtención de los mejores hiperparámetros y construcción del modelo óptimo
best_hps = tuner.get_best_hyperparameters()[0]
mi_mejor_modelo = tuner.hypermodel.build(best_hps)
mi_mejor_modelo.summary()  # Muestra la arquitectura del modelo

# Función para graficar la evolución de la precisión y la pérdida durante el entrenamiento
def plot_hist(hist):
    history = hist.history
    # Gráfica de precisión para entrenamiento y validación
    plt.plot(history["accuracy"], label="Entrenamiento")
    plt.plot(history["val_accuracy"], label="Validación")
    plt.title("Precisión del modelo (Accuracy)")
    plt.ylabel("Precisión")
    plt.xlabel("Época")
    plt.ylim((0, 1.1))
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("accuracy.png")  # Guarda la gráfica en un archivo
    plt.close()

    # Gráfica de pérdida para entrenamiento y validación
    plt.plot(history["loss"], 'r', label="Entrenamiento")
    plt.plot(history["val_loss"], 'b', label="Validación")
    plt.title("Pérdida del modelo (Loss)")
    plt.ylabel("Pérdida")
    plt.xlabel("Época")
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig("loss.png")  # Guarda la gráfica en un archivo
    plt.close()

# Entrenamiento final del modelo óptimo usando el generador de datos
historial = mi_mejor_modelo.fit(
    data_gen_entrenamiento,
    epochs=120,             # Entrenar durante 120 épocas
    batch_size=32,
    validation_data=(X_validacion, y_validacion),
    steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
    validation_steps=int(np.ceil(len(X_validacion) / float(32)))
)

# Graficar la evolución del entrenamiento y guardar las imágenes
plot_hist(historial)

# Guardar el modelo entrenado en formato HDF5 y convertirlo a TensorFlow.js para su uso en aplicaciones web
mi_mejor_modelo.save('perros-gatos-cnn.h5')
tfjs.converters.save_keras_model(mi_mejor_modelo, "modelo_tfjs")

