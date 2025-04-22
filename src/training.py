"""
Entrenamiento con Transfer Learning (MobileNetV2) para mejorar generalización
en clasificación de tarjetas Xbox vs Steam.
"""
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from PIL import Image, UnidentifiedImageError

# Configuración de rutas
dir_current = Path(__file__).parent
base_dir    = dir_current.parent
models_dir  = base_dir / 'trained_model_parameters'
models_dir.mkdir(exist_ok=True)

def transfer_classification():
    # Parámetros
    IMG_H, IMG_W = 150, 150
    BATCH_SIZE   = 32
    EPOCHS       = 50

    data_dir = base_dir / 'images_augmented'
    train_dir = data_dir / 'train'
    valid_dir = data_dir / 'valid'
    test_dir  = data_dir / 'test'

    # Augmentación solo en entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(IMG_H, IMG_W), batch_size=BATCH_SIZE, class_mode='categorical'
    )
    valid_gen = val_datagen.flow_from_directory(
        valid_dir, target_size=(IMG_H, IMG_W), batch_size=BATCH_SIZE, class_mode='categorical'
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=(IMG_H, IMG_W), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )

    # Crear modelo base
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_H, IMG_W, 3)
    )
    base_model.trainable = False  # congelar

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(train_gen.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=optimizers.Adam(lr=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    ts = int(datetime.now().timestamp())
    cp = ModelCheckpoint(
        filepath=str(models_dir / f'best_transfer_{ts}.h5'),
        monitor='val_loss', save_best_only=True, verbose=1
    )
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    # Entrenamiento
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=valid_gen,
        callbacks=[cp, es, rl],
        workers=4,
        use_multiprocessing=False
    )

    # Guardar final
    model.save(models_dir / f'final_transfer_{ts}.h5')

    # Graficar
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # Evaluación
    loss, acc = model.evaluate(test_gen)
    print('Test loss:', loss)
    print('Test accuracy:', acc)


if __name__ == '__main__':
    transfer_classification()