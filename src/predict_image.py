"""
predict_image.py

Script para clasificar imágenes estáticas usando el modelo entrenado.
Puede procesar un único archivo o recorrer toda la carpeta "images_test".
"""
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from keras.models import load_model

# Configurar rutas base
dir_curr = Path(__file__).parent
base_dir = dir_curr.parent
models_dir = base_dir / 'trained_model_parameters'
images_test_dir = base_dir / 'images_test'

# Seleccionar modelo (último best_model_*.h5 o cualquiera .h5)
model_files = sorted(models_dir.glob('best_model_*.h5'), reverse=True)
model_path = model_files[0] if model_files else (
    sorted(models_dir.glob('*.h5'), reverse=True)[0]
)

# Cargar modelo y clases
def load_trained_model():
    print(f"Cargando modelo: {model_path.name}")
    return load_model(str(model_path))

def get_class_names():
    train_dir = base_dir / 'images_augmented' / 'train'
    return sorted([p.name for p in train_dir.iterdir() if p.is_dir()])

# Preprocesamiento
IMG_H, IMG_W = 150, 150
def preprocess_image(frame):
    # Si frame es ruta, lee con OpenCV
    if isinstance(frame, Path) or isinstance(frame, str):
        frame = cv2.imread(str(frame))
    if frame is None:
        raise ValueError("No se pudo leer la imagen")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_W, IMG_H))
    arr = resized.astype('float32') / 255.0
    return np.expand_dims(arr, axis=0), rgb

# Función principal de predicción
def predict_images(image_name=None):
    model = load_trained_model()
    class_names = get_class_names()

    # Lista de rutas a procesar
    if image_name:
        img_path = images_test_dir / image_name
        imgs = [img_path]
    else:
        imgs = sorted([p for p in images_test_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')])

    for img_file in imgs:
        x, orig = preprocess_image(img_file)
        preds = model.predict(x)
        idx = np.argmax(preds, axis=1)[0]
        label = class_names[idx]
        prob = preds[0, idx] * 100

        print(f"{img_file.name} -> {label}: {prob:.1f}%")

        # Mostrar imagen con matplotlib
        plt.figure(figsize=(4,4))
        plt.imshow(orig)
        plt.title(f"{label}: {prob:.1f}%")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clasificar imágenes estáticas con CNN entrenada.')
    parser.add_argument('--image', '-i', help='Nombre de archivo en images_test/', default=None)
    args = parser.parse_args()

    predict_images(image_name=args.image)
