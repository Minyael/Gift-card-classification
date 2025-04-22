"""
Clasificación en tiempo real con suavizado de predicciones:
promedia las últimas N probabilidades para estabilidad.
"""
import cv2
import numpy as np
from collections import deque
from pathlib import Path
from keras.models import load_model

def webcam():

    # Cargar modelo entrenado
    curr = Path(__file__).parent
    models = curr.parent / 'trained_model_parameters'
    model_file = sorted(models.glob('best_transfer_*.h5'), reverse=True)
    if not model_file:
        model_file = sorted(models.glob('*.h5'), reverse=True)
    model = load_model(str(model_file[0]))

    # Clases (debe coincidir con carpetas train)
    class_names = sorted([p.name for p in (curr.parent / 'images_augmented' / 'train').iterdir()])

    # Preprocesamiento
    IMG_H, IMG_W = 150, 150
    def preprocess(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_W, IMG_H))
        arr = resized.astype('float32')/255.0
        return np.expand_dims(arr, axis=0)

    # Ventana de suavizado
    N = 10
    buffer = deque(maxlen=N)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): raise RuntimeError('No webcam')

    while True:
        ret, frame = cap.read()
        if not ret: break

        x = preprocess(frame)
        preds = model.predict(x)[0]
        buffer.append(preds)
        avg = np.mean(buffer, axis=0)
        idx = np.argmax(avg)
        label = class_names[idx]
        prob = avg[idx]

        text = f"{label}: {prob*100:.1f}%"
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0,255,0), 2)

        cv2.imshow('Clasificación Estable', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    webcam()