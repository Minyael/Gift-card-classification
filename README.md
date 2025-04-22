# README - Redes neuronales con numpy

---

## Descripción General

Este proyecto entrena una red neuronal convolucional (CNN) usando Keras y TensorFlow para clasificar tarjetas (como Xbox y Steam) capturadas mediante una cámara web o proporcionadas como imágenes estáticas. El modelo se entrena sobre imágenes augmentadas y permite probar predicciones en tiempo real. Este proyecto fue realizado con Python 3.11.

---

## Requisitos Previos

Para probar el código, clona el repositorio usando:

```bash
cd [ubicación]
git clone https://github.com/Minyael/Gift-card-classification/
```

Opcionalmente, puedes usar un entorno virtual para aislar dependencias:

```bash
python -m venv env
source env/bin/activate  # En Linux/macOS
env\Scripts\activate  # En Windows
pip install -r requirements.txt
```

---

## Estructura del Proyecto

```
📂 project/
├── 📂 images/                    # Imágenes originales
├── 📂 images_augmented/         # Imágenes augmentadas (train/valid/test)
├── 📂 images_test/              # Imágenes para pruebas estáticas
├── 📂 trained_model_parameters/ # Pesos del modelo (.h5)
├── 📂 src/
│   ├── training.py              # Entrenamiento del modelo con transferencia
│   ├── webcam.py                # Clasificación en vivo con webcam
│   ├── predict_image.py         # Clasificación de imágenes estáticas
│   ├── data_augmentation.py     # Augmentación de imágenes
│   └── image_download.py        # (Opcional) Descarga imágenes de internet
├── main.py                      # Archivo principal del proyecto
├── requirements.txt             # Lista de dependencias
└── README.md                    # Documentación del proyecto

```

---

## Implementación del Modelo

### Entrenamiento del modelo

El entrenamiento se realiza con un modelo preentrenado (como MobileNetV2) usando transferencia de aprendizaje. El modelo se entrena sobre imágenes augmentadas y guarda los pesos que logran el mejor rendimiento de validación.

```python
python main.py train
```

Se generan gráficas de accuracy y pérdida para evaluar el entrenamiento. El entrenamiento se detiene automáticamente con EarlyStopping si no hay mejoras.

### Clasificación en tiempo real (Webcam)

Para probar el modelo con cámara web en tiempo real:

```python
python main.py webcam
```

El modelo mostrará predicciones en vivo junto con el porcentaje de certeza.


### Clasificación de imágenes estáticas

También puedes probar imágenes estáticas guardadas en images_test/:

```bash
python main.py test
```

Mostrará la imagen junto a su predicción y porcentaje correspondiente.


### Otras utilidades

Augmentación de imágenes:
Aplica transformaciones como rotación, cambio de brillo y volteo:

```bash
python main.py augmentation
```

Descarga de imágenes (opcional):
Utiliza image_download.py para descargar imágenes desde la web (requiere configuración):

```bash
python src\image_download.py
```

## Resultados esperados

Durante el entrenamiento ideal:

- Accuracy alto desde las primeras épocas.
- Pérdida (loss) cercana a cero.
- Resultados estables en predicciones si se capturan en condiciones similares a las del dataset.

---

## Conclusión

Este proyecto demuestra cómo combinar transfer learning, data augmentation y keras/tensorflow para construir una solución práctica de clasificación de objetos en tiempo real. Puedes extenderlo a nuevas clases de tarjetas, ajustar hiperparámetros o refinar el modelo según tus necesidades.
