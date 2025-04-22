"""
Data augmentation para tu dataset de imágenes.

Lee las imágenes organizadas en:
    images/
      train/
        claseA/
        claseB/
      valid/
        claseA/
        claseB/
      test/
        claseA/
        claseB/

Por cada imagen original genera N variaciones y las guarda en:
    images_augmented/
      train/
        claseA/
        claseB/
      valid/
      test/

Configura el número de variaciones y qué conjuntos procesar.
"""
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from pathlib import Path
import os

# Configuración de augmentación
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.4, 1.5]
)

# Rutas base
dir_base = Path(__file__).parent.parent
imagenes_dir = dir_base / 'images'
output_dir    = dir_base / 'images_augmented'


def data_augmentation(num_variations=5, subsets=('train', 'valid', 'test')):
    """
    Genera `num_variations` para cada imagen en las carpetas indicadas y las guarda
    en una estructura paralela bajo `images_augmented`.

    :param num_variations: int, variaciones por imagen
    :param subsets: tuple de carpetas a procesar (train, valid, test)
    """
    for subset in subsets:
        src_subset = imagenes_dir / subset
        dst_subset = output_dir / subset
        if not src_subset.exists():
            print(f"[AVISO] '{src_subset}' no existe, se omite.")
            continue
        
        # Iterar cada clase
        for class_dir in src_subset.iterdir():
            if not class_dir.is_dir():
                continue
            
            # Crear carpeta de salida para la clase
            dst_class = dst_subset / class_dir.name
            dst_class.mkdir(parents=True, exist_ok=True)

            # Procesar cada imagen en la clase
            for img_file in class_dir.iterdir():
                if not img_file.is_file():
                    continue
                if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                    continue

                # Cargar imagen y convertir a array
                img = load_img(img_file)
                x   = img_to_array(img)
                x   = x.reshape((1,) + x.shape)

                # Generar y guardar variaciones
                count = 0
                for batch in datagen.flow(
                    x,
                    batch_size=1,
                    save_to_dir=str(dst_class),
                    save_prefix=img_file.stem,
                    save_format='jpg'
                ):
                    count += 1
                    if count >= num_variations:
                        break

            print(f"-> {num_variations} imágenes aumentadas para "
                  f"'{class_dir.name}' en '{subset}'")


if __name__ == '__main__':
    # Ejecutar augmentación: 10 variaciones en train y valid
    data_augmentation(num_variations=10, subsets=('train', 'valid', 'test'))
