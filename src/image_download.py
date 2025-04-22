from icrawler.builtin import GoogleImageCrawler
import os

# Clases
classes = [
    "Xbox",
    "Steam"
]

# Carpeta raíz donde se guardarán las imágenes
base_dir = "./images"

# Asegurarse de que la carpeta base exista
os.makedirs(base_dir, exist_ok=True)

# Descargar imágenes para cada juego
for clase in classes:
    print(f"Descargando imágenes de: {clase}")
    game_folder = os.path.join(base_dir, clase.replace(" ", "_"))
    crawler = GoogleImageCrawler(storage={'root_dir': game_folder})
    crawler.crawl(keyword=f"{clase} giftcard", max_num=200)