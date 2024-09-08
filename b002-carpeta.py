import cv2
import json
import numpy as np
import time
import os

# Cargar la imagen del mapa
map_image_path = 'casas.png'
map_image = cv2.imread(map_image_path)

# Escalar la imagen 8 veces
scale_factor = 8
scaled_map_image = cv2.resize(map_image, (map_image.shape[1] * scale_factor, map_image.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST)

# Ruta de la carpeta que contiene los sprites
sprites_folder = "sprites"

# Cargar los sprites
sprites = {
    "yellow": cv2.imread(os.path.join(sprites_folder, "amarillo.png")),
    "blue": cv2.imread(os.path.join(sprites_folder, "azul.png")),
    "white": cv2.imread(os.path.join(sprites_folder, "blanco.png")),
    "cyan": cv2.imread(os.path.join(sprites_folder, "cyan.png")),
    "gray": cv2.imread(os.path.join(sprites_folder, "gris.png")),
    "magenta": cv2.imread(os.path.join(sprites_folder, "magenta.png")),
    "orange": cv2.imread(os.path.join(sprites_folder, "naranja.png")),
    "black": cv2.imread(os.path.join(sprites_folder, "negro.png")),
    "red": cv2.imread(os.path.join(sprites_folder, "rojo.png")),
    "green": cv2.imread(os.path.join(sprites_folder, "verde.png")),
    "dark_green": cv2.imread(os.path.join(sprites_folder, "verdeoscuro.png"))
}

# Dimensiones del sprite (suponiendo que todos son del mismo tamaño)
sprite_height, sprite_width, _ = sprites["yellow"].shape

# Leer la información de los agentes y actualizar la imagen
def update_agents_on_map(map_image, sprites, scale_factor, json_path='agent_positions.json'):
    # Crear una copia de la imagen de fondo para superponer los sprites
    dynamic_layer = np.zeros_like(map_image)
    
    # Leer el archivo JSON
    with open(json_path, 'r') as json_file:
        agent_data = json.load(json_file)
    
    # Dibujar cada agente en su posición correspondiente
    for agent in agent_data["agents"]:
        agent_position = agent["position"]
        x, y = int(agent_position[1]) * scale_factor, int(agent_position[0]) * scale_factor

        # Superponer el sprite del agente en la posición escalada
        dynamic_layer[y:y + sprite_height, x:x + sprite_width] = sprites["black"]

    # Combinar la capa estática (mapa escalado) con la capa dinámica (agentes)
    combined_image = cv2.addWeighted(map_image, 1.0, dynamic_layer, 1.0, 0)
    
    return combined_image

# Bucle para actualizar la imagen cada segundo
while True:
    # Actualizar la imagen con los agentes
    output_image = update_agents_on_map(scaled_map_image, sprites, scale_factor)
    
    # Mostrar la imagen resultante
    cv2.imshow('Mapa con Agentes', output_image)
    
    # Esperar 1 segundo
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

# Liberar recursos y cerrar ventanas
cv2.destroyAllWindows()
