import cv2
import json
import numpy as np
import os

# Cargar la imagen del mapa
map_image_path = 'casas.png'
map_image = cv2.imread(map_image_path)

# Definir el factor de escala
scale_factor = 8

# Ruta de la carpeta que contiene los sprites
sprites_folder = "sprites"

# Cargar los sprites
sprites = {
    "green": cv2.imread(os.path.join(sprites_folder, "verde.png")),
    "magenta": cv2.imread(os.path.join(sprites_folder, "magenta.png")),
    "yellow": cv2.imread(os.path.join(sprites_folder, "amarillo.png")),
    "blue": cv2.imread(os.path.join(sprites_folder, "azul.png")),
    "red": cv2.imread(os.path.join(sprites_folder, "rojo.png")),
    "cyan": cv2.imread(os.path.join(sprites_folder, "cyan.png")),
    "orange": cv2.imread(os.path.join(sprites_folder, "naranja.png")),
    "gray": cv2.imread(os.path.join(sprites_folder, "gris.png")),
    "dark_green": cv2.imread(os.path.join(sprites_folder, "verdeoscuro.png")),
    "white": cv2.imread(os.path.join(sprites_folder, "blanco.png")),  # Para casos donde no haya match de color
    "black": cv2.imread(os.path.join(sprites_folder, "negro.png"))  # Cargar el sprite negro para los agentes
}

# Mapear los colores BGR a los sprites correspondientes
color_to_sprite = {
    (0, 255, 0): sprites["green"],        # Verde
    (255, 0, 255): sprites["magenta"],     # Magenta
    (0, 255, 255): sprites["yellow"],      # Amarillo
    (255, 0, 0): sprites["blue"],          # Azul
    (0, 0, 255): sprites["red"],           # Rojo
    (255, 255, 0): sprites["cyan"],        # Cyan
    (0, 165, 255): sprites["orange"],      # Naranja
    (127, 127, 127): sprites["gray"],      # Gris
    (0, 200, 0): sprites["dark_green"],    # Verde oscuro
}

# Dimensiones del sprite (suponiendo que todos son del mismo tamaño)
sprite_height, sprite_width, _ = sprites["yellow"].shape

# Crear la imagen de fondo compuesta por los sprites
def create_background_image(map_image, color_to_sprite, scale_factor):
    height, width, _ = map_image.shape
    background_image = np.zeros((height * scale_factor, width * scale_factor, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            # Obtener el color del píxel en la imagen original
            pixel_color = tuple(map_image[y, x])
            # Obtener el sprite correspondiente
            sprite = color_to_sprite.get(pixel_color, sprites["white"])  # Usar sprite blanco por defecto
            
            # Superponer el sprite escalado en la posición correcta
            background_image[y*scale_factor:(y+1)*scale_factor, x*scale_factor:(x+1)*scale_factor] = sprite
            
    return background_image

# Crear la imagen de fondo utilizando los sprites
background_image = create_background_image(map_image, color_to_sprite, scale_factor)

# Inicializar un diccionario para almacenar las posiciones anteriores de los agentes
previous_positions = {}

# Función para interpolar entre dos puntos
def interpolate(start, end, t):
    return start + t * (end - start)

# Leer la información de los agentes y actualizar la imagen
def update_agents_on_map(background_image, sprites, scale_factor, previous_positions, json_path='agent_positions.json'):
    # Crear una copia de la imagen de fondo para superponer los sprites
    dynamic_layer = np.zeros_like(background_image)
    
    # Intentar leer el archivo JSON y manejar posibles errores
    try:
        with open(json_path, 'r') as json_file:
            agent_data = json.load(json_file)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error al leer el archivo JSON: {e}")
        return background_image, previous_positions  # Retorna la imagen de fondo sin cambios
    
    # Diccionario para almacenar las nuevas posiciones
    new_positions = {}
    
    # Interpolación de las posiciones
    steps = 20  # Número de pasos para la animación
    for step in range(steps):
        dynamic_layer.fill(0)  # Limpiar la capa dinámica para el nuevo paso de animación

        for agent in agent_data.get("agents", []):
            agent_id = agent["id"]
            agent_position = agent["position"]
            x_new, y_new = int(agent_position[1]) * scale_factor, int(agent_position[0]) * scale_factor
            
            # Obtener la posición anterior
            if agent_id in previous_positions:
                x_old, y_old = previous_positions[agent_id]
            else:
                x_old, y_old = x_new, y_new
            
            # Interpolar la posición
            x_interpolated = int(interpolate(x_old, x_new, step / steps))
            y_interpolated = int(interpolate(y_old, y_new, step / steps))
            
            # Superponer el sprite del agente en la posición interpolada
            dynamic_layer[y_interpolated:y_interpolated + sprite_height, x_interpolated:x_interpolated + sprite_width] = sprites["black"]

        # Combinar la capa estática (fondo) con la capa dinámica (agentes)
        combined_image = cv2.addWeighted(background_image, 1.0, dynamic_layer, 1.0, 0)
        
        # Mostrar la imagen resultante
        cv2.imshow('Mapa con Agentes', combined_image)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            return combined_image, new_positions

    # Actualizar las posiciones anteriores
    for agent in agent_data.get("agents", []):
        agent_id = agent["id"]
        agent_position = agent["position"]
        x_new, y_new = int(agent_position[1]) * scale_factor, int(agent_position[0]) * scale_factor
        new_positions[agent_id] = (x_new, y_new)
    
    return combined_image, new_positions

# Bucle para actualizar la imagen cada segundo
while True:
    # Actualizar la imagen con los agentes y animar el movimiento
    background_image, previous_positions = update_agents_on_map(background_image, sprites, scale_factor, previous_positions)
    
    # Esperar 300 ms antes de la siguiente actualización
    if cv2.waitKey(300) & 0xFF == ord('q'):
        break

# Liberar recursos y cerrar ventanas
cv2.destroyAllWindows()
