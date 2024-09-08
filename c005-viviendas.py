import cv2
import numpy as np
import random

def offset_polygon(points, offset):
    """Desplaza los puntos del polígono hacia adentro o afuera, según el offset."""
    # Calcular el centroide del polígono
    centroid = np.mean(points, axis=0)
    
    # Calcular los puntos desplazados
    offset_points = []
    for point in points:
        direction = point - centroid
        norm_direction = direction / np.linalg.norm(direction)
        new_point = point - offset * norm_direction
        offset_points.append(new_point)
    
    return np.array(offset_points, np.int32)

# Tamaño de la imagen
width, height = 1024, 1024

# Crear una imagen en blanco (negra)
image = np.zeros((height, width, 3), dtype=np.uint8)

# Generar un número aleatorio de lados entre 4 y 6 para el polígono exterior
num_lados = random.randint(4, 6)

# Definir el radio del polígono y el centro de la imagen
radius = 400
center = (width // 2, height // 2)

# Calcular los puntos del polígono exterior
points = []
for i in range(num_lados):
    angle = 2 * np.pi * i / num_lados
    x = int(center[0] + radius * np.cos(angle))
    y = int(center[1] + radius * np.sin(angle))
    points.append((x, y))

# Convertir la lista de puntos a un array de NumPy
points = np.array(points, np.int32)

# Dibujar el polígono exterior en la imagen
cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=5)

# Generar la forma interior (deslunado) desplazada hacia adentro
offset = 250  # Desfase de la forma interior
inner_points = offset_polygon(points, offset)

# Dibujar el polígono interior (deslunado) en la imagen
cv2.polylines(image, [inner_points], isClosed=True, color=(255, 0, 0), thickness=3)

# Guardar la imagen resultante
cv2.imwrite('edificio_con_deslunado.png', image)

# Mostrar la imagen
cv2.imshow('Building with Courtyard', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
