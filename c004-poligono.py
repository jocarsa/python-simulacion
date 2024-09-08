import cv2
import numpy as np
import random

# Tamaño de la imagen
width, height = 1024, 1024

# Crear una imagen en blanco (negra)
image = np.zeros((height, width, 3), dtype=np.uint8)

# Generar un número aleatorio de lados entre 4 y 6
num_lados = random.randint(4, 6)

# Definir el radio del polígono y el centro de la imagen
radius = 300
center = (width // 2, height // 2)

# Calcular los puntos del polígono
points = []
for i in range(num_lados):
    angle = 2 * np.pi * i / num_lados
    x = int(center[0] + radius * np.cos(angle))
    y = int(center[1] + radius * np.sin(angle))
    points.append((x, y))

# Convertir la lista de puntos a un array de NumPy
points = np.array(points, np.int32)

# Dibujar el polígono en la imagen
cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=5)

# Guardar la imagen resultante
cv2.imwrite('poligono_aleatorio.png', image)

# Mostrar la imagen
cv2.imshow('Polygon', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
