import numpy as np
import cv2
import math

# Set canvas size
ancho, alto = 4096, 4096
canvas = np.full((alto, ancho, 3), (220, 220, 220), dtype=np.uint8)

# Function to draw patterns
def draw_pattern(canvas, num_rounds, initial_radius, growth_factor, num_points, line_thickness, circle_outer_radius, circle_inner_radius):
    center_x, center_y = ancho // 2, alto // 2
    radio = initial_radius
    numero = num_points

    for ronda in range(num_rounds):
        oldx1, oldy1 = [], []
        x1, y1 = [], []

        for i in range(numero):
            angle = 2 * math.pi * (i / numero)
            offset_x = (np.random.random() - 0.5) * 15
            offset_y = (np.random.random() - 0.5) * 15
            x = int(center_x + math.cos(angle) * radio + offset_x)
            y = int(center_y + math.sin(angle) * radio + offset_y)
            x1.append(x)
            y1.append(y)

            if ronda > 0:
                best_candidate = 0
                best_distance = 1000000
                for j in range(len(oldx1)):
                    distance = math.sqrt((oldx1[j] - x) ** 2 + (oldy1[j] - y) ** 2)
                    if distance < best_distance:
                        best_distance = distance
                        best_candidate = j
                cv2.line(canvas, (oldx1[best_candidate], oldy1[best_candidate]), (x, y), (255, 255, 255), line_thickness)

            oldx1 = x1
            oldy1 = y1

        for i in range(numero):
            cv2.circle(canvas, (x1[i], y1[i]), circle_outer_radius, (255, 255, 255), -1)
            cv2.circle(canvas, (x1[i], y1[i]), circle_inner_radius, (200, 200, 200), -1)

        numero = int(numero * growth_factor)
        radio *= 1.2

# First pattern
draw_pattern(canvas, num_rounds=6, initial_radius=50, growth_factor=1.0, num_points=9, line_thickness=18, circle_outer_radius=25, circle_inner_radius=10)

# Second pattern
draw_pattern(canvas, num_rounds=26, initial_radius=130, growth_factor=1.1, num_points=6, line_thickness=12, circle_outer_radius=15, circle_inner_radius=5)

# Third pattern
draw_pattern(canvas, num_rounds=126, initial_radius=10, growth_factor=1.055, num_points=6, line_thickness=7, circle_outer_radius=2, circle_inner_radius=1)

# Show the image
cv2.imshow('Generated Pattern', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
