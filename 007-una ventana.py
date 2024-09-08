import cv2
import numpy as np
import heapq
import time

# Cargar la imagen proporcionada
image_path = 'casas.png'  # Reemplaza con la ruta a tu imagen
image = cv2.imread(image_path)

# Ampliar la imagen para que cada píxel sea un bloque de 4x4
scale_factor = 8
resized_image = cv2.resize(image, (image.shape[1] * scale_factor, image.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST)

# Definir los colores en BGR
colors = {
    "green": (0, 255, 0),    # terreno pisable
    "magenta": (255, 0, 255), # asfalto (no pisable)
    "yellow": (0, 255, 255),  # comida
    "blue": (255, 0, 0),      # cama (descanso)
    "red": (0, 0, 255),       # WC (necesidades fisiológicas)
    "cyan": (255, 255, 0)     # descanso
}

# Creación del entorno de simulación
simulation_map = resized_image.copy()

# Parámetros de la simulación
num_agents = 10
agent_positions = np.random.randint(0, simulation_map.shape[1]//scale_factor, size=(num_agents, 2)) * scale_factor

# Inicializar posiciones y necesidades de los agentes
agent_states = {
    "positions": agent_positions,
    "needs": np.random.choice(["food", "rest", "wc"], size=num_agents),
    "paths": [None] * num_agents  # Para almacenar el camino calculado por A*
}

# Función de heurística para A* (distancia Manhattan)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Algoritmo A* para encontrar el camino
def a_star(simulation_map, start, goal, scale_factor):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    directions = [(0, scale_factor), (0, -scale_factor), (scale_factor, 0), (-scale_factor, 0)]
    
    while open_list:
        current = heapq.heappop(open_list)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Devuelve el camino desde el inicio hasta el objetivo
        
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= neighbor[0] < simulation_map.shape[1] and 0 <= neighbor[1] < simulation_map.shape[0]:
                current_color = simulation_map[neighbor[1], neighbor[0]].tolist()
                if current_color == list(colors["green"]) or current_color == list(colors["yellow"]) or current_color == list(colors["blue"]) or current_color == list(colors["red"]):
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
    
    return None  # Retorna None si no hay un camino posible

# Función para mover agentes según necesidades usando A*
def move_agents(agent_states, simulation_map, scale_factor, steps=10):
    window_name = 'Simulacion en Tiempo Real'  # Nombre único para la ventana
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Crear una ventana con un nombre fijo

    for step in range(steps):
        for i in range(len(agent_states["positions"])):
            x, y = agent_states["positions"][i]
            need = agent_states["needs"][i]

            # Si no hay un camino calculado o el camino está agotado, calcula un nuevo camino
            if agent_states["paths"][i] is None or len(agent_states["paths"][i]) == 0:
                # Buscar el color correspondiente según la necesidad
                target_color = {
                    "food": colors["yellow"],
                    "rest": colors["blue"],
                    "wc": colors["red"]
                }[need]

                # Encontrar todas las posiciones del color objetivo
                target_positions = np.argwhere(np.all(simulation_map == target_color, axis=-1))
                if len(target_positions) > 0:
                    # Seleccionar el objetivo más cercano
                    target_position = tuple(target_positions[np.random.choice(len(target_positions))] * scale_factor)
                    path = a_star(simulation_map, (x, y), target_position, scale_factor)
                    agent_states["paths"][i] = path if path else []  # Asignar camino o una lista vacía si no hay camino

            # Moverse al siguiente paso del camino calculado
            if agent_states["paths"][i]:
                next_position = agent_states["paths"][i].pop(0)
                agent_states["positions"][i] = next_position
            
            # Si alcanzó su objetivo o el camino se agotó, asignar una nueva necesidad aleatoria
            if len(agent_states["paths"][i]) == 0:
                agent_states["needs"][i] = np.random.choice(["food", "rest", "wc"])
                agent_states["paths"][i] = None  # Reiniciar el camino

        # Dibujar agentes en el mapa
        temp_map = simulation_map.copy()
        for pos in agent_states["positions"]:
            cv2.rectangle(temp_map, (pos[0], pos[1]), (pos[0]+scale_factor-1, pos[1]+scale_factor-1), (0, 0, 0), -1)

        # Mostrar el estado actual en la ventana con un nombre fijo
        cv2.imshow(window_name, temp_map)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        time.sleep(0.1)  # Pausa entre iteraciones para reducir carga del sistema

    cv2.destroyAllWindows()

# Simular un día (aquí simplificado a unos pocos pasos)
move_agents(agent_states, simulation_map, scale_factor, steps=2400)  # Simulación de 2400 pasos
