import cv2
import numpy as np
import heapq
import time

# Cargar la imagen proporcionada
image_path = 'casas.png'  # Reemplaza con la ruta a tu imagen
image = cv2.imread(image_path)

# Ampliar la imagen para que cada píxel sea un bloque de 4x4
scale_factor = 1
resized_image = cv2.resize(image, (image.shape[1] * scale_factor, image.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST)

# Definir los colores en BGR
colors = {
    "green": (0, 255, 0),    # terreno pisable
    "magenta": (255, 0, 255), # asfalto (no pisable)
    "yellow": (0, 255, 255),  # comida
    "blue": (255, 0, 0),      # cama (descanso)
    "red": (0, 0, 255),       # WC (necesidades fisiológicas)
    "cyan": (255, 255, 0),    # descanso
    "orange": (0, 165, 255)   # polilínea de recorrido
}

# Creación del entorno de simulación
simulation_map = resized_image.copy()

# Identificar todas las posiciones pisables (verdes) en el mapa
walkable_positions = np.argwhere(np.all(simulation_map == colors["green"], axis=-1))

# Parámetros de la simulación
num_agents = 400  # Aumenta el número de agentes para ver múltiples recorridos

# Asegurar que los agentes solo se spawneen en posiciones pisables (verdes)
def find_nearest_green_position(position, walkable_positions):
    distances = np.linalg.norm(walkable_positions - position, axis=1)
    nearest_index = np.argmin(distances)
    return walkable_positions[nearest_index]

agent_positions = []
for _ in range(num_agents):
    initial_position = walkable_positions[np.random.choice(len(walkable_positions))]
    if not np.all(simulation_map[initial_position[0], initial_position[1]] == colors["green"]):
        initial_position = find_nearest_green_position(initial_position, walkable_positions)
    agent_positions.append(initial_position)

agent_positions = np.array(agent_positions) * scale_factor

# Inicializar posiciones y necesidades de los agentes
agent_states = {
    "positions": agent_positions,
    "needs": np.random.choice(["food", "rest", "wc"], size=num_agents),
    "paths": [None] * num_agents,  # Para almacenar el camino calculado por A*
    "targets": [None] * num_agents  # Para almacenar el objetivo actual
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
            if 0 <= neighbor[0] < simulation_map.shape[0] and 0 <= neighbor[1] < simulation_map.shape[1]:
                current_color = simulation_map[neighbor[0], neighbor[1]].tolist()
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
    window_name = 'Simulacion en Tiempo Real'  # Nombre único para la ventana de simulación
    stats_window_name = 'Estadisticas de Agentes'  # Nombre único para la ventana de estadísticas
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Crear una ventana con un nombre fijo
    cv2.namedWindow(stats_window_name, cv2.WINDOW_NORMAL)  # Crear una ventana para las estadísticas

    stats_image = np.zeros((400, 400, 3), dtype=np.uint8)  # Imagen para las estadísticas, de tamaño fijo
    
    for step in range(steps):
        #print(f"\n--- Step {step + 1} ---")  # Indicar el número de paso en la consola

        # Copiar el mapa para dibujar los recorridos
        path_map = simulation_map.copy()

        for i in range(len(agent_states["positions"])):
            x, y = agent_states["positions"][i]
            need = agent_states["needs"][i]

            # Verificar si se ha alcanzado el objetivo actual
            if agent_states["targets"][i] and (x, y) == agent_states["targets"][i]:
                # Satisfacer la necesidad actual y elegir una nueva
                agent_states["needs"][i] = np.random.choice(["food", "rest", "wc"])
                agent_states["paths"][i] = None  # Reiniciar el camino
                agent_states["targets"][i] = None  # Reiniciar el objetivo
                #print(f"Agente {i} ha satisfecho su necesidad y ahora tiene una nueva necesidad: {agent_states['needs'][i]}")

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
                    distances = [heuristic((x, y), tuple(pos * scale_factor)) for pos in target_positions]
                    nearest_index = np.argmin(distances)
                    target_position = tuple(target_positions[nearest_index] * scale_factor)
                    path = a_star(simulation_map, (x, y), target_position, scale_factor)
                    agent_states["paths"][i] = path if path else []  # Asignar camino o una lista vacía si no hay camino
                    agent_states["targets"][i] = target_position  # Establecer el objetivo actual

                    # Imprimir estado del agente
                    #print(f"Agente {i}: Posición actual = {(x, y)}, Necesidad = {need}, Objetivo = {target_position}, Pasos restantes = {len(agent_states['paths'][i])}")

            # Dibujar la polilínea del camino en la copia del mapa
            if agent_states["paths"][i]:
                # Ajustar el escalado aquí si es necesario
                points = np.array(agent_states["paths"][i], np.int32) * scale_factor
                points = points.reshape((-1, 1, 2))
                cv2.polylines(path_map, [points], isClosed=False, color=colors["orange"], thickness=1)  # Grosor reducido a 1

            # Moverse al siguiente paso del camino calculado
            if agent_states["paths"][i]:
                next_position = agent_states["paths"][i].pop(0)
                agent_states["positions"][i] = next_position
                #print(f"Agente {i} se mueve a la posición {next_position}")  # Agrega un mensaje para verificar el movimiento

        # Dibujar agentes en el mapa
        temp_map = path_map.copy()
        for pos in agent_states["positions"]:
            cv2.rectangle(temp_map, (pos[1], pos[0]), (pos[1]+scale_factor-1, pos[0]+scale_factor-1), (0, 0, 0), -1)

        # Mostrar el estado actual en la ventana con un nombre fijo
        cv2.imshow(window_name, temp_map)

        # Actualizar la imagen de estadísticas
        stats_image[:] = (0, 0, 0)  # Limpiar la imagen
        for i in range(len(agent_states["positions"])):
            stat_text = f"Agente {i}: Pos ({agent_states['positions'][i][1]},{agent_states['positions'][i][0]}), Necesidad: {agent_states['needs'][i]}"
            cv2.putText(stats_image, stat_text, (10, 20 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Mostrar la ventana de estadísticas
        cv2.imshow(stats_window_name, stats_image)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        #time.sleep(0.1)  # Pausa entre iteraciones para reducir carga del sistema

    cv2.destroyAllWindows()

# Simular un día (aquí simplificado a unos pocos pasos)
move_agents(agent_states, simulation_map, scale_factor, steps=240000)  # Simulación de 2400 pasos
