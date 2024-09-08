import cv2
import numpy as np
import heapq
import time
import matplotlib.pyplot as plt

# Cargar la imagen proporcionada
image_path = 'casas.png'  # Reemplaza con la ruta a tu imagen
image = cv2.imread(image_path)

# Escalar la imagen según el factor de escala, pero mantener la proporción de la resolución original
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

# Identificar todas las posiciones de las camas (azules)
bed_positions = np.argwhere(np.all(simulation_map == colors["blue"], axis=-1))

# Parámetros de la simulación
num_agents = 140  # Aumenta el número de agentes para ver múltiples recorridos

# Asegurar que los agentes solo se spawneen en posiciones pisables (verdes)
def find_nearest_green_position(position, walkable_positions):
    distances = np.linalg.norm(walkable_positions - position, axis=1)
    nearest_index = np.argmin(distances)
    return walkable_positions[nearest_index]

agent_positions = []
agent_beds = []  # Para almacenar la cama asignada a cada agente
for _ in range(num_agents):
    initial_position = walkable_positions[np.random.choice(len(walkable_positions))]
    if not np.all(simulation_map[initial_position[0], initial_position[1]] == colors["green"]):
        initial_position = find_nearest_green_position(initial_position, walkable_positions)
    agent_positions.append(initial_position)
    
    # Asignar una cama aleatoria a cada agente (pueden compartir camas)
    bed_position = bed_positions[np.random.choice(len(bed_positions))]
    agent_beds.append(tuple(bed_position * scale_factor))

agent_positions = np.array(agent_positions) * scale_factor

# Inicializar posiciones y necesidades de los agentes
agent_states = {
    "positions": agent_positions,
    "needs": np.random.choice(["food", "rest", "wc", "resting"], size=num_agents),
    "paths": [None] * num_agents,  # Para almacenar el camino calculado por A*
    "targets": [None] * num_agents,  # Para almacenar el objetivo actual
    "beds": agent_beds,  # Cama asignada a cada agente
    "wc_timer": [0] * num_agents,  # Para almacenar el tiempo restante en WC
    "previous_need": [None] * num_agents  # Para almacenar la necesidad previa
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
                if current_color == list(colors["green"]) or current_color == list(colors["yellow"]) or current_color == list(colors["blue"]) or current_color == list(colors["red"]) or current_color == list(colors["cyan"]):
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
    
    return None  # Retorna None si no hay un camino posible

# Definir el paso de tiempo en minutos por cada iteración de la simulación
time_step_minutes = 1
current_time = 0  # Tiempo actual en minutos desde las 00:00 (por ejemplo, 0 sería 00:00, 60 sería 01:00)

# Función para convertir minutos a una cadena de hora en formato 24h
def minutes_to_time_str(minutes):
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02}:{mins:02}"

# Función para mover agentes según necesidades usando A*
def move_agents(agent_states, simulation_map, scale_factor, steps=10):
    window_name = 'Simulacion en Tiempo Real'
    stats_window_name = 'Estadisticas de Agentes'
    clock_window_name = 'Reloj Digital'
    pie_chart_window_name = 'Distribucion de Agentes'

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, simulation_map.shape[1], simulation_map.shape[0])
    cv2.namedWindow(stats_window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(clock_window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(pie_chart_window_name, cv2.WINDOW_NORMAL)

    stats_image = np.zeros((400, 400, 3), dtype=np.uint8)
    clock_image = np.zeros((100, 300, 3), dtype=np.uint8)

    global current_time

    for step in range(steps):
        # Copiar el mapa para dibujar los recorridos
        path_map = simulation_map.copy()

        for i in range(len(agent_states["positions"])):
            x, y = agent_states["positions"][i]
            need = agent_states["needs"][i]

            # Si el agente está en el WC, contar los minutos de permanencia
            if agent_states["wc_timer"][i] > 0:
                agent_states["wc_timer"][i] -= time_step_minutes
                if agent_states["wc_timer"][i] <= 0:
                    # El tiempo en WC ha terminado, restaurar la necesidad previa
                    agent_states["needs"][i] = agent_states["previous_need"][i]
                continue  # No hacer nada más por este ciclo

            # Obtener la hora actual
            current_hour = (current_time // 60) % 24

            # Establecer la necesidad según la hora y con pequeñas variaciones aleatorias
            if 22 <= current_hour or current_hour < 8:
                need = "rest"
            elif 8 <= current_hour < 9 or 13 <= current_hour < 14 or 20 <= current_hour < 21:
                if np.random.rand() < 0.1:  # 10% de variación aleatoria
                    if np.random.rand() < 0.5:
                        need = "food"
                    else:
                        need = "rest"
                else:
                    need = "food"
            else:
                if np.random.rand() < 0.05:
                    need = "wc"
                else:
                    need = "resting"

            agent_states["needs"][i] = need

            # Verificar si se ha alcanzado el objetivo actual
            if agent_states["targets"][i] and (x, y) == agent_states["targets"][i]:
                agent_states["paths"][i] = None
                agent_states["targets"][i] = None

            # Calcular un nuevo camino si no hay uno o el actual está agotado
            if agent_states["paths"][i] is None or len(agent_states["paths"][i]) == 0:
                if need == "rest":
                    target_position = agent_states["beds"][i]
                else:
                    target_color = {
                        "food": colors["yellow"],
                        "rest": colors["blue"],
                        "wc": colors["red"],
                        "resting": colors["cyan"]
                    }[need]

                    target_positions = np.argwhere(np.all(simulation_map == target_color, axis=-1))
                    if len(target_positions) > 0:
                        distances = [heuristic((x, y), tuple(pos * scale_factor)) for pos in target_positions]
                        nearest_index = np.argmin(distances)
                        target_position = tuple(target_positions[nearest_index] * scale_factor)

                path = a_star(simulation_map, (x, y), target_position, scale_factor)
                agent_states["paths"][i] = path if path else []
                agent_states["targets"][i] = target_position

            # Dibujar la polilínea del camino en la copia del mapa
            if agent_states["paths"][i]:
                points = np.array(agent_states["paths"][i], np.int32)
                points = points[:, [1, 0]] * scale_factor
                points = points.reshape((-1, 1, 2))
                cv2.polylines(path_map, [points], isClosed=False, color=colors["orange"], thickness=1)

            # Moverse al siguiente paso del camino calculado
            if agent_states["paths"][i]:
                next_position = agent_states["paths"][i].pop(0)
                agent_states["positions"][i] = next_position

                # Si el agente llega al WC, iniciar el temporizador
                if need == "wc" and tuple(next_position) == agent_states["targets"][i]:
                    agent_states["wc_timer"][i] = 10  # 10 minutos de permanencia en el WC
                    agent_states["previous_need"][i] = need

        # Dibujar agentes en el mapa
        temp_map = path_map.copy()
        for pos in agent_states["positions"]:
            cv2.rectangle(temp_map, (pos[1], pos[0]), (pos[1]+scale_factor-1, pos[0]+scale_factor-1), (0, 0, 0), -1)

        # Mostrar el estado actual en la ventana
        cv2.imshow(window_name, temp_map)

        # Actualizar la imagen de estadísticas
        stats_image[:] = (0, 0, 0)
        for i in range(len(agent_states["positions"])):
            stat_text = f"Agente {i}: Pos ({agent_states['positions'][i][1]},{agent_states['positions'][i][0]}), Necesidad: {agent_states['needs'][i]}"
            cv2.putText(stats_image, stat_text, (10, 20 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow(stats_window_name, stats_image)

        # Actualizar y mostrar la hora actual
        clock_image[:] = (0, 0, 0)
        time_str = minutes_to_time_str(current_time)
        cv2.putText(clock_image, time_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(clock_window_name, clock_image)

        # Crear la gráfica de tarta
        needs_counts = {
            "food": np.sum(np.array(agent_states["needs"]) == "food"),
            "rest": np.sum(np.array(agent_states["needs"]) == "rest"),
            "wc": np.sum(np.array(agent_states["needs"]) == "wc"),
            "resting": np.sum(np.array(agent_states["needs"]) == "resting")
        }
        labels = list(needs_counts.keys())
        sizes = list(needs_counts.values())
        colors_pie = ['yellow', 'blue', 'red', 'cyan']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Mantener la proporción de la gráfica

        # Guardar la gráfica en un archivo temporal
        plt.savefig('pie_chart.png')
        plt.close()

        # Leer la imagen de la gráfica y mostrarla en una ventana flotante
        pie_chart_image = cv2.imread('pie_chart.png')
        cv2.imshow(pie_chart_window_name, pie_chart_image)

        current_time = (current_time + time_step_minutes) % (24 * 60)  # Avanzar el tiempo en la simulación

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Ejecutar la simulación con el nuevo temporizador en WC
move_agents(agent_states, simulation_map, scale_factor, steps=240000)
