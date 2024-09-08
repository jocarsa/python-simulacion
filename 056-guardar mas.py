import cv2
import numpy as np
import heapq
import json
import matplotlib
matplotlib.use('Agg')
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
    "orange": (0, 165, 255),  # polilínea de recorrido
    "gray": (127, 127, 127),  # puestos de trabajo
    "dark_green": (0, 200, 0) # parques
}

# Creación del entorno de simulación
terrain_layer = resized_image.copy()

# Identificar todas las posiciones pisables (verdes) en el mapa
walkable_positions = np.argwhere(np.all(terrain_layer == colors["green"], axis=-1))

# Identificar todas las posiciones de las camas (azules)
bed_positions = np.argwhere(np.all(terrain_layer == colors["blue"], axis=-1))

# Identificar todas las posiciones de los puestos de trabajo (grises)
work_positions = np.argwhere(np.all(terrain_layer == colors["gray"], axis=-1))

# Identificar todas las posiciones de comida (amarillo)
food_positions = np.argwhere(np.all(terrain_layer == colors["yellow"], axis=-1))

# Identificar todas las posiciones de los parques (verde oscuro)
park_positions = np.argwhere(np.all(terrain_layer == colors["dark_green"], axis=-1))

# Parámetros de la simulación
num_agents = 1000  # Aumenta el número de agentes para ver múltiples recorridos

# Asegurar que los agentes solo se spawneen en posiciones pisables (verdes)
def find_nearest_green_position(position, walkable_positions):
    distances = np.linalg.norm(walkable_positions - position, axis=1)
    nearest_index = np.argmin(distances)
    return walkable_positions[nearest_index]

agent_positions = []
agent_beds = []  # Para almacenar la cama asignada a cada agente
agent_workplaces = []  # Para almacenar el puesto de trabajo asignado a cada agente
agent_food_positions = []  # Para almacenar el punto de comida más cercano a cada cama

for _ in range(num_agents):
    initial_position = walkable_positions[np.random.choice(len(walkable_positions))]
    if not np.all(terrain_layer[initial_position[0], initial_position[1]] == colors["green"]):
        initial_position = find_nearest_green_position(initial_position, walkable_positions)
    agent_positions.append(initial_position)
    
    # Asignar una cama aleatoria a cada agente (cada agente tiene su propia cama)
    bed_position = bed_positions[np.random.choice(len(bed_positions))]
    agent_beds.append(tuple(bed_position * scale_factor))

    # Asignar un puesto de trabajo aleatorio a cada agente (pueden compartir puestos de trabajo)
    work_position = work_positions[np.random.choice(len(work_positions))]
    agent_workplaces.append(tuple(work_position * scale_factor))

    # Encontrar el punto de comida más cercano a la cama asignada
    distances_to_food = np.linalg.norm(food_positions - bed_position, axis=1)
    nearest_food_index = np.argmin(distances_to_food)
    nearest_food_position = food_positions[nearest_food_index]
    agent_food_positions.append(tuple(nearest_food_position * scale_factor))

agent_positions = np.array(agent_positions) * scale_factor

# Inicializar posiciones y necesidades de los agentes
agent_states = {
    "positions": agent_positions,
    "needs": np.random.choice(["food", "rest", "wc", "resting"], size=num_agents),
    "paths": [None] * num_agents,  # Para almacenar el camino calculado por A*
    "targets": [None] * num_agents,  # Para almacenar el objetivo actual
    "beds": agent_beds,  # Cama asignada a cada agente
    "workplaces": agent_workplaces,  # Puesto de trabajo asignado a cada agente
    "food_positions": agent_food_positions,  # Punto de comida más cercano a cada cama
    "wc_timer": [0] * num_agents,  # Para almacenar el tiempo restante en WC
    "previous_need": [None] * num_agents  # Para almacenar la necesidad previa
}

# Función de heurística para A* (distancia Manhattan)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Algoritmo A* para encontrar el camino
def a_star(terrain_layer, start, goal, scale_factor):
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
            if 0 <= neighbor[0] < terrain_layer.shape[0] and 0 <= neighbor[1] < terrain_layer.shape[1]:
                current_color = terrain_layer[neighbor[0], neighbor[1]].tolist()
                if current_color == list(colors["green"]) or current_color == list(colors["yellow"]) or current_color == list(colors["blue"]) or current_color == list(colors["red"]) or current_color == list(colors["cyan"]) or current_color == list(colors["gray"]) or current_color == list(colors["dark_green"]):
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
    
    return None  # Retorna None si no hay un camino posible

# Lista de días de la semana, comenzando con sábado
days_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# Definir el paso de tiempo en segundos por cada iteración de la simulación
time_step_seconds = 10
current_time_seconds = 0  # Tiempo actual en segundos desde las 00:00

# Función para convertir segundos a una cadena de fecha, hora y día de la semana en formato YYYY:MM:DD:Day:HH:MM:SS
def seconds_to_date_time_str(seconds):
    days = seconds // 86400  # Un día tiene 86400 segundos
    years = 2024 + (days // (12 * 31))
    months = ((days // 31) % 12) + 1
    day_of_month = (days % 31) + 1
    day_of_week = days_of_week[days % 7]
    hours = (seconds // 3600) % 24
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{years:04}:{months:02}:{day_of_month:02}:{day_of_week}:{hours:02}:{mins:02}:{secs:02}"

def move_agents(agent_states, terrain_layer, scale_factor, steps=10):
    window_name = 'Simulacion en Tiempo Real'
    stats_window_name = 'Estadisticas de Agentes'
    clock_window_name = 'Reloj Digital'
    pie_chart_window_name = 'Distribucion de Agentes'

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, terrain_layer.shape[1], terrain_layer.shape[0])
    cv2.namedWindow(stats_window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(clock_window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(pie_chart_window_name, cv2.WINDOW_NORMAL)

    stats_image = np.zeros((400, 400, 3), dtype=np.uint8)
    clock_image = np.zeros((150, 700, 3), dtype=np.uint8)  # Aumentar el tamaño de la imagen del reloj para incluir las etiquetas

    global current_time_seconds

    agents_layer = np.zeros_like(terrain_layer)

    for step in range(steps):
        agents_layer.fill(0)

        for i in range(len(agent_states["positions"])):
            x, y = agent_states["positions"][i]
            need = agent_states["needs"][i]

            if agent_states["wc_timer"][i] > 0:
                agent_states["wc_timer"][i] -= time_step_seconds // 10
                if agent_states["wc_timer"][i] <= 0:
                    agent_states["needs"][i] = agent_states["previous_need"][i]
                continue

            current_hour = (current_time_seconds // 3600) % 24
            day_of_week = (current_time_seconds // 86400) % 7

            if 22 <= current_hour or current_hour < 8:
                need = "rest"
            elif 8 <= current_hour < 9:
                need = "food"
            elif 9 <= current_hour < 13:
                if days_of_week[day_of_week] in ["Saturday", "Sunday"]:
                    need = "resting"
                else:
                    need = "work"
            elif 13 <= current_hour < 14:
                need = "food"
            elif 14 <= current_hour < 20:
                need = "resting"
            elif 20 <= current_hour < 21:
                need = "food"
            elif 21 <= current_hour < 22:
                need = "resting"

            if need == "rest" and (22 <= current_hour or current_hour < 8):
                target_position = agent_states["beds"][i]
            else:
                if np.random.rand() < 0.05:
                    need = "wc"

                if need == "work":
                    target_position = agent_states["workplaces"][i]
                elif need == "food":
                    target_position = agent_states["food_positions"][i]
                elif need == "wc":
                    wc_positions = np.argwhere(np.all(terrain_layer == colors["red"], axis=-1))
                    if len(wc_positions) > 0:
                        distances_to_wc = np.linalg.norm(wc_positions - np.array([x, y]), axis=1)
                        nearest_wc_index = np.argmin(distances_to_wc)
                        target_position = tuple(wc_positions[nearest_wc_index] * scale_factor)
                else:
                    if np.random.rand() < 0.5 and len(park_positions) > 0:
                        target_position = tuple(park_positions[np.random.choice(len(park_positions))] * scale_factor)
                    else:
                        target_color = colors["cyan"]
                        resting_positions = np.argwhere(np.all(terrain_layer == target_color, axis=-1))
                        if len(resting_positions) > 0:
                            distances = [heuristic((x, y), tuple(pos * scale_factor)) for pos in resting_positions]
                            nearest_index = np.argmin(distances)
                            target_position = tuple(resting_positions[nearest_index] * scale_factor)

            agent_states["needs"][i] = need

            if agent_states["targets"][i] and (x, y) == agent_states["targets"][i]:
                agent_states["paths"][i] = None
                agent_states["targets"][i] = None

            if agent_states["paths"][i] is None or len(agent_states["paths"][i]) == 0:
                if agent_states["targets"][i] != target_position:  # Solo recalcular si el objetivo ha cambiado
                    path = a_star(terrain_layer, (x, y), target_position, scale_factor)
                    agent_states["paths"][i] = path if path else []
                    agent_states["targets"][i] = target_position

            if agent_states["paths"][i]:
                points = np.array(agent_states["paths"][i], np.int32)
                points = points[:, [1, 0]] * scale_factor
                points = points.reshape((-1, 1, 2))
                cv2.polylines(agents_layer, [points], isClosed=False, color=colors["orange"], thickness=1)

            if agent_states["paths"][i]:
                next_position = agent_states["paths"][i].pop(0)
                agent_states["positions"][i] = next_position

                if need == "wc" and tuple(next_position) == agent_states["targets"][i]:
                    agent_states["wc_timer"][i] = 60
                    agent_states["previous_need"][i] = need

        for pos in agent_states["positions"]:
            cv2.rectangle(agents_layer, (pos[1], pos[0]), (pos[1]+scale_factor-1, pos[0]+scale_factor-1), (0, 0, 0), -1)

        mask = cv2.cvtColor(agents_layer, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        terrain_layer_bg = cv2.bitwise_and(terrain_layer, terrain_layer, mask=mask_inv)
        agents_layer_fg = cv2.bitwise_and(agents_layer, agents_layer, mask=mask)
        combined_image = cv2.add(terrain_layer_bg, agents_layer_fg)

        for pos in agent_states["positions"]:
            cv2.rectangle(combined_image, (pos[1], pos[0]), (pos[1]+scale_factor-1, pos[0]+scale_factor-1), (0, 0, 0), -1)

        cv2.imshow(window_name, combined_image)

        stats_image[:] = (0, 0, 0)
        for i in range(len(agent_states["positions"])):
            stat_text = f"Agente {i}: Pos ({agent_states['positions'][i][1]},{agent_states['positions'][i][0]}), Necesidad: {agent_states['needs'][i]}"
            cv2.putText(stats_image, stat_text, (10, 20 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow(stats_window_name, stats_image)

        clock_image[:] = (0, 0, 0)
        date_time_str = seconds_to_date_time_str(current_time_seconds)
        labels = ["Year", "Month", "Day", "Day of Week", "Hour", "Min", "Sec"]
        x_positions = [20, 110, 200, 290, 430, 540, 650]
        for i, label in enumerate(labels):
            cv2.putText(clock_image, label, (x_positions[i], 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(clock_image, date_time_str.split(":")[i], (x_positions[i], 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(clock_window_name, clock_image)

        needs_counts = {
            "moving": 0,
            "food": 0,
            "rest": 0,
            "wc": 0,
            "resting": 0,
            "work": 0
        }

        for i in range(len(agent_states["positions"])):
            if agent_states["paths"][i]:
                needs_counts["moving"] += 1
            else:
                need = agent_states["needs"][i]
                if need == "food":
                    needs_counts["food"] += 1
                elif need == "rest":
                    needs_counts["rest"] += 1
                elif need == "wc":
                    needs_counts["wc"] += 1
                elif need == "resting":
                    needs_counts["resting"] += 1
                elif need == "work":
                    needs_counts["work"] += 1

        labels = list(needs_counts.keys())
        sizes = list(needs_counts.values())
        colors_pie = ['orange', 'yellow', 'blue', 'red', 'cyan', 'gray']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')

        plt.savefig('pie_chart.png')
        plt.close()

        pie_chart_image = cv2.imread('pie_chart.png')
        cv2.imshow(pie_chart_window_name, pie_chart_image)

        agent_data = {
            "agents": [
                {
                    "id": i,
                    "position": agent_states["positions"][i].tolist(),
                    "need": agent_states["needs"][i]
                } for i in range(num_agents)
            ]
        }
        with open('agent_positions.json', 'w') as json_file:
            json.dump(agent_data, json_file, indent=4)

        current_time_seconds = (current_time_seconds + time_step_seconds) % (12 * 31 * 86400)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# Ejecutar la simulación con el tiempo ajustado y visualización mejorada
move_agents(agent_states, terrain_layer, scale_factor, steps=240000)
