import cv2
import numpy as np
import heapq
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Cargar las imágenes de las plantas
image_path_plant0 = 'casas.png'
image_path_plant1 = 'casasplanta1.png'

# Cargar las imágenes de las plantas
image_plant0 = cv2.imread(image_path_plant0)
image_plant1 = cv2.imread(image_path_plant1)

# Escalar las imágenes según el factor de escala, manteniendo la proporción original
scale_factor = 1
resized_image_plant0 = cv2.resize(image_plant0, (image_plant0.shape[1] * scale_factor, image_plant0.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST)
resized_image_plant1 = cv2.resize(image_plant1, (image_plant1.shape[1] * scale_factor, image_plant1.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST)

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
    "dark_green": (0, 200, 0),# parques
    "elevator": (255, 127, 0) # ascensores
}

# Creación del entorno de simulación para ambas plantas
terrain_layers = [resized_image_plant0.copy(), resized_image_plant1.copy()]

# Identificar posiciones en las diferentes capas
def identify_positions(terrain_layer, color):
    return np.argwhere(np.all(terrain_layer == color, axis=-1))

# Identificar todas las posiciones pisables (verdes) en el mapa
walkable_positions_plant0 = identify_positions(terrain_layers[0], colors["green"])
walkable_positions_plant1 = identify_positions(terrain_layers[1], colors["green"])

# Identificar todas las posiciones de las camas (azules)
bed_positions = identify_positions(terrain_layers[0], colors["blue"])

# Identificar todas las posiciones de los puestos de trabajo (grises)
work_positions = identify_positions(terrain_layers[0], colors["gray"])

# Identificar todas las posiciones de comida (amarillo)
food_positions = identify_positions(terrain_layers[0], colors["yellow"])

# Identificar todas las posiciones de los parques (verde oscuro)
park_positions = identify_positions(terrain_layers[0], colors["dark_green"])

# Identificar todas las posiciones de ascensores (naranja)
elevator_positions_plant0 = identify_positions(terrain_layers[0], colors["elevator"])
elevator_positions_plant1 = identify_positions(terrain_layers[1], colors["elevator"])

# Parámetros de la simulación
num_agents = 100

# Asegurar que los agentes solo se spawneen en posiciones pisables (verdes)
def find_nearest_green_position(position, walkable_positions):
    distances = np.linalg.norm(walkable_positions - position, axis=1)
    nearest_index = np.argmin(distances)
    return walkable_positions[nearest_index]

# Inicializar agentes
agent_positions = []
agent_floors = []
agent_beds = []
agent_workplaces = []
agent_food_positions = []

for _ in range(num_agents):
    initial_position = walkable_positions_plant0[np.random.choice(len(walkable_positions_plant0))]
    agent_positions.append(initial_position)
    agent_floors.append(0)  # Todos los agentes inician en la planta 0
    
    bed_position = bed_positions[np.random.choice(len(bed_positions))]
    agent_beds.append((0, tuple(bed_position * scale_factor)))
    
    work_position = work_positions[np.random.choice(len(work_positions))]
    agent_workplaces.append((0, tuple(work_position * scale_factor)))
    
    distances_to_food = np.linalg.norm(food_positions - bed_position, axis=1)
    nearest_food_index = np.argmin(distances_to_food)
    nearest_food_position = food_positions[nearest_food_index]
    agent_food_positions.append((0, tuple(nearest_food_position * scale_factor)))

agent_positions = np.array(agent_positions) * scale_factor

# Inicializar posiciones y necesidades de los agentes
agent_states = {
    "positions": agent_positions,
    "floors": agent_floors,
    "needs": np.random.choice(["food", "rest", "wc", "resting"], size=num_agents),
    "paths": [None] * num_agents,
    "targets": [None] * num_agents,
    "beds": agent_beds,
    "workplaces": agent_workplaces,
    "food_positions": agent_food_positions,
    "wc_timer": [0] * num_agents,
    "previous_need": [None] * num_agents
}

# Función de heurística para A* (distancia Manhattan)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Algoritmo A* para encontrar el camino en una planta específica
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
            return path[::-1]
        
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= neighbor[0] < terrain_layer.shape[0] and 0 <= neighbor[1] < terrain_layer.shape[1]:
                current_color = terrain_layer[neighbor[0], neighbor[1]].tolist()
                if current_color in [list(colors["green"]), list(colors["yellow"]), list(colors["blue"]), list(colors["red"]), list(colors["cyan"]), list(colors["gray"]), list(colors["dark_green"]), list(colors["elevator"])]:
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
    
    return None

# Función para convertir segundos a una cadena de fecha, hora y día de la semana en formato YYYY:MM:DD:Day:HH:MM:SS
def seconds_to_date_time_str(seconds):
    days = seconds // 86400
    years = 2024 + (days // (12 * 31))
    months = ((days // 31) % 12) + 1
    day_of_month = (days % 31) + 1
    day_of_week = days_of_week[days % 7]
    hours = (seconds // 3600) % 24
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{years:04}:{months:02}:{day_of_month:02}:{day_of_week}:{hours:02}:{mins:02}:{secs:02}"

# Mover agentes según necesidades
def move_agents(agent_states, terrain_layers, scale_factor, steps=10):
    window_name_plant0 = 'Simulacion Planta 0'
    window_name_plant1 = 'Simulacion Planta 1'
    stats_window_name = 'Estadisticas de Agentes'
    clock_window_name = 'Reloj Digital'
    pie_chart_window_name = 'Distribucion de Agentes'

    cv2.namedWindow(window_name_plant0, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_plant0, terrain_layers[0].shape[1], terrain_layers[0].shape[0])
    cv2.namedWindow(window_name_plant1, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_plant1, terrain_layers[1].shape[1], terrain_layers[1].shape[0])
    cv2.namedWindow(stats_window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(clock_window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(pie_chart_window_name, cv2.WINDOW_NORMAL)

    stats_image = np.zeros((400, 400, 3), dtype=np.uint8)
    clock_image = np.zeros((150, 700, 3), dtype=np.uint8)

    global current_time_seconds
    current_time_seconds = 0  # Inicializar tiempo

    agents_layers = [np.zeros_like(terrain_layers[0]), np.zeros_like(terrain_layers[1])]

    for step in range(steps):
        for floor in range(2):
            agents_layers[floor].fill(0)

        for i in range(len(agent_states["positions"])):
            x, y = agent_states["positions"][i]
            floor = agent_states["floors"][i]
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
                target_position = agent_states["beds"][i][1]
                target_floor = agent_states["beds"][i][0]
            else:
                if np.random.rand() < 0.05:
                    need = "wc"
                
                if need == "work":
                    target_position = agent_states["workplaces"][i][1]
                    target_floor = agent_states["workplaces"][i][0]
                elif need == "food":
                    target_position = agent_states["food_positions"][i][1]
                    target_floor = agent_states["food_positions"][i][0]
                elif need == "wc":
                    wc_positions = identify_positions(terrain_layers[floor], colors["red"])
                    if len(wc_positions) > 0:
                        distances_to_wc = np.linalg.norm(wc_positions - np.array([x, y]), axis=1)
                        nearest_wc_index = np.argmin(distances_to_wc)
                        nearest_wc_position = wc_positions[nearest_wc_index]
                        target_position = tuple(nearest_wc_position * scale_factor)
                        target_floor = floor
                else:
                    if np.random.rand() < 0.5 and len(park_positions) > 0:
                        target_position = tuple(park_positions[np.random.choice(len(park_positions))] * scale_factor)
                        target_floor = 0
                    else:
                        target_color = colors["cyan"]
                        resting_positions = identify_positions(terrain_layers[floor], target_color)
                        if len(resting_positions) > 0:
                            distances = [heuristic((x, y), tuple(pos * scale_factor)) for pos in resting_positions]
                            nearest_index = np.argmin(distances)
                            target_position = tuple(resting_positions[nearest_index] * scale_factor)
                            target_floor = floor

            agent_states["needs"][i] = need

            if agent_states["targets"][i] and (x, y) == agent_states["targets"][i] and floor == target_floor:
                agent_states["paths"][i] = None
                agent_states["targets"][i] = None

            if agent_states["paths"][i] is None or len(agent_states["paths"][i]) == 0:
                if floor != target_floor:
                    if floor == 0:
                        nearest_elevator = find_nearest_green_position((x, y), elevator_positions_plant0)
                    else:
                        nearest_elevator = find_nearest_green_position((x, y), elevator_positions_plant1)
                    path_to_elevator = a_star(terrain_layers[floor], (x, y), tuple(nearest_elevator), scale_factor)
                    path_to_goal = a_star(terrain_layers[target_floor], tuple(nearest_elevator), target_position, scale_factor)
                    agent_states["paths"][i] = path_to_elevator + [(target_position[0], target_position[1])] + path_to_goal
                else:
                    path = a_star(terrain_layers[floor], (x, y), target_position, scale_factor)
                    agent_states["paths"][i] = path if path else []
                agent_states["targets"][i] = target_position

            if agent_states["paths"][i]:
                points = np.array(agent_states["paths"][i], np.int32)
                points = points[:, [1, 0]] * scale_factor
                points = points.reshape((-1, 1, 2))
                cv2.polylines(agents_layers[floor], [points], isClosed=False, color=colors["orange"], thickness=1)

            if agent_states["paths"][i]:
                next_position = agent_states["paths"][i].pop(0)
                agent_states["positions"][i] = next_position
                if need == "wc" and tuple(next_position) == agent_states["targets"][i]:
                    agent_states["wc_timer"][i] = 60
                    agent_states["previous_need"][i] = need

            for pos in agent_states["positions"]:
                cv2.rectangle(agents_layers[floor], (pos[1], pos[0]), (pos[1]+scale_factor-1, pos[0]+scale_factor-1), (0, 0, 0), -1)

        for floor in range(2):
            mask = cv2.cvtColor(agents_layers[floor], cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            terrain_layer_bg = cv2.bitwise_and(terrain_layers[floor], terrain_layers[floor], mask=mask_inv)
            agents_layer_fg = cv2.bitwise_and(agents_layers[floor], agents_layers[floor], mask=mask)
            combined_image = cv2.add(terrain_layer_bg, agents_layer_fg)
            for pos in agent_states["positions"]:
                cv2.rectangle(combined_image, (pos[1], pos[0]), (pos[1]+scale_factor-1, pos[0]+scale_factor-1), (0, 0, 0), -1)
            if floor == 0:
                cv2.imshow(window_name_plant0, combined_image)
            else:
                cv2.imshow(window_name_plant1, combined_image)

        stats_image[:] = (0, 0, 0)
        for i in range(len(agent_states["positions"])):
            stat_text = f"Agente {i}: Pos ({agent_states['positions'][i][1]},{agent_states['positions'][i][0]}), Planta: {agent_states['floors'][i]}, Necesidad: {agent_states['needs'][i]}"
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
                    "floor": agent_states["floors"][i],
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
time_step_seconds = 10  # Definido en el código para la simulación
days_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

move_agents(agent_states, terrain_layers, scale_factor, steps=240000)
