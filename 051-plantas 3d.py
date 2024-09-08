import cv2
import numpy as np
import heapq
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Cargar las imágenes de ambas plantas
image_path_0 = 'casas.png'
image_path_1 = 'casasplanta1.png'

# Cargar ambas plantas
image_0 = cv2.imread(image_path_0)
image_1 = cv2.imread(image_path_1)

# Escalar las imágenes según el factor de escala, pero mantener la proporción de la resolución original
scale_factor = 1
resized_image_0 = cv2.resize(image_0, (image_0.shape[1] * scale_factor, image_0.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST)
resized_image_1 = cv2.resize(image_1, (image_1.shape[1] * scale_factor, image_1.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST)

# Combinar ambas plantas en una sola estructura tridimensional
terrain_layers = np.stack((resized_image_0, resized_image_1), axis=2)

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

# Identificar las posiciones de los ascensores (color RGB(255,127,0))
elevator_positions = np.argwhere(np.all(terrain_layers == (255, 127, 0), axis=-1))

# Función para encontrar el ascensor más cercano
def find_nearest_elevator(position, elevator_positions):
    distances = np.linalg.norm(elevator_positions - position, axis=1)
    nearest_index = np.argmin(distances)
    return elevator_positions[nearest_index]

# Parámetros de la simulación
num_agents = 10  # Número de agentes

# Asignación de agentes, camas y puestos de trabajo
agent_positions = []
agent_beds = []
agent_workplaces = []
agent_food_positions = []

for _ in range(num_agents):
    # Elegir aleatoriamente la planta de nacimiento (0 o 1)
    initial_z = np.random.choice([0, 1])
    
    # Elegir una posición aleatoria en la planta de nacimiento
    initial_position = np.argwhere(np.all(terrain_layers[:, :, initial_z] == colors["green"], axis=-1))
    initial_position = initial_position[np.random.choice(len(initial_position))]
    
    agent_positions.append((initial_position[0], initial_position[1], initial_z))
    
    # Asignar una cama en la misma planta o en la otra planta
    bed_z = np.random.choice([0, 1])
    bed_position = np.argwhere(np.all(terrain_layers[:, :, bed_z] == colors["blue"], axis=-1))
    bed_position = bed_position[np.random.choice(len(bed_position))]
    agent_beds.append((bed_position[0], bed_position[1], bed_z))
    
    # Asignar un puesto de trabajo en la misma planta o en la otra planta
    work_z = np.random.choice([0, 1])
    work_position = np.argwhere(np.all(terrain_layers[:, :, work_z] == colors["gray"], axis=-1))
    work_position = work_position[np.random.choice(len(work_position))]
    agent_workplaces.append((work_position[0], work_position[1], work_z))
    
    # Encontrar el punto de comida más cercano a la cama asignada
    food_positions_in_bed_layer = np.argwhere(np.all(terrain_layers[:, :, bed_z] == colors["yellow"], axis=-1))
    distances_to_food = np.linalg.norm(food_positions_in_bed_layer - np.array([bed_position[0], bed_position[1]]), axis=1)
    nearest_food_index = np.argmin(distances_to_food)
    nearest_food_position = food_positions_in_bed_layer[nearest_food_index]
    agent_food_positions.append((nearest_food_position[0], nearest_food_position[1], bed_z))

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
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

# Algoritmo A* para encontrar el camino en 3D
def a_star_3d(terrain_layers, start, goal, scale_factor):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    directions = [(0, scale_factor, 0), (0, -scale_factor, 0), (scale_factor, 0, 0), (-scale_factor, 0, 0)]
    
    while open_list:
        current = heapq.heappop(open_list)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Devuelve el camino desde el inicio hasta el objetivo
        
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1], current[2])
            if 0 <= neighbor[0] < terrain_layers.shape[0] and 0 <= neighbor[1] < terrain_layers.shape[1]:
                current_color = terrain_layers[neighbor[0], neighbor[1], neighbor[2]].tolist()
                if current_color == list(colors["green"]) or current_color == list(colors["yellow"]) or current_color == list(colors["blue"]) or current_color == list(colors["red"]) or current_color == list(colors["cyan"]) or current_color == list(colors["gray"]) or current_color == list(colors["dark_green"]):
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
        
        # Incluir la posibilidad de moverse entre plantas si hay un ascensor en la posición actual
        if any(np.all(np.array(current) == np.array(e), axis=-1) for e in elevator_positions):
            for dz in [-1, 1]:  # Solo puede moverse entre la planta 0 y 1
                neighbor = (current[0], current[1], current[2] + dz)
                if 0 <= neighbor[2] < terrain_layers.shape[2]:  # Verificar que no salga del rango de plantas
                    current_color = terrain_layers[neighbor[0], neighbor[1], neighbor[2]].tolist()
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

# Función para mover agentes según necesidades usando A* y teniendo en cuenta los días de la semana
def move_agents_3d(agent_states, terrain_layers, scale_factor, steps=10):
    window_name_0 = 'Planta 0'
    window_name_1 = 'Planta 1'
    stats_window_name = 'Estadisticas de Agentes'
    clock_window_name = 'Reloj Digital'
    pie_chart_window_name = 'Distribucion de Agentes'

    cv2.namedWindow(window_name_0, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_0, terrain_layers.shape[1], terrain_layers.shape[0])
    cv2.namedWindow(window_name_1, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_1, terrain_layers.shape[1], terrain_layers.shape[0])
    cv2.namedWindow(stats_window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(clock_window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(pie_chart_window_name, cv2.WINDOW_NORMAL)

    stats_image = np.zeros((400, 400, 3), dtype=np.uint8)
    clock_image = np.zeros((150, 700, 3), dtype=np.uint8)  # Aumentar el tamaño de la imagen del reloj para incluir las etiquetas

    global current_time_seconds

    # Crear capas de agentes para cada planta que se actualizarán en cada iteración
    agents_layer_0 = np.zeros_like(terrain_layers[:, :, 0])
    agents_layer_1 = np.zeros_like(terrain_layers[:, :, 0])

    for step in range(steps):
        # Reiniciar las capas de agentes
        agents_layer_0.fill(0)
        agents_layer_1.fill(0)

        for i in range(len(agent_states["positions"])):
            x, y, z = agent_states["positions"][i]
            need = agent_states["needs"][i]

            # Si el agente está en el WC, contar los minutos de permanencia
            if agent_states["wc_timer"][i] > 0:
                agent_states["wc_timer"][i] -= time_step_seconds // 10
                if agent_states["wc_timer"][i] <= 0:
                    # El tiempo en WC ha terminado, restaurar la necesidad previa
                    agent_states["needs"][i] = agent_states["previous_need"][i]
                continue  # No hacer nada más por este ciclo

            # Obtener la hora actual en horas, minutos y segundos
            current_hour = (current_time_seconds // 3600) % 24
            day_of_week = (current_time_seconds // 86400) % 7  # Día de la semana (0 = Saturday, 1 = Sunday, ...)

            # Establecer la necesidad según la hora y con pequeñas variaciones aleatorias
            if 22 <= current_hour or current_hour < 8:
                need = "rest"
            elif 8 <= current_hour < 9:
                need = "food"
            elif 9 <= current_hour < 13:
                if days_of_week[day_of_week] in ["Saturday", "Sunday"]:
                    need = "resting"  # En fines de semana, los agentes descansan en lugar de trabajar
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

            # Si es hora de dormir, los agentes deben ir a su propia cama y no ir al WC
            if need == "rest" and (22 <= current_hour or current_hour < 8):
                target_position = agent_states["beds"][i]
            else:
                # Asignar la necesidad de ir al WC de forma ocasional (excepto cuando están durmiendo)
                if np.random.rand() < 0.05:  # 5% de probabilidad de necesitar ir al WC
                    need = "wc"
                
                if need == "work":
                    target_position = agent_states["workplaces"][i]
                elif need == "food":
                    target_position = agent_states["food_positions"][i]
                elif need == "wc":
                    # Encontrar el WC más cercano
                    wc_positions = np.argwhere(np.all(terrain_layers[:, :, z] == colors["red"], axis=-1))
                    if len(wc_positions) > 0:
                        distances_to_wc = np.linalg.norm(wc_positions - np.array([x, y]), axis=1)
                        nearest_wc_index = np.argmin(distances_to_wc)
                        target_position = (wc_positions[nearest_wc_index][0], wc_positions[nearest_wc_index][1], z)
                else:  # Resting
                    if np.random.rand() < 0.5 and len(park_positions) > 0:  # 50% de probabilidad de elegir un parque
                        # Elegir una posición aleatoria en el parque
                        target_position = (park_positions[np.random.choice(len(park_positions))][0], park_positions[np.random.choice(len(park_positions))][1], z)
                    else:
                        # Mantener la lógica existente para descansar en áreas cyan
                        target_color = colors["cyan"]
                        resting_positions = np.argwhere(np.all(terrain_layers[:, :, z] == target_color, axis=-1))
                        if len(resting_positions) > 0:
                            distances = [heuristic((x, y, z), (pos[0], pos[1], z)) for pos in resting_positions]
                            nearest_index = np.argmin(distances)
                            target_position = (resting_positions[nearest_index][0], resting_positions[nearest_index][1], z)

            agent_states["needs"][i] = need

            # Verificar si se ha alcanzado el objetivo actual
            if agent_states["targets"][i] and (x, y, z) == agent_states["targets"][i]:
                agent_states["paths"][i] = None
                agent_states["targets"][i] = None

            # Calcular un nuevo camino si no hay uno o el actual está agotado
            if agent_states["paths"][i] is None or len(agent_states["paths"][i]) == 0:
                path = a_star_3d(terrain_layers, (x, y, z), target_position, scale_factor)
                agent_states["paths"][i] = path if path else []
                agent_states["targets"][i] = target_position

            # Dibujar la polilínea del camino en la capa de agentes
            if agent_states["paths"][i]:
                points = np.array(agent_states["paths"][i], np.int32)
                points = points[:, [1, 0]] * scale_factor
                points = points.reshape((-1, 1, 2))
                if z == 0:
                    cv2.polylines(agents_layer_0, [points], isClosed=False, color=colors["orange"], thickness=1)
                else:
                    cv2.polylines(agents_layer_1, [points], isClosed=False, color=colors["orange"], thickness=1)

            # Moverse al siguiente paso del camino calculado
            if agent_states["paths"][i]:
                next_position = agent_states["paths"][i].pop(0)
                agent_states["positions"][i] = next_position

                # Si el agente llega al WC, iniciar el temporizador
                if need == "wc" and tuple(next_position) == agent_states["targets"][i]:
                    agent_states["wc_timer"][i] = 60  # 60 segundos de permanencia en el WC (equivalente a 10 minutos de simulación)
                    agent_states["previous_need"][i] = need

        # Dibujar agentes en la capa de agentes
        for pos in agent_states["positions"]:
            if pos[2] == 0:
                cv2.rectangle(agents_layer_0, (pos[1], pos[0]), (pos[1]+scale_factor-1, pos[0]+scale_factor-1), (0, 0, 0), -1)
            else:
                cv2.rectangle(agents_layer_1, (pos[1], pos[0]), (pos[1]+scale_factor-1, pos[0]+scale_factor-1), (0, 0, 0), -1)

        # Crear una máscara de transparencia para superponer la capa de agentes sobre la capa de terreno
        mask_0 = cv2.cvtColor(agents_layer_0, cv2.COLOR_BGR2GRAY)
        _, mask_0 = cv2.threshold(mask_0, 1, 255, cv2.THRESH_BINARY)
        mask_inv_0 = cv2.bitwise_not(mask_0)

        mask_1 = cv2.cvtColor(agents_layer_1, cv2.COLOR_BGR2GRAY)
        _, mask_1 = cv2.threshold(mask_1, 1, 255, cv2.THRESH_BINARY)
        mask_inv_1 = cv2.bitwise_not(mask_1)

        # Mantener los píxeles de la capa de terreno donde la máscara es negra
        terrain_layer_bg_0 = cv2.bitwise_and(terrain_layers[:, :, 0], terrain_layers[:, :, 0], mask=mask_inv_0)
        terrain_layer_bg_1 = cv2.bitwise_and(terrain_layers[:, :, 1], terrain_layers[:, :, 1], mask=mask_inv_1)

        # Superponer la capa de agentes donde la máscara es blanca
        agents_layer_fg_0 = cv2.bitwise_and(agents_layer_0, agents_layer_0, mask=mask_0)
        agents_layer_fg_1 = cv2.bitwise_and(agents_layer_1, agents_layer_1, mask=mask_1)

        # Combinar ambas capas
        combined_image_0 = cv2.add(terrain_layer_bg_0, agents_layer_fg_0)
        combined_image_1 = cv2.add(terrain_layer_bg_1, agents_layer_fg_1)

        # **Dibujar nuevamente los agentes en negro sobre la imagen combinada**
        for pos in agent_states["positions"]:
            if pos[2] == 0:
                cv2.rectangle(combined_image_0, (pos[1], pos[0]), (pos[1]+scale_factor-1, pos[0]+scale_factor-1), (0, 0, 0), -1)
            else:
                cv2.rectangle(combined_image_1, (pos[1], pos[0]), (pos[1]+scale_factor-1, pos[0]+scale_factor-1), (0, 0, 0), -1)

        # Mostrar el estado actual en las ventanas de cada planta
        cv2.imshow(window_name_0, combined_image_0)
        cv2.imshow(window_name_1, combined_image_1)

        # Actualizar la imagen de estadísticas
        stats_image[:] = (0, 0, 0)
        for i in range(len(agent_states["positions"])):
            stat_text = f"Agente {i}: Pos ({agent_states['positions'][i][1]},{agent_states['positions'][i][0]}, Planta {agent_states['positions'][i][2]}), Necesidad: {agent_states['needs'][i]}"
            cv2.putText(stats_image, stat_text, (10, 20 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow(stats_window_name, stats_image)

        # Actualizar y mostrar la fecha, hora y día de la semana actual
        clock_image[:] = (0, 0, 0)
        date_time_str = seconds_to_date_time_str(current_time_seconds)
        labels = ["Year", "Month", "Day", "Day of Week", "Hour", "Min", "Sec"]
        x_positions = [20, 110, 200, 290, 430, 540, 650]
        for i, label in enumerate(labels):
            cv2.putText(clock_image, label, (x_positions[i], 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(clock_image, date_time_str.split(":")[i], (x_positions[i], 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(clock_window_name, clock_image)

        # Crear la gráfica de tarta, asegurando que cada agente solo cuente una vez
        needs_counts = {
            "moving": 0,  # Agentes que están en movimiento
            "food": 0,
            "rest": 0,
            "wc": 0,
            "resting": 0,
            "work": 0
        }

        for i in range(len(agent_states["positions"])):
            if agent_states["paths"][i]:  # Si el agente está en movimiento
                needs_counts["moving"] += 1
            else:  # Si el agente ha llegado a su destino
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
        ax.axis('equal')  # Mantener la proporción de la gráfica

        # Guardar la gráfica en un archivo temporal
        plt.savefig('pie_chart.png')
        plt.close()

        # Leer la imagen de la gráfica y mostrarla en una ventana flotante
        pie_chart_image = cv2.imread('pie_chart.png')
        cv2.imshow(pie_chart_window_name, pie_chart_image)

        # Guardar el estado de los agentes en un archivo JSON
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

        current_time_seconds = (current_time_seconds + time_step_seconds) % (12 * 31 * 86400)  # Avanzar el tiempo en la simulación, con límite de 12 meses

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Ejecutar la simulación
move_agents_3d(agent_states, terrain_layers, scale_factor, steps=240000)
