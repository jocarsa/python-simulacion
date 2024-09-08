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
    "orange": (0, 165, 255),  # polilínea de recorrido
    "gray": (127, 127, 127)   # puestos de trabajo
}

# Creación del entorno de simulación
simulation_map = resized_image.copy()

# Identificar todas las posiciones pisables (verdes) en el mapa
walkable_positions = np.argwhere(np.all(simulation_map == colors["green"], axis=-1))

# Identificar todas las posiciones de las camas (azules)
bed_positions = np.argwhere(np.all(simulation_map == colors["blue"], axis=-1))

# Identificar todas las posiciones de los puestos de trabajo (grises)
work_positions = np.argwhere(np.all(simulation_map == colors["gray"], axis=-1))

# Identificar todas las posiciones de comida (amarillo)
food_positions = np.argwhere(np.all(simulation_map == colors["yellow"], axis=-1))

# Parámetros de la simulación
num_agents = 140  # Aumenta el número de agentes para ver múltiples recorridos

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
    if not np.all(simulation_map[initial_position[0], initial_position[1]] == colors["green"]):
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
                if current_color == list(colors["green"]) or current_color == list(colors["yellow"]) or current_color == list(colors["blue"]) or current_color == list(colors["red"]) or current_color == list(colors["cyan"]) or current_color == list(colors["gray"]):
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
    
    return None  # Retorna None si no hay un camino posible

# Definir el paso de tiempo en segundos por cada iteración de la simulación
time_step_seconds = 30
current_time_seconds = 0  # Tiempo actual en segundos desde las 00:00

# Función para convertir segundos a una cadena de fecha y hora en formato YYYY:MM:DD:HH:MM:SS
def seconds_to_date_time_str(seconds):
    days = seconds // 86400  # Un día tiene 86400 segundos
    years = 0000 + (days // (12 * 31))
    months = ((days // 31) % 12) + 1
    day_of_month = (days % 31) + 1
    hours = (seconds // 3600) % 24
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{years:04}:{months:02}:{day_of_month:02}:{hours:02}:{mins:02}:{secs:02}"

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
    clock_image = np.zeros((150, 700, 3), dtype=np.uint8)  # Aumentar el tamaño de la imagen del reloj para incluir las etiquetas

    global current_time_seconds

    for step in range(steps):
        # Copiar el mapa para dibujar los recorridos
        path_map = simulation_map.copy()

        for i in range(len(agent_states["positions"])):
            x, y = agent_states["positions"][i]
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

            # Establecer la necesidad según la hora y con pequeñas variaciones aleatorias
            if 22 <= current_hour or current_hour < 8:
                need = "rest"
            elif 8 <= current_hour < 9:
                need = "food"
            elif 9 <= current_hour < 13:
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
                    wc_positions = np.argwhere(np.all(simulation_map == colors["red"], axis=-1))
                    if len(wc_positions) > 0:
                        distances_to_wc = np.linalg.norm(wc_positions - np.array([x, y]), axis=1)
                        nearest_wc_index = np.argmin(distances_to_wc)
                        target_position = tuple(wc_positions[nearest_wc_index] * scale_factor)
                else:  # Resting
                    target_color = colors["cyan"]
                    resting_positions = np.argwhere(np.all(simulation_map == target_color, axis=-1))
                    if len(resting_positions) > 0:
                        distances = [heuristic((x, y), tuple(pos * scale_factor)) for pos in resting_positions]
                        nearest_index = np.argmin(distances)
                        target_position = tuple(resting_positions[nearest_index] * scale_factor)

            agent_states["needs"][i] = need

            # Verificar si se ha alcanzado el objetivo actual
            if agent_states["targets"][i] and (x, y) == agent_states["targets"][i]:
                agent_states["paths"][i] = None
                agent_states["targets"][i] = None

            # Calcular un nuevo camino si no hay uno o el actual está agotado
            if agent_states["paths"][i] is None or len(agent_states["paths"][i]) == 0:
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
                    agent_states["wc_timer"][i] = 60  # 60 segundos de permanencia en el WC (equivalente a 10 minutos de simulación)
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

        # Actualizar y mostrar la fecha y hora actual
        clock_image[:] = (0, 0, 0)
        date_time_str = seconds_to_date_time_str(current_time_seconds)
        labels = ["Year", "Month", "Day", "Hour", "Min", "Sec"]
        x_positions = [40, 150, 260, 370, 480, 590]
        for i, label in enumerate(labels):
            cv2.putText(clock_image, label, (x_positions[i], 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(clock_image, date_time_str.split(":")[i], (x_positions[i], 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(clock_window_name, clock_image)

        # Crear la gráfica de tarta
        needs_counts = {
            "food": np.sum(np.array(agent_states["needs"]) == "food"),
            "rest": np.sum(np.array(agent_states["needs"]) == "rest"),
            "wc": np.sum(np.array(agent_states["needs"]) == "wc"),
            "resting": np.sum(np.array(agent_states["needs"]) == "resting"),
            "work": np.sum(np.array(agent_states["needs"]) == "work")
        }
        labels = list(needs_counts.keys())
        sizes = list(needs_counts.values())
        colors_pie = ['yellow', 'blue', 'red', 'cyan', 'gray']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Mantener la proporción de la gráfica

        # Guardar la gráfica en un archivo temporal
        plt.savefig('pie_chart.png')
        plt.close()

        # Leer la imagen de la gráfica y mostrarla en una ventana flotante
        pie_chart_image = cv2.imread('pie_chart.png')
        cv2.imshow(pie_chart_window_name, pie_chart_image)

        current_time_seconds = (current_time_seconds + time_step_seconds) % (12 * 31 * 86400)  # Avanzar el tiempo en la simulación, con límite de 12 meses

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Ejecutar la simulación con el tiempo ajustado y visualización mejorada
move_agents(agent_states, simulation_map, scale_factor, steps=240000)
