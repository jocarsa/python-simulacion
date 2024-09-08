import cv2
import numpy as np
import os
import heapq
import matplotlib
matplotlib.use('Agg')  # Cambiar el backend de matplotlib para evitar errores con tkinter
import matplotlib.pyplot as plt

# Cargar la imagen proporcionada
image_path = 'casas.png'  # Reemplaza con la ruta a tu imagen
image = cv2.imread(image_path)

# Escalar la imagen según el factor de escala, pero mantener la proporción de la resolución original
scale_factor = 8  # Escalamos por 8 para trabajar con sprites de 8x8
resized_image = cv2.resize(image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

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
    "white": (255, 255, 255), # píxeles blancos
    "black": (0, 0, 0)        # agentes
}

# Ruta donde se encuentran los sprites
sprite_folder = './sprites/'  # Carpeta donde están los sprites

# Cargar los sprites correspondientes
sprites = {
    "green": cv2.imread(os.path.join(sprite_folder, 'verde.png')),
    "magenta": cv2.imread(os.path.join(sprite_folder, 'magenta.png')),
    "yellow": cv2.imread(os.path.join(sprite_folder, 'amarillo.png')),
    "blue": cv2.imread(os.path.join(sprite_folder, 'azul.png')),
    "red": cv2.imread(os.path.join(sprite_folder, 'rojo.png')),
    "cyan": cv2.imread(os.path.join(sprite_folder, 'cyan.png')),
    "orange": cv2.imread(os.path.join(sprite_folder, 'naranja.png')),
    "gray": cv2.imread(os.path.join(sprite_folder, 'gris.png')),
    "dark_green": cv2.imread(os.path.join(sprite_folder, 'verdeoscuro.png')),
    "black": cv2.imread(os.path.join(sprite_folder, 'negro.png')),  # Sprite de los agentes
    "white": cv2.imread(os.path.join(sprite_folder, 'blanco.png'))  # Sprite para los píxeles blancos
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

# Identificar todas las posiciones de los parques (verde oscuro)
park_positions = np.argwhere(np.all(simulation_map == colors["dark_green"], axis=-1))

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
    agent_beds.append(tuple(bed_position))

    # Asignar un puesto de trabajo aleatorio a cada agente (pueden compartir puestos de trabajo)
    work_position = work_positions[np.random.choice(len(work_positions))]
    agent_workplaces.append(tuple(work_position))

    # Encontrar el punto de comida más cercano a la cama asignada
    distances_to_food = np.linalg.norm(food_positions - bed_position, axis=1)
    nearest_food_index = np.argmin(distances_to_food)
    nearest_food_position = food_positions[nearest_food_index]
    agent_food_positions.append(tuple(nearest_food_position))

agent_positions = np.array(agent_positions)

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
def a_star(simulation_map, start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
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
                if current_color in colors.values():  # Solo considerar colores válidos
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
    
    return None  # Retorna None si no hay un camino posible

# Lista de días de la semana, comenzando con sábado
days_of_week = [ "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday","Saturday"]

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

# Función para actualizar y mostrar la imagen del mapa con los sprites
def update_simulation_image(simulation_map, agent_states, sprites):
    new_image_size = (simulation_map.shape[1] * scale_factor, simulation_map.shape[0] * scale_factor)
    new_image = np.zeros((new_image_size[1], new_image_size[0], 3), dtype=np.uint8)

    for y in range(simulation_map.shape[0]):
        for x in range(simulation_map.shape[1]):
            pixel_color = tuple(simulation_map[y, x])
            
            # Encontrar el sprite correspondiente al color del píxel
            if pixel_color == colors["green"]:  # green
                sprite = sprites["green"]
            elif pixel_color == colors["magenta"]:  # magenta
                sprite = sprites["magenta"]
            elif pixel_color == colors["yellow"]:  # yellow
                sprite = sprites["yellow"]
            elif pixel_color == colors["blue"]:  # blue
                sprite = sprites["blue"]
            elif pixel_color == colors["red"]:  # red
                sprite = sprites["red"]
            elif pixel_color == colors["cyan"]:  # cyan
                sprite = sprites["cyan"]
            elif pixel_color == colors["gray"]:  # gray
                sprite = sprites["gray"]
            elif pixel_color == colors["dark_green"]:  # dark green
                sprite = sprites["dark_green"]
            elif pixel_color == colors["white"]:  # white
                sprite = sprites["white"]
            else:
                sprite = np.zeros((scale_factor, scale_factor, 3), dtype=np.uint8)  # Si no coincide con ningún color, usar un cuadro negro

            # Posición donde se colocará el sprite en la nueva imagen
            start_x = x * scale_factor
            start_y = y * scale_factor
            new_image[start_y:start_y+scale_factor, start_x:start_x+scale_factor] = sprite

    # Superponer las trayectorias A* (naranja) y los agentes (negro)
    for i in range(len(agent_states["positions"])):
        if agent_states["paths"][i]:  # Si hay un camino calculado
            for pos in agent_states["paths"][i]:
                start_x = pos[1] * scale_factor
                start_y = pos[0] * scale_factor
                # Superponer el sprite naranja sobre el sprite del suelo
                base_sprite = new_image[start_y:start_y+scale_factor, start_x:start_x+scale_factor]
                new_image[start_y:start_y+scale_factor, start_x:start_x+scale_factor] = cv2.addWeighted(base_sprite, 0.5, sprites["orange"], 0.5, 0)

        # Posición del agente
        pos = agent_states["positions"][i]
        start_x = pos[1] * scale_factor
        start_y = pos[0] * scale_factor
        # Superponer el sprite negro (agente) sobre el sprite del suelo
        base_sprite = new_image[start_y:start_y+scale_factor, start_x:start_x+scale_factor]
        new_image[start_y:start_y+scale_factor, start_x:start_x+scale_factor] = cv2.addWeighted(base_sprite, 0.5, sprites["black"], 0.5, 0)

    return new_image

# Función para dibujar agentes y trayectorias en la imagen original de baja resolución
def update_original_image(simulation_map, agent_states):
    original_image = simulation_map.copy()

    # Dibujar las trayectorias A* (naranja) y los agentes (negro) en la imagen original
    for i in range(len(agent_states["positions"])):
        if agent_states["paths"][i]:  # Si hay un camino calculado
            for pos in agent_states["paths"][i]:
                original_image[pos[0], pos[1]] = colors["orange"]

        # Posición del agente
        pos = agent_states["positions"][i]
        original_image[pos[0], pos[1]] = colors["black"]

    return original_image

# Función para mover agentes según necesidades usando A* y teniendo en cuenta los días de la semana
def move_agents(agent_states, simulation_map, scale_factor, sprites, steps=10):
    window_name = 'Simulacion en Tiempo Real'
    original_window_name = 'Mapa Original'
    stats_window_name = 'Estadisticas de Agentes'
    clock_window_name = 'Reloj Digital'
    pie_chart_window_name = 'Distribucion de Agentes'

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(original_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, simulation_map.shape[1] * scale_factor, simulation_map.shape[0] * scale_factor)
    cv2.resizeWindow(original_window_name, simulation_map.shape[1], simulation_map.shape[0])
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
                    wc_positions = np.argwhere(np.all(simulation_map == colors["red"], axis=-1))
                    if len(wc_positions) > 0:
                        distances_to_wc = np.linalg.norm(wc_positions - np.array([x, y]), axis=1)
                        nearest_wc_index = np.argmin(distances_to_wc)
                        target_position = tuple(wc_positions[nearest_wc_index])
                else:  # Resting
                    if np.random.rand() < 0.5 and len(park_positions) > 0:  # 50% de probabilidad de elegir un parque
                        # Elegir una posición aleatoria en el parque
                        target_position = tuple(park_positions[np.random.choice(len(park_positions))])
                    else:
                        # Mantener la lógica existente para descansar en áreas cyan
                        target_color = colors["cyan"]
                        resting_positions = np.argwhere(np.all(simulation_map == target_color, axis=-1))
                        if len(resting_positions) > 0:
                            distances = [heuristic((x, y), tuple(pos)) for pos in resting_positions]
                            nearest_index = np.argmin(distances)
                            target_position = tuple(resting_positions[nearest_index])

            agent_states["needs"][i] = need

            # Verificar si se ha alcanzado el objetivo actual
            if agent_states["targets"][i] and (x, y) == agent_states["targets"][i]:
                agent_states["paths"][i] = None
                agent_states["targets"][i] = None

            # Calcular un nuevo camino si no hay uno o el actual está agotado
            if agent_states["paths"][i] is None or len(agent_states["paths"][i]) == 0:
                path = a_star(simulation_map, (x, y), target_position)
                agent_states["paths"][i] = path if path else []
                agent_states["targets"][i] = target_position

            # Moverse al siguiente paso del camino calculado
            if agent_states["paths"][i]:
                next_position = agent_states["paths"][i].pop(0)
                agent_states["positions"][i] = next_position

                # Si el agente llega al WC, iniciar el temporizador
                if need == "wc" and tuple(next_position) == agent_states["targets"][i]:
                    agent_states["wc_timer"][i] = 60  # 60 segundos de permanencia en el WC (equivalente a 10 minutos de simulación)
                    agent_states["previous_need"][i] = need

        # Actualizar la imagen original con agentes y trayectorias
        original_image_with_agents = update_original_image(path_map, agent_states)

        # Actualizar la imagen de la simulación con los sprites y superposiciones
        simulation_image = update_simulation_image(path_map, agent_states, sprites)

        # Mostrar la imagen original (de baja resolución) con agentes y trayectorias
        cv2.imshow(original_window_name, original_image_with_agents)

        # Mostrar el estado actual en la ventana con los sprites
        cv2.imshow(window_name, simulation_image)

        # Actualizar la imagen de estadísticas
        stats_image[:] = (0, 0, 0)
        for i in range(len(agent_states["positions"])):
            stat_text = f"Agente {i}: Pos ({agent_states['positions'][i][1]},{agent_states['positions'][i][0]}), Necesidad: {agent_states['needs'][i]}"
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

        current_time_seconds = (current_time_seconds + time_step_seconds) % (12 * 31 * 86400)  # Avanzar el tiempo en la simulación, con límite de 12 meses

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Ejecutar la simulación con el tiempo ajustado y visualización mejorada
move_agents(agent_states, simulation_map, scale_factor, sprites, steps=240000)
