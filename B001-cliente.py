import socket
import json
import numpy as np
import heapq
import time
import cv2

# Configuración del cliente
server_ip = '127.0.0.1'
server_port = 9999

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))

# Parámetros de simulación
scale_factor = 1
terrain_layer = np.array(cv2.imread('casas.png'))
colors = {
    "green": (0, 255, 0),
    "magenta": (255, 0, 255),
    "yellow": (0, 255, 255),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "cyan": (255, 255, 0),
    "orange": (0, 165, 255),
    "gray": (127, 127, 127),
    "dark_green": (0, 200, 0)
}

walkable_positions = np.argwhere(np.all(terrain_layer == colors["green"], axis=-1))
bed_positions = np.argwhere(np.all(terrain_layer == colors["blue"], axis=-1))
work_positions = np.argwhere(np.all(terrain_layer == colors["gray"], axis=-1))
food_positions = np.argwhere(np.all(terrain_layer == colors["yellow"], axis=-1))
park_positions = np.argwhere(np.all(terrain_layer == colors["dark_green"], axis=-1))

# Estado inicial del agente
agent_id = np.random.randint(1000, 9999)
initial_position = walkable_positions[np.random.choice(len(walkable_positions))]
agent_position = tuple(initial_position * scale_factor)
agent_bed = tuple(bed_positions[np.random.choice(len(bed_positions))] * scale_factor)
agent_workplace = tuple(work_positions[np.random.choice(len(work_positions))] * scale_factor)
distances_to_food = np.linalg.norm(food_positions - np.array(agent_bed), axis=1)
nearest_food_index = np.argmin(distances_to_food)
agent_food = tuple(food_positions[nearest_food_index] * scale_factor)
agent_need = 'rest'
agent_path = []
agent_target = None

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
            return path[::-1]
        
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= neighbor[0] < terrain_layer.shape[0] and 0 <= neighbor[1] < terrain_layer.shape[1]:
                current_color = terrain_layer[neighbor[0], neighbor[1]].tolist()
                if current_color in [list(colors[key]) for key in ["green", "yellow", "blue", "red", "cyan", "gray", "dark_green"]]:
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
    
    return None

# Función para enviar el estado del agente al servidor
def send_agent_state():
    agent_data = {
        "id": int(agent_id),  # Asegurarse de que el ID es un entero
        "position": [int(agent_position[0]), int(agent_position[1])],  # Convertir cada valor de la posición a int
        "need": agent_need,  # El campo "need" ya es una cadena de texto, no necesita conversión
        "path": [[int(p[0]), int(p[1])] for p in agent_path],  # Convertir cada punto de la trayectoria
        "target_position": [int(agent_target[0]), int(agent_target[1])] if agent_target else None
    }
    message = json.dumps(agent_data).encode('utf-8')
    message_length = len(message)
    client_socket.sendall(message_length.to_bytes(4, 'big'))
    client_socket.sendall(message)

# Simulación del movimiento y comportamiento del agente
while True:
    try:
        # Esperar recibir el tiempo del servidor
        print("Esperando tiempo del servidor...")
        length_bytes = client_socket.recv(4)
        if not length_bytes:
            raise ValueError("No se pudo recibir la longitud del mensaje de tiempo.")
        
        message_length = int.from_bytes(length_bytes, 'big')
        server_time_data = client_socket.recv(message_length)
        if not server_time_data:
            raise ValueError("No se pudo recibir los datos de tiempo del servidor.")
        
        server_time = json.loads(server_time_data.decode('utf-8'))['current_time']
        print(f"Tiempo recibido del servidor: {server_time}")

        current_hour = (server_time // 3600) % 24

        if 22 <= current_hour or current_hour < 8:
            agent_need = "rest"
            agent_target = agent_bed
        elif 8 <= current_hour < 9:
            agent_need = "food"
            agent_target = agent_food
        elif 9 <= current_hour < 13:
            agent_need = "work"
            agent_target = agent_workplace
        elif 13 <= current_hour < 14:
            agent_need = "food"
            agent_target = agent_food
        elif 14 <= current_hour < 20:
            agent_need = "resting"
            agent_target = agent_bed
        elif 20 <= current_hour < 21:
            agent_need = "food"
            agent_target = agent_food
        elif 21 <= current_hour < 22:
            agent_need = "resting"
            agent_target = agent_bed

        if agent_target and agent_position != agent_target:
            print(f"Calculando camino a {agent_target} desde {agent_position}")
            agent_path = a_star(terrain_layer, agent_position, agent_target, scale_factor)
        
        if agent_path:
            print(f"Camino calculado: {agent_path}")
            agent_position = agent_path.pop(0)

        # Mostrar en la consola la información del agente
        print(f"Agente ID: {agent_id}, Posición actual: {agent_position}, Objetivo: {agent_target}, Necesidad: {agent_need}")
        
        send_agent_state()
        time.sleep(1)
    except Exception as e:
        print(f"Error durante la simulación: {e}")
        break

client_socket.close()
