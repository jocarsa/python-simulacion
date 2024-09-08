import socket
import threading
import cv2
import numpy as np
import json
import time

# Cargar la imagen proporcionada
image_path = 'casas.png'
image = cv2.imread(image_path)

# Escalar la imagen según el factor de escala, pero mantener la proporción de la resolución original
scale_factor = 1
resized_image = cv2.resize(image, (image.shape[1] * scale_factor, image.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST)

# Definir los colores en BGR
colors = {
    "green": (0, 255, 0),
    "magenta": (255, 0, 255),
    "yellow": (0, 255, 255),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "cyan": (255, 255, 0),
    "orange": (0, 165, 255),
    "gray": (127, 127, 127),
    "dark_green": (0, 200, 0),
    "agent_color": (0, 0, 0),  # Color para los agentes (negro)
    "path_color": (0, 165, 255)  # Color para la trayectoria (naranja)
}

# Creación del entorno de simulación
terrain_layer = resized_image.copy()

# Almacenar posiciones de los agentes conectados
agent_states = {}

# Lista de días de la semana
days_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# Función para manejar la conexión con un cliente
def handle_client(client_socket, client_address):
    global agent_states
    agent_id = None
    try:
        while True:
            length_bytes = client_socket.recv(4)
            if not length_bytes:
                break
            message_length = int.from_bytes(length_bytes, 'big')
            data = client_socket.recv(message_length)
            if not data:
                break
            agent_data = json.loads(data.decode('utf-8'))
            agent_id = agent_data['id']
            agent_states[agent_id] = agent_data
            agent_states[agent_id]['last_update'] = time.time()
            agent_states[agent_id]['client_socket'] = client_socket  # Almacena el socket del cliente

    except (json.JSONDecodeError, ConnectionError) as e:
        print(f"Error con el cliente {client_address}: {e}")
    finally:
        if agent_id in agent_states:
            del agent_states[agent_id]
        client_socket.close()

# Iniciar el servidor de sockets
def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 9999))
    server.listen(5)
    print("Servidor esperando conexiones en el puerto 9999...")

    while True:
        client_socket, client_address = server.accept()
        print(f"Conexión establecida con {client_address}")
        client_handler = threading.Thread(target=handle_client, args=(client_socket, client_address))
        client_handler.start()

# Función para enviar el tiempo actual a todos los clientes conectados
def send_time_to_clients():
    while True:
        current_time = int(time.time())  # Obtén el tiempo actual en segundos desde Epoch
        time_data = json.dumps({"current_time": current_time}).encode('utf-8')
        for agent_id, state in agent_states.items():
            try:
                client_socket = state['client_socket']
                message_length = len(time_data)
                client_socket.sendall(message_length.to_bytes(4, 'big'))
                client_socket.sendall(time_data)
            except Exception as e:
                print(f"Error enviando el tiempo al agente {agent_id}: {e}")
        time.sleep(1)  # Envía el tiempo cada segundo

# Función para convertir segundos a una cadena de fecha, hora y día de la semana
def seconds_to_date_time_str(seconds):
    days = seconds // 86400  # Un día tiene 86400 segundos
    years = (days // (12 * 31))
    months = ((days // 31) % 12) + 1
    day_of_month = (days % 31) + 1
    day_of_week = days_of_week[days % 7]
    hours = (seconds // 3600) % 24
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{years:04}:{months:02}:{day_of_month:02}:{day_of_week}:{hours:02}:{mins:02}:{secs:02}"

# Ejecutar la simulación y pintar los agentes
def run_simulation():
    global agent_states

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
    clock_image = np.zeros((150, 700, 3), dtype=np.uint8)

    start_time = time.time()

    while True:
        current_time_seconds = time.time() - start_time
        agents_layer = np.zeros_like(terrain_layer)

        # Eliminar agentes inactivos (desconectados)
        current_time = time.time()
        inactive_agents = [agent_id for agent_id, state in agent_states.items() if current_time - state['last_update'] > 10]
        for agent_id in inactive_agents:
            del agent_states[agent_id]

        # Dibujar a cada agente y su trayectoria en el mapa
        for agent_id, state in agent_states.items():
            pos = state['position']
            path = state.get('path', None)
            
            # Dibujar la trayectoria si existe
            if path and len(path) > 1:
                points = np.array(path, np.int32)
                points = points[:, [1, 0]] * scale_factor
                points = points.reshape((-1, 1, 2))
                cv2.polylines(agents_layer, [points], isClosed=False, color=colors["path_color"], thickness=1) 
            
            # Dibujar al agente como un pequeño cuadrado para mayor visibilidad
            x, y = pos
            square_size = 1  # Tamaño del cuadrado (3x3 píxeles)
            top_left = (max(y - square_size, 0), max(x - square_size, 0))
            bottom_right = (min(y + square_size, terrain_layer.shape[1] - 1), min(x + square_size, terrain_layer.shape[0] - 1))
            cv2.rectangle(agents_layer, top_left, bottom_right, colors["agent_color"], -1)

        combined_image = cv2.addWeighted(terrain_layer, 1, agents_layer, 1, 0)

        cv2.imshow(window_name, combined_image)

        # Actualizar las estadísticas
        stats_image[:] = (0, 0, 0)
        for i, (agent_id, state) in enumerate(agent_states.items()):
            stat_text = f"Agente {agent_id}: Pos ({state['position'][1]}, {state['position'][0]}), Necesidad: {state['need']}"
            cv2.putText(stats_image, stat_text, (10, 20 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow(stats_window_name, stats_image)

        # Actualizar el reloj
        clock_image[:] = (0, 0, 0)
        date_time_str = seconds_to_date_time_str(int(current_time_seconds))
        labels = ["Year", "Month", "Day", "Day of Week", "Hour", "Min", "Sec"]
        x_positions = [20, 110, 200, 290, 430, 540, 650]
        for i, label in enumerate(labels):
            cv2.putText(clock_image, label, (x_positions[i], 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(clock_image, date_time_str.split(":")[i], (x_positions[i], 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(clock_window_name, clock_image)

        # Simular avance del tiempo
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Iniciar el servidor y la simulación
if __name__ == "__main__":
    threading.Thread(target=start_server, daemon=True).start()
    threading.Thread(target=send_time_to_clients, daemon=True).start()
    run_simulation()
