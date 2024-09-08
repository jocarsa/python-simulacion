import cv2
import numpy as np

# Cargar la imagen proporcionada
image_path = 'casas.png'  # Reemplaza con la ruta a tu imagen
image = cv2.imread(image_path)

# Ampliar la imagen para que cada píxel sea un bloque de 4x4
scale_factor = 4
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
    "needs": np.random.choice(["food", "rest", "wc"], size=num_agents)
}

# Función para mover agentes según necesidades
def move_agents(agent_states, simulation_map, scale_factor, steps=10):
    for step in range(steps):
        for i in range(len(agent_states["positions"])):
            x, y = agent_states["positions"][i]
            need = agent_states["needs"][i]

            # Buscar el color correspondiente según la necesidad
            target_color = {
                "food": colors["yellow"],
                "rest": colors["blue"],
                "wc": colors["red"]
            }[need]

            # Movimiento aleatorio simple por ahora (puede ser mejorado)
            direction = np.random.choice(["up", "down", "left", "right"])
            if direction == "up" and y > 0:
                y -= scale_factor
            elif direction == "down" and y < simulation_map.shape[0] - scale_factor:
                y += scale_factor
            elif direction == "left" and x > 0:
                x -= scale_factor
            elif direction == "right" and x < simulation_map.shape[1] - scale_factor:
                x += scale_factor
            
            # Verificar si el nuevo lugar es pisable y adecuado
            current_color = simulation_map[y, x].tolist()
            if current_color == list(colors["green"]) or current_color == list(target_color):
                agent_states["positions"][i] = [x, y]
            
            # Si alcanzó su objetivo, asignar una nueva necesidad aleatoria
            if current_color == list(target_color):
                agent_states["needs"][i] = np.random.choice(["food", "rest", "wc"])

        # Dibujar agentes en el mapa
        temp_map = simulation_map.copy()
        for pos in agent_states["positions"]:
            cv2.rectangle(temp_map, (pos[0], pos[1]), (pos[0]+scale_factor-1, pos[1]+scale_factor-1), (0, 0, 0), -1)

        # Mostrar el estado actual en una ventana
        cv2.imshow('Simulacion en Tiempo Real', temp_map)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Simular un día (aquí simplificado a unos pocos pasos)
move_agents(agent_states, simulation_map, scale_factor, steps=2400)  # Simulación de 240 pasos (aprox. 24 horas simplificadas)
