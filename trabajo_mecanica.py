import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


# 1. PARÁMETROS DEL SISTEMA
M = 500.0     # Masa de la pluma (kg)
m = 100.0      # Masa del péndulo/gancho (kg)
L = 10.0      # Longitud de la pluma (m)
l = 2.0       # Longitud de la cuerda del péndulo (m)
g = 9.81      # Aceleración de la gravedad (m/s^2)
H = 50.0      # Altura de la grúa sobre el suelo (m)
I_cm = (1.0 / 12.0) * M * L**2

# 2. ECUACIONES DE MOVIMIENTO (SISTEMA MATRICIAL)
def system_dynamics(t, state):
    # Desempaquetar estado
    x_cm, y_cm, phi, theta, vx, vy, w_phi, w_theta = state

    # Matriz de Masa M_mat * [ddot_x, ddot_y, ddot_phi, ddot_theta]^T = RHS
    M_mat = np.array([
        [M + m, 0.0, 0.0, m * l * np.cos(theta)],
        [0.0, M + m, 0.0, -m * l * np.sin(theta)],
        [0.0, 0.0, I_cm, 0.0],
        [m * l * np.cos(theta), -m * l * np.sin(theta), 0.0, m * l**2]
    ])

    RHS = np.array([
        m * l * (w_theta**2) * np.sin(theta),
        (M + m) * g + m * l * (w_theta**2) * np.cos(theta),
        0.0,
        -m * g * l * np.sin(theta)
    ])

    # Resolver aceleraciones
    accelerations = np.linalg.solve(M_mat, RHS)
    ax_cm, ay_cm, a_phi, a_theta = accelerations

    return [vx, vy, w_phi, w_theta, ax_cm, ay_cm, a_phi, a_theta]

# Evento para detener la simulación cuando la pluma o el péndulo toquen el suelo (y = H)
def ground_impact(t, state):
    x_cm, y_cm, phi, theta = state[0], state[1], state[2], state[3]
    
    # Coordenadas y de los dos extremos de la pluma
    y_end1 = y_cm - 0.5 * L * np.sin(phi)
    y_end2 = y_cm + 0.5 * L * np.sin(phi)
    # Coordenada y de la masa del péndulo
    y_hook = y_cm + l * np.cos(theta)
    
    # Detener cuando el punto más bajo alcance el suelo (y = H)
    lowest_point = max(y_end1, y_end2, y_hook)
    return H - lowest_point

ground_impact.terminal = True
ground_impact.direction = -1

# 3. CONDICIONES INICIALES Y SIMULACIÓN
# [x_cm, y_cm, phi, theta, vx_cm, vy_cm, w_phi, w_theta]
x_cm_0 = L / 2.0   # Posición inicial del CM de la pluma
y_cm_0 = 0.0       # Altura inicial (origen en el punto de anclaje)
phi_0= 0.0         # Ángulo inicial de la pluma
theta_0 = 1.2      # Ángulo inicial de desviación del péndulo en radianes 
vx_0 = 0.0         # Velocidad incial en el eje x
vy_0 = 0.0         # Velocidad incial en el eje y
w_phi_0 = 0.5      # Velocidad angular inicial para la pluma (rad/s)
w_theta_0 = 0.0    # Velocidad angular incial para el pendulo (rad/s)

state_0 = [x_cm_0, y_cm_0, phi_0, theta_0, vx_0, vy_0, w_phi_0, w_theta_0]

t_span = (0, 5.0)
t_eval = np.linspace(0, 5.0, 500)

sol = solve_ivp(system_dynamics, t_span, state_0, t_eval=t_eval, 
                events=ground_impact, rtol=1e-8, atol=1e-8)

t_arr = sol.t
x_cm_arr = sol.y[0]
y_cm_arr = sol.y[1]
phi_arr = sol.y[2]
theta_arr = sol.y[3]


# 4. VISUALIZACIÓN Y ANIMACIÓN
fig = plt.figure(figsize=(14, 7))
# Subplot 1: Representación gráfica / Animación espacial
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_xlim(-L, L * 2)
ax1.set_ylim(H + 2, -2) # Invertido porque y positivo es hacia abajo
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m) [0 = Anclaje inicial, H = Suelo]")
ax1.set_title("Simulación en vivo de la caída de la pluma")
ax1.plot([0,0], [0, H], color='gold', lw=3, linestyle='-', label="H")
ax1.axhline(H, color='brown', lw=3, label="Suelo")
ax1.axhline(0, color='gray', linestyle='--', label="Nivel inicial")
ax1.grid(True)

line_boom, = ax1.plot([], [], 'o-', lw=4, color='darkorange', label="Pluma")
line_pendulum, = ax1.plot([], [], 'o-', lw=2, color='blue', label="Péndulo")
trace_cm, = ax1.plot([], [], 'r--', alpha=0.5, label="Trayectoria CM")

ax1.legend(loc="upper left")

# Subplot 2: Gráficas de evolución temporal de variables
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(t_arr, x_cm_arr, label=r"$x_{cm}$ (m)", color='red')
ax2.plot(t_arr, y_cm_arr, label=r"$y_{cm}$ (m)", color='green')
ax2.set_ylabel("Posición (m)")
ax2.set_title("Evolución de $x_{cm}$ e $y_{cm}$")
ax2.grid(True)
ax2.legend()

ax3 = fig.add_subplot(2, 2, 4)
ax3.plot(t_arr, np.degrees(phi_arr), label=r"$\phi$ (grados)", color='purple')
ax3.plot(t_arr, np.degrees(theta_arr), label=r"$\theta$ (grados)", color='teal')
ax3.set_xlabel("Tiempo (s)")
ax3.set_ylabel("Ángulo (grados)")
ax3.set_title(r"Evolución de ángulos $\phi$ y $\theta$")
ax3.grid(True)
ax3.legend()

plt.tight_layout()

# Función de actualización para la animación
def update(frame):
    xc = x_cm_arr[frame]
    yc = y_cm_arr[frame]
    p = phi_arr[frame]
    th = theta_arr[frame]

    # Extremos de la pluma
    x1 = xc - 0.5 * L * np.cos(p)
    y1 = yc - 0.5 * L * np.sin(p)
    x2 = xc + 0.5 * L * np.cos(p)
    y2 = yc + 0.5 * L * np.sin(p)

    # Posición de la masa del péndulo
    xp = xc + l * np.sin(th)
    yp = yc + l * np.cos(th)

    line_boom.set_data([x1, x2], [y1, y2])
    line_pendulum.set_data([xc, xp], [yc, yp])
    trace_cm.set_data(x_cm_arr[:frame], y_cm_arr[:frame])

    return line_boom, line_pendulum, trace_cm

ani = FuncAnimation(fig, update, frames=len(t_arr), interval=20, blit=True, repeat=False)

plt.show()