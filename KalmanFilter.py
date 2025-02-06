import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def sim_new_gps_point(last_pos, process_noise): #simulate new GPS points base don last known pos
    step = np.random.normal(0, process_noise, size=2)
    new_pos = last_pos + step
    return new_pos

def kalman_filter(x_est_prev, P_prev, measurement, process_variance, measurement_variance): #update for streaming data
    #prediction update
    x_pred = x_est_prev
    P_pred = P_prev + process_variance
    
    #measurement update
    K = P_pred / (P_pred + measurement_variance)
    x_est_new = x_pred + K * (measurement - x_pred)
    P_new = (1 - K) * P_pred
    return x_est_new, P_new


initial_position = np.array([40.7128, -74.0060])  #coordinates
process_noise_level = 0.0005
measurement_variance = 0.0001
process_variance = 0.0001

#plot simulated
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
line_gps, = ax1.plot([], [], 'o-', label='Simulated GPS Path')
line_est, = ax1.plot([], [], 'g-', label='Kalman Filter Path', linewidth=2)
ax1.set_title('Real-Time GPS Tracking with Kalman Filter')
ax1.set_xlabel('Longitude (deg)')
ax1.set_ylabel('Latitude (deg)')
ax1.legend()
ax1.grid(True)

#error plot
error_line, = ax2.plot([], [], 'r-', label='Position Error')
ax2.set_title('Kalman Filter Position Error')
ax2.set_xlabel('Time Step (index)')
ax2.set_ylabel('Error (m)')
ax2.legend()
ax2.grid(True)

#initial states
last_pos = np.array(initial_position)
x_est = np.array(initial_position)
P = np.array([1.0, 1.0])  #initial covariance estimates
path_gps = [initial_position]
lat_est, lon_est = [initial_position[0]], [initial_position[1]]
errors = []

def init():
    line_gps.set_data([], [])
    line_est.set_data([], [])
    error_line.set_data([], [])
    return line_gps, line_est, error_line

def update(frame):
    global last_pos, x_est, P, path_gps, lat_est, lon_est, errors
    new_pos = sim_new_gps_point(last_pos, process_noise_level) #simulate
    last_pos = new_pos
    path_gps.append(new_pos)
    x_est, P = kalman_filter(x_est, P, new_pos, process_variance, measurement_variance) #update filter
    lat_est.append(x_est[0])
    lon_est.append(x_est[1])
    error = np.sqrt((x_est[0] - new_pos[0])**2 + (x_est[1] - new_pos[1])**2) #error calc
    errors.append(error)
    line_gps.set_data([p[1] for p in path_gps], [p[0] for p in path_gps])
    line_est.set_data(lon_est, lat_est)
    error_line.set_data(range(len(errors)), errors)
    ax1.relim()
    ax1.autoscale_view()
    ax2.set_xlim(0, len(errors))
    ax2.set_ylim(0, max(errors) + 0.0001)
    return line_gps, line_est, error_line,

ani = FuncAnimation(fig, update, init_func=init, frames=np.arange(200), blit=True, interval=100, repeat=False)

plt.show()
