# --- Author: Nihara Randini ---

import numpy as np
import matplotlib.pyplot as plt

# --- Environment and Robot Parameters ---
MAP_SIZE = 10.0  # meters (map is a 10x10 square)
NUM_PARTICLES = 5000
NUM_LANDMARKS = 5
MOTION_NOISE = 0.1  # Standard deviation for motion uncertainty (position/heading)
SENSOR_NOISE = 0.5  # Standard deviation for sensor measurement uncertainty (distance)
DT = 0.1  # Time step for simulation
SIMULATION_STEPS = 100

# Landmark positions (a known feature list in the environment)
LANDMARKS = np.array([
    [2.0, 2.0],
    [8.0, 8.0],
    [2.0, 8.0],
    [8.0, 2.0],
    [5.0, 5.0]
])

# Global state variable for the actual robot pose (Ground Truth)
robot_pose = np.array([1.0, 1.0, 0.0])  # [x, y, theta]

# --- Helper Functions ---

def wrap_angle(angle):
    """Wraps an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def initialize_particles(num_particles, map_size):
    """Initializes particles with random poses and uniform weights."""
    particles = np.zeros((num_particles, 3))  # [x, y, theta]
    particles[:, 0] = np.random.uniform(0, map_size, num_particles)  # x
    particles[:, 1] = np.random.uniform(0, map_size, num_particles)  # y
    particles[:, 2] = np.random.uniform(-np.pi, np.pi, num_particles)  # theta
    weights = np.ones(num_particles) / num_particles
    return particles, weights

def move_robot_noisy(pose, u_linear, u_angular, dt, noise):
    """Updates the actual robot pose with command and adds noise."""
    v_noisy = u_linear + np.random.normal(0, noise)
    omega_noisy = u_angular + np.random.normal(0, noise / 2.0) # Lower noise on rotation

    pose[0] += v_noisy * np.cos(pose[2]) * dt
    pose[1] += v_noisy * np.sin(pose[2]) * dt
    pose[2] += omega_noisy * dt
    pose[2] = wrap_angle(pose[2])

    # Simple boundary check
    pose[0] = np.clip(pose[0], 0, MAP_SIZE)
    pose[1] = np.clip(pose[1], 0, MAP_SIZE)
    
    return pose

def sense_landmarks(pose, landmarks, noise):
    """Calculates the noisy distance measurements from the robot to each landmark."""
    measurements = []
    robot_x, robot_y, _ = pose
    
    for lx, ly in landmarks:
        # True distance
        true_dist = np.hypot(lx - robot_x, ly - robot_y)
        # Noisy measurement
        noisy_dist = true_dist + np.random.normal(0, noise)
        measurements.append(noisy_dist)
        
    return np.array(measurements)

# --- Particle Filter Core Functions ---

def predict_particles(particles, u_linear, u_angular, dt, noise):
    """Applies motion model (Forward Kinematics) to all particles with noise."""
    
    # Add noise to movement commands for each particle
    v_noisy = u_linear + np.random.normal(0, noise, len(particles))
    omega_noisy = u_angular + np.random.normal(0, noise / 2.0, len(particles))

    # Update particle positions
    particles[:, 0] += v_noisy * np.cos(particles[:, 2]) * dt
    particles[:, 1] += v_noisy * np.sin(particles[:, 2]) * dt
    particles[:, 2] += omega_noisy * dt
    
    # Wrap heading angle and enforce boundaries
    particles[:, 2] = wrap_angle(particles[:, 2])
    particles[:, 0] = np.clip(particles[:, 0], 0, MAP_SIZE)
    particles[:, 1] = np.clip(particles[:, 1], 0, MAP_SIZE)
    
    return particles

def update_weights(particles, weights, actual_measurement, landmarks, noise):
    """Calculates the likelihood of each particle based on the sensor measurement."""
    
    for i in range(len(particles)):
        p_x, p_y, _ = particles[i]
        
        # 1. Calculate expected distance from particle to each landmark
        expected_distances = np.hypot(landmarks[:, 0] - p_x, landmarks[:, 1] - p_y)
        
        # 2. Calculate the difference between actual (robot) and expected (particle) measurements
        error = actual_measurement - expected_distances
        
        # 3. Calculate the Gaussian likelihood (Probability Density Function)
        # L = (1 / sqrt(2*pi*sigma^2)) * exp(-0.5 * error^2 / sigma^2)
        # We can ignore the normalization term (1 / sqrt(2*pi*sigma^2)) as we normalize the weights later
        likelihoods = np.exp(-0.5 * (error / noise)**2)
        
        # The total likelihood (weight) is the product of likelihoods for all landmarks
        weights[i] *= np.prod(likelihoods)

    # Normalize weights so they sum to 1, avoiding division by zero
    sum_weights = np.sum(weights)
    if sum_weights == 0:
        # If all weights are near zero, re-initialize to avoid filter divergence
        weights = np.ones_like(weights) / len(weights)
    else:
        weights /= sum_weights
        
    return weights

def resample_particles(particles, weights):
    """Resamples particles based on their weights (Roulette Wheel Resampling)."""
    
    # Generate random indices based on weights distribution
    indices = np.random.choice(
        len(particles), 
        size=len(particles), 
        replace=True, 
        p=weights
    )
    # Create new particle set by sampling the old set
    particles = particles[indices]
    
    # Reset weights to uniform after resampling
    weights = np.ones(len(particles)) / len(particles)
    return particles, weights

def get_estimate(particles, weights):
    """Calculates the best estimated pose (weighted average)."""
    # Use weighted average for the estimate
    x_est = np.average(particles[:, 0], weights=weights)
    y_est = np.average(particles[:, 1], weights=weights)
    # Average of angles needs special care, but simple mean is often sufficient here
    theta_est = np.average(particles[:, 2], weights=weights)
    
    return np.array([x_est, y_est, wrap_angle(theta_est)])

# --- Main Simulation Loop ---

def run_mcl_simulation():
    global robot_pose
    
    particles, weights = initialize_particles(NUM_PARTICLES, MAP_SIZE)
    
    # Command velocities (simple square path or continuous turn)
    V_CMD, OMEGA_CMD = 0.5, 0.1 

    # History for plotting
    history_robot = [robot_pose.copy()]
    history_estimate = [get_estimate(particles, weights).copy()]

    for step in range(SIMULATION_STEPS):
        
        # --- 1. Robot Motion (Ground Truth) ---
        robot_pose = move_robot_noisy(robot_pose, V_CMD, OMEGA_CMD, DT, MOTION_NOISE)

        # --- 2. Sensing (Actual Measurement) ---
        measurement = sense_landmarks(robot_pose, LANDMARKS, SENSOR_NOISE)

        # --- 3. Particle Filter Prediction (Motion Update) ---
        particles = predict_particles(particles, V_CMD, OMEGA_CMD, DT, MOTION_NOISE)

        # --- 4. Particle Filter Correction (Weight Update) ---
        weights = update_weights(particles, weights, measurement, LANDMARKS, SENSOR_NOISE)

        # --- 5. Particle Filter Resampling ---
        particles, weights = resample_particles(particles, weights)

        # --- 6. Localization Estimate ---
        estimate = get_estimate(particles, weights)
        
        # Store history
        history_robot.append(robot_pose.copy())
        history_estimate.append(estimate.copy())
        
        # Optionally, plot every 10 steps to visualize the cloud collapse
        if step % 20 == 0:
            plot_state(particles, robot_pose, estimate, LANDMARKS, step)

    # Final Plotting
    plot_results(history_robot, history_estimate, LANDMARKS)


def plot_state(particles, robot_pose, estimate, landmarks, step):
    """Plots the current particle cloud, robot, and estimate."""
    plt.figure(figsize=(8, 8))
    plt.title(f"MCL Step: {step}. Particles: {len(particles)}")
    
    # 1. Plot Landmarks
    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='*', color='red', s=200, label='Landmarks')
    
    # 2. Plot Particles (the cloud)
    plt.scatter(particles[:, 0], particles[:, 1], marker='.', color='gray', alpha=0.1, label='Particles')
    
    # 3. Plot Robot (Ground Truth)
    plt.plot(robot_pose[0], robot_pose[1], marker='o', color='blue', markersize=10, label='Robot (Actual)')
    
    # 4. Plot Estimate (Mean of the particles)
    plt.plot(estimate[0], estimate[1], marker='s', color='green', markersize=10, label='Estimate (MCL)')
    
    plt.xlim(0, MAP_SIZE)
    plt.ylim(0, MAP_SIZE)
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.legend()
    plt.grid(True)
    plt.pause(0.001) # Small pause to allow plot update
    plt.close()


def plot_results(history_robot, history_estimate, landmarks):
    """Plots the final path comparison."""
    history_robot = np.array(history_robot)
    history_estimate = np.array(history_estimate)
    
    plt.figure(figsize=(10, 8))
    plt.title("Particle Filter Localization: Actual vs. Estimate Path")
    
    # Plot Landmarks
    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='*', color='red', s=200, label='Landmarks')
    
    # Plot Paths
    plt.plot(history_robot[:, 0], history_robot[:, 1], 'b-', linewidth=2, label='Robot (Ground Truth)')
    plt.plot(history_estimate[:, 0], history_estimate[:, 1], 'g--', linewidth=2, label='MCL Estimate')
    
    # Highlight start/end points
    plt.plot(history_robot[0, 0], history_robot[0, 1], 'bo', markersize=8) # Start
    plt.plot(history_robot[-1, 0], history_robot[-1, 1], 'b^', markersize=8) # End
    
    plt.xlim(0, MAP_SIZE)
    plt.ylim(0, MAP_SIZE)
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_mcl_simulation()