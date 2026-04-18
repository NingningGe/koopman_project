from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.universal_robots.tasks import PickPlace
from omni.isaac.universal_robots.controllers import PickPlaceController
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core import SimulationContext
import numpy as np
import time

my_world = World(stage_units_in_meters=1.0, physics_dt=1/1000)
my_task = PickPlace()
my_world.add_task(my_task)
simulation_context = SimulationContext()

my_world.reset()
task_params = my_task.get_params()
my_ur = my_world.scene.get_object(task_params["robot_name"]["value"])
my_ur.disable_gravity()

articulation_controller = my_ur.get_articulation_controller()
articulation_controller.set_effort_modes("force")
articulation_controller.switch_control_mode("effort")

start_time2 = time.time()
start_time = time.time()
data = []
segment_data = {}  
initial_angles = [0.0, -np.pi/2, np.pi/4, - np.pi / 2, 0.0, 0]  
noise_angles = np.array(initial_angles) + np.concatenate([np.random.uniform(-0.3,0.3,6)])
my_ur.set_joint_positions(noise_angles)
initial_velocities = np.zeros(6)
noise_velocities = initial_velocities + np.concatenate([np.random.uniform(-0.3,0.3,6)])
my_ur.set_joint_velocities(noise_velocities)

segment_counter = 0  

while (time.time() - start_time2) < 5:
    A1 = np.random.uniform(0,0.2)
    A2 = np.random.uniform(0,0.04) 
    A3 = np.random.uniform(0,0.03)
    A4 = np.random.uniform(0,0.02)
    A5 = np.random.uniform(0,0.01)
    A6 = np.random.uniform(0,0.008)
    T1,T2,T3,T4,T5,T6 = np.random.uniform(0.1,1,6)
    theta1, theta2, theta3, theta4, theta5, theta6, = np.random.uniform(0,2 * np.pi,6)
    if (time.time() - start_time) <= 1:
        joint_torques = [
            A1 * np.sin(2 * np.pi/T1 * (time.time() - start_time) + theta1),
            A2 * np.sin(2 * np.pi/T2 * (time.time() - start_time) + theta2),
            A3 * np.sin(2 * np.pi/T3 * (time.time() - start_time) + theta3),
            A4 * np.sin(2 * np.pi/T4 * (time.time() - start_time) + theta4),
            A5 * np.sin(2 * np.pi/T5 * (time.time() - start_time) + theta5),
            A6 * np.sin(2 * np.pi/T6 * (time.time() - start_time) + theta6),
        ]
        noise_std = [0.04, 0.008, 0.006, 0.004, 0.002, 0.0016]
        noisy_torques = [
            joint_torques[i] + np.random.normal(0, noise_std[i]) for i in range(len(joint_torques))
        ]
        actions = ArticulationAction(
            joint_positions=None,
            joint_velocities=None,
            joint_efforts=noisy_torques,
        )

        articulation_controller.apply_action(actions)
        
        positions = my_ur.get_joint_positions()
        velocities = my_ur.get_joint_velocities()
        combined_data = np.concatenate([noisy_torques, positions, velocities])

        combined_data = combined_data.flatten()
        data.append(combined_data)

        for i in range(50):
            my_world.step(render=False)

    else:
        segment_data[f"segment_{segment_counter}"] = np.array(data)
        print(f"Segment {segment_counter} saved.")
        
        segment_counter += 1 
        data = [] 
     
        initial_angles = [0.0, -np.pi/2, np.pi/4, - np.pi / 2, 0.0, 0]  
        noise_angles = np.array(initial_angles) + np.concatenate([np.random.uniform(-0.3,0.3,6)])
        my_ur.set_joint_positions(noise_angles)
        initial_velocities = np.zeros(6)
        noise_velocities = initial_velocities + np.concatenate([np.random.uniform(-0.3,0.3,6)])
        my_ur.set_joint_velocities(noise_velocities)
        
        start_time = time.time()  
        print(f"Reinitializing robot...")

np.save('/home/cxf/franka/traj_origin_file/all_segment_testdata_ur.npy', segment_data)
# np.save('/home/cxf/桌面/210310205/data/ur_1200.npy', segment_data)
segments = segment_data.keys()
print(f"Total number of segments saved: {len(segments)}")
simulation_app.close()
