#   ~/isaacsim/python.sh "/home/nng/koopman_project/issac_code/issac sim/franka_speed_control.py"

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.franka.tasks import PickPlace
from omni.isaac.franka.controllers import PickPlaceController
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
my_franka = my_world.scene.get_object(task_params["robot_name"]["value"])
my_franka.disable_gravity()

articulation_controller = my_franka.get_articulation_controller()
articulation_controller.switch_control_mode("velocity")

start_time2 = time.time()
start_time = time.time()
data = []
segment_data = {}
initial_angles = [0.0, -np.pi/4, 0.0, - np.pi / 2, 0.0, np.pi/2,0,0,0]
noise_angles = np.array(initial_angles) + np.concatenate([np.random.uniform(-0.3,0.3,7), [0, 0]])
my_franka.set_joint_positions(noise_angles)
noise_velocities = np.concatenate([np.random.uniform(-0.3,0.3,7), [0, 0]])
my_franka.set_joint_velocities(noise_velocities)

segment_counter = 0  # 计数器，用于标记每一段数据

while (time.time() - start_time2) < 500:
    A1,A2,A3,A4,A5,A6,A7 = np.random.uniform(0,0.3,7)
    T1,T2,T3,T4,T5,T6,T7 = np.random.uniform(0.1,1,7)
    theta1, theta2, theta3, theta4, theta5, theta6, theta7, = np.random.uniform(0,2 * np.pi,7)
    if (time.time() - start_time) <= 1:
        joint_speeds = [
            A1 * np.sin(2 * np.pi/T1 * (time.time() - start_time) + theta1),
            A2 * np.sin(2 * np.pi/T2 * (time.time() - start_time) + theta2),
            A3 * np.sin(2 * np.pi/T3 * (time.time() - start_time) + theta3),
            A4 * np.sin(2 * np.pi/T4 * (time.time() - start_time) + theta4),
            A5 * np.sin(2 * np.pi/T5 * (time.time() - start_time) + theta5),
            A6 * np.sin(2 * np.pi/T6 * (time.time() - start_time) + theta6),
            A7 * np.sin(2 * np.pi/T7 * (time.time() - start_time) + theta7),
            0,
            0
        ]
        noise_std = [0.06, 0.06,0.06,0.06,0.06,0.06,0.06,0,0]
        noisy_speeds = [
            joint_speeds[i] + np.random.normal(0, noise_std[i]) for i in range(len(joint_speeds))
        ]
        actions = ArticulationAction(
            joint_positions=None,
            joint_velocities=noisy_speeds,
            joint_efforts=None,
        )

        articulation_controller.apply_action(actions)
        
        positions = my_franka.get_joint_positions()
        velocities = my_franka.get_joint_velocities()
        combined_data = np.concatenate([noisy_speeds, positions, velocities])

        combined_data = combined_data.flatten()
        data.append(combined_data)

        for i in range(20):
            #my_world.step(render=False)#是否显示
            my_world.step(render=True)

    else:
        segment_data[f"segment_{segment_counter}"] = np.array(data)
        print(f"Segment {segment_counter} saved.")
        
        segment_counter += 1 
        data = [] 
        
     
        initial_angles = [0.0, -np.pi/4, 0.0, -np.pi / 2, 0.0, np.pi/2,0,0,0] 
        noise_angles = np.array(initial_angles) + np.concatenate([np.random.uniform(-0.3,0.3,7), [0, 0]])
        my_franka.set_joint_positions(noise_angles)
        noise_velocities = np.concatenate([np.random.uniform(-0.3,0.3,7), [0, 0]])
        my_franka.set_joint_velocities(noise_velocities)
        
        start_time = time.time()  
        print(f"Reinitializing robot...")


np.save('/home/nng/koopman_project/issac_code/franka/traj_speed_file/all_segment_testdata2_franka.npy', segment_data)
segments = segment_data.keys()
print(f"Total number of segments saved: {len(segments)}")
simulation_app.close()
