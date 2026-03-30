from scipy.io import savemat
import numpy as np
franka_data = np.load("/home/ccr/unified_represention/speed_transfer_learning/traj_data/traj_train_file/Ktest_data_franka.npy")
franka_data = franka_data.reshape(-1,21)
ur_data = np.load("/home/ccr/unified_represention/speed_transfer_learning/traj_data/traj_train_file/Ktest_data_ur.npy")
ur_data = ur_data.reshape(-1,18)
dofbot_data = np.load("/home/ccr/unified_represention/speed_transfer_learning/traj_data/traj_train_file/Ktest_data_dofbot.npy")
dofbot_data = dofbot_data.reshape(-1,15)

savemat("/home/ccr/unified_represention/speed_transfer_learning/traj_data/data_processing/robot.mat", {'dofbot_data':dofbot_data,'ur_data':ur_data,'franka_data':franka_data})
