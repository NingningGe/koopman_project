# #small_sample_ur
# import scipy.io as scio
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import gymnasium.envs.box2d.transfer_train as lka
#
# dicts = torch.load("/home/ccr/project-koopman/model_transfer_robot/single_small_sample_ur/Data/single_ur_12000/unifiedFRANKAlayer3_edim100_eloss1.pth",map_location=torch.device('cpu'))
# net_state_dict = dicts["net_state_dict"]
# enc_net1_state_dict = dicts["enc_net1_state_dict"]
# enc_net4_state_dict = dicts["enc_net4_state_dict"]
# dec_net1_state_dict = dicts["dec_net1_state_dict"]
# dec_net4_state_dict = dicts["dec_net4_state_dict"]
# primary_udim1 = 6
# primary_sdim1 = 12
# common_sdim = 20
# common_udim = 10
# in_dim = common_sdim
# u_dim = common_udim
# Nstate = in_dim
# layer_width = 128
# layer_width2 = 256
# layer_depth = 3
# encode_dim = 100
# XENC_layers1 = [primary_sdim1] + [layer_width] * layer_depth + [common_sdim]
# enc_net1 = lka.ENC_net(XENC_layers1)
# UENC_layers1 = [primary_udim1 + primary_sdim1] + [layer_width] * layer_depth + [common_udim]
# enc_net4 = lka.ENC_net(UENC_layers1)
# DEC_layers1 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim1]
# dec_net1 = lka.DEC_net(DEC_layers1)
# DEC_layers4 = [common_udim + primary_sdim1] + [layer_width] * layer_depth + [primary_udim1]
# dec_net4 = lka.DEC_net(DEC_layers4)
# layers = [in_dim] + [layer_width2] * layer_depth + [encode_dim]
# Nkoopman = in_dim + encode_dim
# net = lka.Network(layers, Nkoopman, u_dim)
# net.cpu().double().load_state_dict(net_state_dict)
# enc_net1.cpu().double().load_state_dict(enc_net1_state_dict)
# enc_net4.cpu().double().load_state_dict(enc_net4_state_dict)
# dec_net1.cpu().double().load_state_dict(dec_net1_state_dict)
# dec_net4.cpu().double().load_state_dict(dec_net4_state_dict)
# low = torch.tensor([-0.3, -1.9, 0.5, -1.9, -0.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4],
#                    dtype=torch.float32)
# high = torch.tensor([0.3, -1.3, 1.1, -1.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
#                     dtype=torch.float32)
#
# __credits__ = ["Andrea PIERRÉ"]
#
# import math
# from typing import TYPE_CHECKING, List, Optional
#
# import numpy as np
#
# import gymnasium as gym
# from gymnasium import error, spaces
# from gymnasium.error import DependencyNotInstalled
# from gymnasium.utils import EzPickle
#
# try:
#     import Box2D
#     from Box2D.b2 import (
#         circleShape,
#         contactListener,
#         edgeShape,
#         fixtureDef,
#         polygonShape,
#         revoluteJointDef,
#     )
# except ImportError as e:
#     raise DependencyNotInstalled(
#         "Box2D is not installed, run `pip install gymnasium[box2d]`"
#     ) from e
#
# if TYPE_CHECKING:
#     import pygame
#
# FPS = 50
#
#
# class BipedalWalker(gym.Env, EzPickle):
#     metadata = {
#         "render_modes": ["human", "rgb_array"],
#         "render_fps": FPS,
#     }
#
#     def __init__(self, render_mode: Optional[str] = None, hardcore: bool = False):
#         EzPickle.__init__(self, render_mode, hardcore)
#
#         low = np.array(
#             [-0.3, -1.9, 0.5, -1.9, -0.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4]
#         ).astype(np.float32)
#         high = np.array(
#             [0.3, -1.3, 1.1, -1.3, 0.3,  0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
#         ).astype(np.float32)
#         self.action_space = spaces.Box(
#             np.array([-0.4, -0.4, -0.4, -0.4, -0.4, -0.4]).astype(np.float32),
#             np.array([ 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]).astype(np.float32),
#         )
#         self.observation_space = spaces.Box(low, high)
#         self.n = None
#         self.state = None
#         self.sws = None
#
#
#     def reset(
#             self,
#             *,
#             seed: Optional[int] = None,
#             options: Optional[dict] = None,
#     ):
#         super().reset(seed=seed)
#         np.random.seed(100)
#         self.state = np.array([np.random.uniform(low[i], high[i]) for i in range(len(low))]).astype(np.float32)
#         s_enc = enc_net1.ENC(torch.tensor(self.state).to(torch.double))
#         self.sws = net.encode(s_enc)
#         return self.state, {}
#
#     def step(self, action: np.ndarray):
#         U_enc = enc_net4.ENC(torch.cat((torch.Tensor(action).to(torch.double), torch.tensor(self.state)), dim=-1))
#         self.sws = net.forward(self.sws, U_enc)
#         s_enc=self.sws[:Nstate]
#         self.state = dec_net1.DEC(s_enc)
#         self.state = torch.clamp(self.state.cpu(), min=low, max=high)
#         s_enc = enc_net1.ENC(self.state)
#         self.sws = net.encode(s_enc)
#
#         self.state = self.state.flatten().tolist()
#         state_array = np.array(self.state)
#         middle_array = np.array([0, -1.6, 0.8, -1.6, 0, 0])
#         reward = 0
#         reward -= 10*np.linalg.norm(state_array[:6] - middle_array)+1*np.linalg.norm(state_array[6:])+ 0.1*np.linalg.norm(action)
#         terminated = False
#
#         if self.render_mode == "human":
#             self.render()
#         return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
#
#     def close(self):
#         if self.screen is not None:
#             import pygame
#
#             pygame.display.quit()
#             pygame.quit()
#             self.isopen = False
#
#
# class BipedalWalkerHardcore:
#     def __init__(self):
#         raise error.Error(
#             "Error initializing BipedalWalkerHardcore Environment.\n"
#             "Currently, we do not support initializing this mode of environment by calling the class directly.\n"
#             "To use this environment, instead create it by specifying the hardcore keyword in gym.make, i.e.\n"
#             'gym.make("BipedalWalker-v3", hardcore=True)'
#         )


# #small_sample_transfer_ur
# import scipy.io as scio
# import numpy as np
# import matplotlib.pyplot as plt
# import gymnasium.envs.box2d.transfer_train as lka
# import torch
# 
# dicts = torch.load("/home/ccr/project-koopman/model_transfer_robot/single_robot_train/Data/single_franka/unifiedFRANKAlayer3_edim100_eloss1.pth",map_location=torch.device('cpu'))
# net_state_dict = dicts["net_state_dict"]
# enc_net1_state_dict = dicts["enc_net1_state_dict"]
# enc_net4_state_dict = dicts["enc_net4_state_dict"]
# dec_net1_state_dict = dicts["dec_net1_state_dict"]
# dec_net4_state_dict = dicts["dec_net4_state_dict"]
# primary_udim1 = 7
# primary_sdim1 = 14
# common_sdim = 20
# common_udim = 10
# in_dim = common_sdim
# u_dim = common_udim
# Nstate = in_dim
# layer_width = 128
# layer_width2 = 256
# layer_depth = 3
# encode_dim = 100
# XENC_layers1 = [primary_sdim1] + [layer_width] * layer_depth + [common_sdim]
# enc_net1 = lka.ENC_net(XENC_layers1)
# UENC_layers1 = [primary_udim1 + primary_sdim1] + [layer_width] * layer_depth + [common_udim]
# enc_net4 = lka.ENC_net(UENC_layers1)
# DEC_layers1 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim1]
# dec_net1 = lka.DEC_net(DEC_layers1)
# DEC_layers4 = [common_udim + primary_sdim1] + [layer_width] * layer_depth + [primary_udim1]
# dec_net4 = lka.DEC_net(DEC_layers4)
# layers = [in_dim] + [layer_width2] * layer_depth + [encode_dim]
# Nkoopman = in_dim + encode_dim
# net = lka.Network(layers, Nkoopman, u_dim)
# dicts = torch.load("/home/ccr/project-koopman/model_transfer_robot/model_transfer/Data/transfer_ur10000/unifiedtwo_transferlayer3_edim100_eloss1.pth",map_location=torch.device('cpu'))
# enc_net2_state_dict = dicts["enc_net2_state_dict"]
# enc_net5_state_dict = dicts["enc_net5_state_dict"]
# dec_net2_state_dict = dicts["dec_net2_state_dict"]
# dec_net5_state_dict = dicts["dec_net5_state_dict"]
# primary_udim2 = 6
# primary_sdim2 = 12
# XENC_layers2 = [primary_sdim2] + [layer_width] * layer_depth + [common_sdim]
# enc_net2 = lka.ENC_net(XENC_layers2)
# UENC_layers2 = [primary_udim2 + primary_sdim2] + [layer_width] * layer_depth + [common_udim]
# enc_net5 = lka.ENC_net(UENC_layers2)
# DEC_layers2 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim2]
# dec_net2 = lka.DEC_net(DEC_layers2)
# DEC_layers5 = [common_udim + primary_sdim2] + [layer_width] * layer_depth + [primary_udim2]
# dec_net5 = lka.DEC_net(DEC_layers5)
# net.cpu().double().load_state_dict(net_state_dict)
# enc_net1.cpu().double().load_state_dict(enc_net1_state_dict)
# enc_net4.cpu().double().load_state_dict(enc_net4_state_dict)
# dec_net1.cpu().double().load_state_dict(dec_net1_state_dict)
# dec_net4.cpu().double().load_state_dict(dec_net4_state_dict)
# enc_net2.cpu().double().load_state_dict(enc_net2_state_dict)
# enc_net5.cpu().double().load_state_dict(enc_net5_state_dict)
# dec_net2.cpu().double().load_state_dict(dec_net2_state_dict)
# dec_net5.cpu().double().load_state_dict(dec_net5_state_dict)
# low = torch.tensor([-0.3, -1.9, 0.5, -1.9, -0.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4],
#                    dtype=torch.float32)
# high = torch.tensor([0.3, -1.3, 1.1, -1.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
#                     dtype=torch.float32)
# 
# __credits__ = ["Andrea PIERRÉ"]
# 
# import math
# from typing import TYPE_CHECKING, List, Optional
# 
# import numpy as np
# 
# import gymnasium as gym
# from gymnasium import error, spaces
# from gymnasium.error import DependencyNotInstalled
# from gymnasium.utils import EzPickle
# 
# try:
#     import Box2D
#     from Box2D.b2 import (
#         circleShape,
#         contactListener,
#         edgeShape,
#         fixtureDef,
#         polygonShape,
#         revoluteJointDef,
#     )
# except ImportError as e:
#     raise DependencyNotInstalled(
#         "Box2D is not installed, run `pip install gymnasium[box2d]`"
#     ) from e
# 
# if TYPE_CHECKING:
#     import pygame
# 
# FPS = 50
# 
# class BipedalWalker(gym.Env, EzPickle):
#     metadata = {
#         "render_modes": ["human", "rgb_array"],
#         "render_fps": FPS,
#     }
# 
#     def __init__(self, render_mode: Optional[str] = None, hardcore: bool = False):
#         EzPickle.__init__(self, render_mode, hardcore)
# 
#         low = np.array(
#             [-0.3, -1.9, 0.5, -1.9, -0.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4]
#         ).astype(np.float32)
#         high = np.array(
#             [0.3, -1.3, 1.1, -1.3, 0.3,  0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
#         ).astype(np.float32)
#         self.action_space = spaces.Box(
#             np.array([-0.4, -0.4, -0.4, -0.4, -0.4, -0.4]).astype(np.float32),
#             np.array([ 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]).astype(np.float32),
#         )
#         self.observation_space = spaces.Box(low, high)
#         self.state = None
#         self.sws = None
# 
#     def reset(
#             self,
#             *,
#             seed: Optional[int] = None,
#             options: Optional[dict] = None,
#     ):
#         super().reset(seed=seed)
#         np.random.seed(100)
#         self.state = np.array([np.random.uniform(low[i], high[i]) for i in range(len(low))]).astype(np.float32)
#         s_enc = enc_net2.ENC(torch.tensor(self.state).to(torch.double))
#         self.sws = net.encode(s_enc)
#         return self.state, {}
# 
#     def step(self, action: np.ndarray):
#         # U_enc = enc_net5.ENC(torch.cat((torch.Tensor(action).to(torch.double), torch.tensor(self.state)), dim=-1))
#         # self.sws = net.forward(self.sws, U_enc)
#         # s_enc=self.sws[:Nstate]
#         # self.state = dec_net2.DEC(s_enc)
# 
#         U_enc = enc_net5.ENC(torch.cat((torch.Tensor(action).to(torch.double), torch.tensor(self.state)), dim=-1))
#         self.sws = net.forward(self.sws, U_enc)
#         s_enc=self.sws[:Nstate]
#         self.state = dec_net2.DEC(s_enc)
#         self.state = torch.clamp(self.state.cpu(), min=low, max=high)
#         s_enc = enc_net2.ENC(self.state)
#         self.sws = net.encode(s_enc)
# 
#         self.state = self.state.flatten().tolist()
#         state_array = np.array(self.state)
#         middle_array = np.array([0, -1.6, 0.8, -1.6, 0, 0])
#         reward = 0
#         reward -= 10*np.linalg.norm(state_array[:6] - middle_array)+1*np.linalg.norm(state_array[6:])+ 0.1*np.linalg.norm(action)
#         terminated = False
# 
#         if self.render_mode == "human":
#             self.render()
#         return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
# 
# 
#     def close(self):
#         if self.screen is not None:
#             import pygame
# 
#             pygame.display.quit()
#             pygame.quit()
#             self.isopen = False
# 
# 
# class BipedalWalkerHardcore:
#     def __init__(self):
#         raise error.Error(
#             "Error initializing BipedalWalkerHardcore Environment.\n"
#             "Currently, we do not support initializing this mode of environment by calling the class directly.\n"
#             "To use this environment, instead create it by specifying the hardcore keyword in gym.make, i.e.\n"
#             'gym.make("BipedalWalker-v3", hardcore=True)'
#         )




# #新方法分界线___________
# #训练franka
# import scipy.io as scio
# import numpy as np
# import matplotlib.pyplot as plt
# import gymnasium.envs.box2d.transfer_train as lka
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dicts = torch.load("/home/ccr/project-koopman/control_transfer/A_to_B/Data/franka/unifiedur_transferlayer3_edim100_eloss1.pth",map_location=torch.device('cpu'))
# net_state_dict = dicts["net_state_dict"]
# dnet_state_dict = dicts["dnet_state_dict"]
# enc_net1_state_dict = dicts["enc_net1_state_dict"]
# enc_net4_state_dict = dicts["enc_net4_state_dict"]
# dec_net1_state_dict = dicts["dec_net1_state_dict"]
# dec_net4_state_dict = dicts["dec_net4_state_dict"]
# primary_udim1 = 7
# primary_sdim1 = 14
# common_sdim = 20
# common_udim = 10
# in_dim = common_sdim
# u_dim = common_udim
# Nstate = in_dim
# layer_width = 128
# layer_width2 = 256
# layer_depth = 3
# encode_dim = 100
# XENC_layers1 = [primary_sdim1] + [layer_width] * layer_depth + [common_sdim]
# enc_net1 = lka.ENC_net(XENC_layers1)
# UENC_layers1 = [primary_udim1 + primary_sdim1] + [layer_width] * layer_depth + [common_udim]
# enc_net4 = lka.ENC_net(UENC_layers1)
# DEC_layers1 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim1]
# dec_net1 = lka.DEC_net(DEC_layers1)
# DEC_layers4 = [common_udim + primary_sdim1] + [layer_width] * layer_depth + [primary_udim1]
# dec_net4 = lka.DEC_net(DEC_layers4)
# DEC7 = [common_sdim + common_sdim] + [layer_width] * layer_depth + [common_udim]
# dnet = lka.DEC_net(DEC7)
# layers = [in_dim] + [layer_width2] * layer_depth + [encode_dim]
# Nkoopman = in_dim + encode_dim
# net = lka.Network(layers, Nkoopman, u_dim)
# net.cuda().double().load_state_dict(net_state_dict)
# dnet.cuda().double().load_state_dict(dnet_state_dict)
# enc_net1.cuda().double().load_state_dict(enc_net1_state_dict)
# enc_net4.cuda().double().load_state_dict(enc_net4_state_dict)
# dec_net1.cuda().double().load_state_dict(dec_net1_state_dict)
# dec_net4.cuda().double().load_state_dict(dec_net4_state_dict)
# low = torch.tensor([-0.3, -1.1, -0.3, -1.9, -0.3, 1.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4],
#                    dtype=torch.float32).to(device)
# high = torch.tensor([0.3, -0.5, 0.3, -1.3, 0.3, 1.9, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
#                     dtype=torch.float32).to(device)
#
# __credits__ = ["Andrea PIERRÉ"]
#
# import math
# from typing import TYPE_CHECKING, List, Optional
#
# import numpy as np
#
# import gymnasium as gym
# from gymnasium import error, spaces
# from gymnasium.error import DependencyNotInstalled
# from gymnasium.utils import EzPickle
#
# try:
#     import Box2D
#     from Box2D.b2 import (
#         circleShape,
#         contactListener,
#         edgeShape,
#         fixtureDef,
#         polygonShape,
#         revoluteJointDef,
#     )
# except ImportError as e:
#     raise DependencyNotInstalled(
#         "Box2D is not installed, run `pip install gymnasium[box2d]`"
#     ) from e
#
# if TYPE_CHECKING:
#     import pygame
#
# FPS = 50
#
# class BipedalWalker(gym.Env, EzPickle):
#     metadata = {
#         "render_modes": ["human", "rgb_array"],
#         "render_fps": FPS,
#     }
#
#     def __init__(self, render_mode: Optional[str] = None, hardcore: bool = False):
#         EzPickle.__init__(self, render_mode, hardcore)
#
#         low = np.array(
#             [-0.3, -1.1, -0.3, -1.9, -0.3, 1.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4]
#         ).astype(np.float32)
#         high = np.array(
#             [0.3, -0.5, 0.3, -1.3, 0.3, 1.9, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
#         ).astype(np.float32)
#         self.action_space = spaces.Box(
#             np.array([-0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4]).astype(np.float32),
#             np.array([ 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]).astype(np.float32),
#         )
#         self.observation_space = spaces.Box(low, high)
#         self.state = None
#         self.sws = None
#
#     def reset(
#             self,
#             *,
#             seed: Optional[int] = None,
#             options: Optional[dict] = None,
#     ):
#         super().reset(seed=seed)
#         np.random.seed(100)
#         self.state = np.array([np.random.uniform(low[i].cpu(), high[i].cpu()) for i in range(len(low))]).astype(np.float32)
#         s_enc = enc_net1.ENC(torch.tensor(self.state).to(device).to(torch.double))
#         self.sws = net.encode(s_enc)
#         return self.state, {}
#
#     def step(self, action: np.ndarray):
#         # U_enc = enc_net4.ENC(torch.cat((torch.Tensor(action).to(torch.double).to(device), torch.tensor(self.state).to(device)), dim=-1))
#         # self.sws = net.forward(self.sws, U_enc)
#         # s_enc=self.sws[:Nstate]
#         # self.state = dec_net1.DEC(s_enc)
#
#         U_enc = enc_net4.ENC(torch.cat((torch.Tensor(action).to(device).to(torch.double), torch.tensor(self.state).to(device)), dim=-1))
#         self.sws = net.forward(self.sws, U_enc)
#         s_enc=self.sws[:Nstate]
#         self.state = dec_net1.DEC(s_enc)
#         self.state = torch.clamp(self.state, min=low, max=high)
#         s_enc = enc_net1.ENC(self.state)
#         self.sws = net.encode(s_enc)
#
#
#         self.state = self.state.flatten().tolist()
#         state_array = np.array(self.state)
#         middle_array = np.array([0, -0.8, 0, -1.6, 0, 1.6, 0])
#         reward = 0
#         reward -= 10*np.linalg.norm(state_array[:7] - middle_array)+1*np.linalg.norm(state_array[7:])+ 0.1*np.linalg.norm(action)
#         terminated = False
#
#         if self.render_mode == "human":
#             self.render()
#         return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
#
#
#     def close(self):
#         if self.screen is not None:
#             import pygame
#
#             pygame.display.quit()
#             pygame.quit()
#             self.isopen = False
#
#
# class BipedalWalkerHardcore:
#     def __init__(self):
#         raise error.Error(
#             "Error initializing BipedalWalkerHardcore Environment.\n"
#             "Currently, we do not support initializing this mode of environment by calling the class directly.\n"
#             "To use this environment, instead create it by specifying the hardcore keyword in gym.make, i.e.\n"
#             'gym.make("BipedalWalker-v3", hardcore=True)'
#         )


# #训练UR5
# import scipy.io as scio
# import numpy as np
# import matplotlib.pyplot as plt
# import gymnasium.envs.box2d.transfer_train as lka
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dicts = torch.load("/home/ccr/project-koopman/control_transfer/A_to_B/Data/franka/unifiedur_transferlayer3_edim100_eloss1.pth",map_location=torch.device('cpu'))
# net_state_dict = dicts["net_state_dict"]
# dnet_state_dict = dicts["dnet_state_dict"]
# enc_net1_state_dict = dicts["enc_net1_state_dict"]
# enc_net4_state_dict = dicts["enc_net4_state_dict"]
# dec_net1_state_dict = dicts["dec_net1_state_dict"]
# dec_net4_state_dict = dicts["dec_net4_state_dict"]
# primary_udim1 = 7
# primary_sdim1 = 14
# common_sdim = 20
# common_udim = 10
# in_dim = common_sdim
# u_dim = common_udim
# Nstate = in_dim
# layer_width = 128
# layer_width2 = 256
# layer_depth = 3
# encode_dim = 100
# XENC_layers1 = [primary_sdim1] + [layer_width] * layer_depth + [common_sdim]
# enc_net1 = lka.ENC_net(XENC_layers1)
# UENC_layers1 = [primary_udim1 + primary_sdim1] + [layer_width] * layer_depth + [common_udim]
# enc_net4 = lka.ENC_net(UENC_layers1)
# DEC_layers1 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim1]
# dec_net1 = lka.DEC_net(DEC_layers1)
# DEC_layers4 = [common_udim + primary_sdim1] + [layer_width] * layer_depth + [primary_udim1]
# dec_net4 = lka.DEC_net(DEC_layers4)
# DEC7 = [common_sdim + common_sdim] + [layer_width] * layer_depth + [common_udim]
# dnet = lka.DEC_net(DEC7)
# layers = [in_dim] + [layer_width2] * layer_depth + [encode_dim]
# Nkoopman = in_dim + encode_dim
# net = lka.Network(layers, Nkoopman, u_dim)
# dicts = torch.load("/home/ccr/project-koopman/control_transfer/A_to_B/Data/franka_to_ur/2unifiedur_transferlayer3_edim100_eloss1.pth", map_location=torch.device('cpu'))
# enc_net2_state_dict = dicts["enc_net2_state_dict"]
# enc_net5_state_dict = dicts["enc_net5_state_dict"]
# dec_net2_state_dict = dicts["dec_net2_state_dict"]
# dec_net5_state_dict = dicts["dec_net5_state_dict"]
# primary_udim2 = 6
# primary_sdim2 = 12
# XENC_layers2 = [primary_sdim2] + [layer_width] * layer_depth + [common_sdim]
# enc_net2 = lka.ENC_net(XENC_layers2)
# UENC_layers2 = [primary_udim2 + primary_sdim2] + [layer_width] * layer_depth + [common_udim]
# enc_net5 = lka.ENC_net(UENC_layers2)
# DEC_layers2 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim2]
# dec_net2 = lka.DEC_net(DEC_layers2)
# DEC_layers5 = [common_udim + primary_sdim2] + [layer_width] * layer_depth + [primary_udim2]
# dec_net5 = lka.DEC_net(DEC_layers5)
# net.cuda().double().load_state_dict(net_state_dict)
# dnet.cuda().double().load_state_dict(dnet_state_dict)
# enc_net1.cuda().double().load_state_dict(enc_net1_state_dict)
# enc_net4.cuda().double().load_state_dict(enc_net4_state_dict)
# dec_net1.cuda().double().load_state_dict(dec_net1_state_dict)
# dec_net4.cuda().double().load_state_dict(dec_net4_state_dict)
# enc_net2.cuda().double().load_state_dict(enc_net2_state_dict)
# enc_net5.cuda().double().load_state_dict(enc_net5_state_dict)
# dec_net2.cuda().double().load_state_dict(dec_net2_state_dict)
# dec_net5.cuda().double().load_state_dict(dec_net5_state_dict)
# middle_array_franka = np.array([0, -0.8, 0, -1.6, 0, 1.6, 0, 0, 0, 0, 0, 0, 0, 0])
# middle_array_ur = dec_net2.DEC(enc_net1.ENC(torch.tensor(middle_array_franka).to(device).to(torch.double))).detach().cpu().numpy()
# 
# low1 = torch.tensor([-0.3, -1.9, 0.5, -1.9, -0.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4],
#                    dtype=torch.float32).to(device)
# high1 = torch.tensor([0.3, -1.3, 1.1, -1.3, 0.3,  0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
#                     dtype=torch.float32).to(device)
# 
# __credits__ = ["Andrea PIERRÉ"]
# 
# import math
# from typing import TYPE_CHECKING, List, Optional
# 
# import numpy as np
# 
# import gymnasium as gym
# from gymnasium import error, spaces
# from gymnasium.error import DependencyNotInstalled
# from gymnasium.utils import EzPickle
# 
# try:
#     import Box2D
#     from Box2D.b2 import (
#         circleShape,
#         contactListener,
#         edgeShape,
#         fixtureDef,
#         polygonShape,
#         revoluteJointDef,
#     )
# except ImportError as e:
#     raise DependencyNotInstalled(
#         "Box2D is not installed, run `pip install gymnasium[box2d]`"
#     ) from e
# 
# if TYPE_CHECKING:
#     import pygame
# 
# FPS = 50
# 
# class BipedalWalker(gym.Env, EzPickle):
#     metadata = {
#         "render_modes": ["human", "rgb_array"],
#         "render_fps": FPS,
#     }
# 
#     def __init__(self, render_mode: Optional[str] = None, hardcore: bool = False):
#         EzPickle.__init__(self, render_mode, hardcore)
# 
#         low = np.array(
#             [-0.3, -1.9, 0.5, -1.9, -0.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4]
#         ).astype(np.float32)
#         high = np.array(
#             [0.3, -1.3, 1.1, -1.3, 0.3,  0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
#         ).astype(np.float32)
#         self.action_space = spaces.Box(
#             np.array([-0.4, -0.4, -0.4, -0.4, -0.4, -0.4]).astype(np.float32),
#             np.array([ 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]).astype(np.float32),
#         )
#         self.observation_space = spaces.Box(low, high)
#         self.state = None
#         self.sws = None
# 
#     def reset(
#             self,
#             *,
#             seed: Optional[int] = None,
#             options: Optional[dict] = None,
#     ):
#         super().reset(seed=seed)
#         np.random.seed(100)
#         self.state = np.array([np.random.uniform(low1[i].cpu(), high1[i].cpu()) for i in range(len(low1))]).astype(np.float32)
#         s_enc = enc_net2.ENC(torch.tensor(self.state).to(device).to(torch.double))
#         self.sws = net.encode(s_enc)
#         return self.state, {}
# 
#     def step(self, action: np.ndarray):
#         # U_enc = enc_net5.ENC(torch.cat((torch.Tensor(action).to(torch.double).to(device), torch.tensor(self.state).to(device)), dim=-1))
#         # self.sws = net.forward(self.sws, U_enc)
#         # s_enc=self.sws[:Nstate]
#         # self.state = dec_net2.DEC(s_enc)
# 
#         U_enc = enc_net5.ENC(torch.cat((torch.Tensor(action).to(device).to(torch.double), torch.tensor(self.state).to(device)), dim=-1))
#         self.sws = net.forward(self.sws, U_enc)
#         s_enc=self.sws[:Nstate]
#         self.state = dec_net2.DEC(s_enc)
#         self.state = torch.clamp(self.state, min=low1, max=high1)
#         s_enc = enc_net2.ENC(self.state)
#         self.sws = net.encode(s_enc)
# 
# 
#         self.state = self.state.flatten().tolist()
#         state_array = np.array(self.state)
#         reward = 0
#         reward -= 10*np.linalg.norm(state_array[:6] - middle_array_ur[:6])+1*np.linalg.norm(state_array[6:])+ 0.1*np.linalg.norm(action)
#         terminated = False
# 
#         if self.render_mode == "human":
#             self.render()
#         return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
# 
# 
#     def close(self):
#         if self.screen is not None:
#             import pygame
# 
#             pygame.display.quit()
#             pygame.quit()
#             self.isopen = False
# 
# 
# class BipedalWalkerHardcore:
#     def __init__(self):
#         raise error.Error(
#             "Error initializing BipedalWalkerHardcore Environment.\n"
#             "Currently, we do not support initializing this mode of environment by calling the class directly.\n"
#             "To use this environment, instead create it by specifying the hardcore keyword in gym.make, i.e.\n"
#             'gym.make("BipedalWalker-v3", hardcore=True)'
#         )



# 分界线
# franka(Y to U)
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import gymnasium.envs.box2d.transfer_train as lka
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dicts = torch.load("/home/ccr/project-koopman/control_transfer/A_to_B/Data/franka/unifiedur_transferlayer3_edim100_eloss1.pth",map_location=torch.device('cpu'))
net_state_dict = dicts["net_state_dict"]
dnet_state_dict = dicts["dnet_state_dict"]
enc_net1_state_dict = dicts["enc_net1_state_dict"]
enc_net4_state_dict = dicts["enc_net4_state_dict"]
dec_net1_state_dict = dicts["dec_net1_state_dict"]
dec_net4_state_dict = dicts["dec_net4_state_dict"]
primary_udim1 = 7
primary_sdim1 = 14
common_sdim = 20
common_udim = 10
in_dim = common_sdim
u_dim = common_udim
Nstate = in_dim
layer_width = 128
layer_width2 = 256
layer_depth = 3
encode_dim = 100
XENC_layers1 = [primary_sdim1] + [layer_width] * layer_depth + [common_sdim]
enc_net1 = lka.ENC_net(XENC_layers1)
UENC_layers1 = [primary_udim1 + primary_sdim1] + [layer_width] * layer_depth + [common_udim]
enc_net4 = lka.ENC_net(UENC_layers1)
DEC_layers1 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim1]
dec_net1 = lka.DEC_net(DEC_layers1)
DEC_layers4 = [common_udim + primary_sdim1] + [layer_width] * layer_depth + [primary_udim1]
dec_net4 = lka.DEC_net(DEC_layers4)
DEC7 = [common_sdim + common_sdim] + [layer_width] * layer_depth + [common_udim]
dnet = lka.DEC_net(DEC7)
layers = [in_dim] + [layer_width2] * layer_depth + [encode_dim]
Nkoopman = in_dim + encode_dim
net = lka.Network(layers, Nkoopman, u_dim)
dicts = torch.load("/home/ccr/project-koopman/control_transfer/A_to_B/Data/franka_to_ur/2unifiedur_transferlayer3_edim100_eloss1.pth", map_location=torch.device('cpu'))
enc_net2_state_dict = dicts["enc_net2_state_dict"]
enc_net5_state_dict = dicts["enc_net5_state_dict"]
dec_net2_state_dict = dicts["dec_net2_state_dict"]
dec_net5_state_dict = dicts["dec_net5_state_dict"]
primary_udim2 = 6
primary_sdim2 = 12
XENC_layers2 = [primary_sdim2] + [layer_width] * layer_depth + [common_sdim]
enc_net2 = lka.ENC_net(XENC_layers2)
UENC_layers2 = [primary_udim2 + primary_sdim2] + [layer_width] * layer_depth + [common_udim]
enc_net5 = lka.ENC_net(UENC_layers2)
DEC_layers2 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim2]
dec_net2 = lka.DEC_net(DEC_layers2)
DEC_layers5 = [common_udim + primary_sdim2] + [layer_width] * layer_depth + [primary_udim2]
dec_net5 = lka.DEC_net(DEC_layers5)
net.cuda().double().load_state_dict(net_state_dict)
dnet.cuda().double().load_state_dict(dnet_state_dict)
enc_net1.cuda().double().load_state_dict(enc_net1_state_dict)
enc_net4.cuda().double().load_state_dict(enc_net4_state_dict)
dec_net1.cuda().double().load_state_dict(dec_net1_state_dict)
dec_net4.cuda().double().load_state_dict(dec_net4_state_dict)
enc_net2.cuda().double().load_state_dict(enc_net2_state_dict)
enc_net5.cuda().double().load_state_dict(enc_net5_state_dict)
dec_net2.cuda().double().load_state_dict(dec_net2_state_dict)
dec_net5.cuda().double().load_state_dict(dec_net5_state_dict)
low1 = torch.tensor([-0.3, -1.1, -0.3, -1.9, -0.3, 1.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4],
                    dtype=torch.float32).to(device)
high1 = torch.tensor([0.3, -0.5, 0.3, -1.3, 0.3, 1.9, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                     dtype=torch.float32).to(device)

middle_array_franka = np.array([0, -0.8, 0, -1.6, 0, 1.6, 0, 0, 0, 0, 0, 0, 0, 0])
middle_array_ur = dec_net2.DEC(enc_net1.ENC(torch.tensor(middle_array_franka).to(device).to(torch.double))).detach().cpu().numpy()

__credits__ = ["Andrea PIERRÉ"]

import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle
import random

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        "Box2D is not installed, run `pip install gymnasium[box2d]`"
    ) from e

if TYPE_CHECKING:
    import pygame

FPS = 50

class BipedalWalker(gym.Env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self, render_mode: Optional[str] = None, hardcore: bool = False):
        EzPickle.__init__(self, render_mode, hardcore)
        low = np.array(
            [ -0.1447, -0.3694, -0.1571, -0.3394, -0.2518, -0.3535, -0.2278, -0.3109, -0.2114, -0.2000,
            -0.4093, -0.2955, -0.2017, -0.2927, -0.2782, -0.1364, -0.1511, -0.3998, -0.2175, -0.2170]
        ).astype(np.float32)
        high = np.array(
            [0.1629, 0.3086, 0.1745, 0.4130, 0.2384, 0.3204, 0.2471, 0.3564, 0.2566, 0.2106,
            0.3168, 0.3144, 0.2520, 0.2572, 0.2416, 0.1483, 0.1427, 0.3739, 0.2293, 0.2159]
        ).astype(np.float32)
        self.action_space = spaces.Box(
            np.array([-0.4533, -0.4103, -0.0397, -0.4026, -0.3591, -0.3698, -0.1891, -0.1338, -0.4911, -0.3657]).astype(np.float32),
            np.array([0.4189, 0.4068, 0.0370, 0.3453, 0.3644, 0.4608, 0.1827, 0.1348, 0.4151, 0.4022]).astype(np.float32),
        )
        self.observation_space = spaces.Box(low, high)
        self.state = None
        self.sws = None
        self.franka_state = None

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        franka_state = np.array([np.random.uniform(np.array(low1[i].cpu()), np.array(high1[i].cpu())) for i in range(len(low1))]).astype(np.float32)
        self.franka_state = torch.tensor(franka_state).to(device).to(torch.double)
        s_enc = enc_net1.ENC(self.franka_state)
        self.sws = net.encode(s_enc)
        self.state = s_enc
        return np.array(self.state.detach().cpu(), dtype=np.float32), {}

    def step(self, action: np.ndarray):
        U_enc = torch.Tensor(action).to(device).to(torch.double)
        self.sws = net.forward(self.sws, U_enc)
        s_enc=self.sws[:Nstate]
        self.franka_state = dec_net1.DEC(s_enc)
        s_enc = enc_net1.ENC(self.franka_state)
        self.sws = net.encode(s_enc)

        self.state = s_enc
        franka_state_array = np.array(self.franka_state.detach().cpu())
        reward = 0
        reward -= 10 * np.linalg.norm(franka_state_array[:7] - middle_array_franka[:7]) + 1 * np.linalg.norm(
            franka_state_array[7:]) + 0.1 * np.linalg.norm(action)
        terminated = False


        if self.render_mode == "human":
            self.render()
        return np.array(self.state.detach().cpu(), dtype=np.float32), reward, terminated, False, {}


    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class BipedalWalkerHardcore:
    def __init__(self):
        raise error.Error(
            "Error initializing BipedalWalkerHardcore Environment.\n"
            "Currently, we do not support initializing this mode of environment by calling the class directly.\n"
            "To use this environment, instead create it by specifying the hardcore keyword in gym.make, i.e.\n"
            'gym.make("BipedalWalker-v3", hardcore=True)'
        )


# # ur(Y to U)
# import scipy.io as scio
# import numpy as np
# import matplotlib.pyplot as plt
# import gymnasium.envs.box2d.transfer_train as lka
# import torch
# import gymnasium.envs.box2d.td3_continuous_action as td3
# 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dicts = torch.load("/home/ccr/project-koopman/control_transfer/A_to_B/Data/franka/unifiedur_transferlayer3_edim100_eloss1.pth",map_location=torch.device('cpu'))
# net_state_dict = dicts["net_state_dict"]
# dnet_state_dict = dicts["dnet_state_dict"]
# enc_net1_state_dict = dicts["enc_net1_state_dict"]
# enc_net4_state_dict = dicts["enc_net4_state_dict"]
# dec_net1_state_dict = dicts["dec_net1_state_dict"]
# dec_net4_state_dict = dicts["dec_net4_state_dict"]
# primary_udim1 = 7
# primary_sdim1 = 14
# common_sdim = 20
# common_udim = 10
# in_dim = common_sdim
# u_dim = common_udim
# Nstate = in_dim
# layer_width = 128
# layer_width2 = 256
# layer_depth = 3
# encode_dim = 100
# XENC_layers1 = [primary_sdim1] + [layer_width] * layer_depth + [common_sdim]
# enc_net1 = lka.ENC_net(XENC_layers1)
# UENC_layers1 = [primary_udim1 + primary_sdim1] + [layer_width] * layer_depth + [common_udim]
# enc_net4 = lka.ENC_net(UENC_layers1)
# DEC_layers1 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim1]
# dec_net1 = lka.DEC_net(DEC_layers1)
# DEC_layers4 = [common_udim + primary_sdim1] + [layer_width] * layer_depth + [primary_udim1]
# dec_net4 = lka.DEC_net(DEC_layers4)
# DEC7 = [common_sdim + common_sdim] + [layer_width] * layer_depth + [common_udim]
# dnet = lka.DEC_net(DEC7)
# layers = [in_dim] + [layer_width2] * layer_depth + [encode_dim]
# Nkoopman = in_dim + encode_dim
# net = lka.Network(layers, Nkoopman, u_dim)
# dicts = torch.load("/home/ccr/project-koopman/control_transfer/A_to_B/Data/franka_to_ur/2unifiedur_transferlayer3_edim100_eloss1.pth", map_location=torch.device('cpu'))
# enc_net2_state_dict = dicts["enc_net2_state_dict"]
# enc_net5_state_dict = dicts["enc_net5_state_dict"]
# dec_net2_state_dict = dicts["dec_net2_state_dict"]
# dec_net5_state_dict = dicts["dec_net5_state_dict"]
# primary_udim2 = 6
# primary_sdim2 = 12
# XENC_layers2 = [primary_sdim2] + [layer_width] * layer_depth + [common_sdim]
# enc_net2 = lka.ENC_net(XENC_layers2)
# UENC_layers2 = [primary_udim2 + primary_sdim2] + [layer_width] * layer_depth + [common_udim]
# enc_net5 = lka.ENC_net(UENC_layers2)
# DEC_layers2 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim2]
# dec_net2 = lka.DEC_net(DEC_layers2)
# DEC_layers5 = [common_udim + primary_sdim2] + [layer_width] * layer_depth + [primary_udim2]
# dec_net5 = lka.DEC_net(DEC_layers5)
# net.cuda().double().load_state_dict(net_state_dict)
# dnet.cuda().double().load_state_dict(dnet_state_dict)
# enc_net1.cuda().double().load_state_dict(enc_net1_state_dict)
# enc_net4.cuda().double().load_state_dict(enc_net4_state_dict)
# dec_net1.cuda().double().load_state_dict(dec_net1_state_dict)
# dec_net4.cuda().double().load_state_dict(dec_net4_state_dict)
# enc_net2.cuda().double().load_state_dict(enc_net2_state_dict)
# enc_net5.cuda().double().load_state_dict(enc_net5_state_dict)
# dec_net2.cuda().double().load_state_dict(dec_net2_state_dict)
# dec_net5.cuda().double().load_state_dict(dec_net5_state_dict)
# middle_array_franka = np.array([0, -0.8, 0, -1.6, 0, 1.6, 0, 0, 0, 0, 0, 0, 0, 0])
# middle_array_ur = dec_net2.DEC(enc_net1.ENC(torch.tensor(middle_array_franka).to(device).to(torch.double))).detach().cpu().numpy()
# 
# low1 = torch.tensor([-0.3, -1.9, 0.5, -1.9, -0.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4],
#                    dtype=torch.float32).to(device)
# high1 = torch.tensor([0.3, -1.3, 1.1, -1.3, 0.3,  0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
#                     dtype=torch.float32).to(device)
# 
# 
# __credits__ = ["Andrea PIERRÉ"]
# 
# import math
# from typing import TYPE_CHECKING, List, Optional
# 
# import numpy as np
# 
# import gymnasium as gym
# from gymnasium import error, spaces
# from gymnasium.error import DependencyNotInstalled
# from gymnasium.utils import EzPickle
# 
# try:
#     import Box2D
#     from Box2D.b2 import (
#         circleShape,
#         contactListener,
#         edgeShape,
#         fixtureDef,
#         polygonShape,
#         revoluteJointDef,
#     )
# except ImportError as e:
#     raise DependencyNotInstalled(
#         "Box2D is not installed, run `pip install gymnasium[box2d]`"
#     ) from e
# 
# if TYPE_CHECKING:
#     import pygame
# 
# FPS = 50
# 
# class BipedalWalker(gym.Env, EzPickle):
#     metadata = {
#         "render_modes": ["human", "rgb_array"],
#         "render_fps": FPS,
#     }
# 
#     def __init__(self, render_mode: Optional[str] = None, hardcore: bool = False):
#         EzPickle.__init__(self, render_mode, hardcore)
#         low = np.array(
#             [ -0.1447, -0.3694, -0.1571, -0.3394, -0.2518, -0.3535, -0.2278, -0.3109, -0.2114, -0.2000,
#             -0.4093, -0.2955, -0.2017, -0.2927, -0.2782, -0.1364, -0.1511, -0.3998, -0.2175, -0.2170]
#         ).astype(np.float32)
#         high = np.array(
#             [0.1629, 0.3086, 0.1745, 0.4130, 0.2384, 0.3204, 0.2471, 0.3564, 0.2566, 0.2106,
#             0.3168, 0.3144, 0.2520, 0.2572, 0.2416, 0.1483, 0.1427, 0.3739, 0.2293, 0.2159]
#         ).astype(np.float32)
#         self.action_space = spaces.Box(
#             np.array([-0.4533, -0.4103, -0.0397, -0.4026, -0.3591, -0.3698, -0.1891, -0.1338, -0.4911, -0.3657]).astype(np.float32),
#             np.array([0.4189, 0.4068, 0.0370, 0.3453, 0.3644, 0.4608, 0.1827, 0.1348, 0.4151, 0.4022]).astype(np.float32),
#         )
#         self.observation_space = spaces.Box(low, high)
#         self.state = None
#         self.sws = None
#         self.ur_state = None
# 
#     def reset(
#             self,
#             *,
#             seed: Optional[int] = None,
#             options: Optional[dict] = None,
#     ):
#         super().reset(seed=seed)
#         ur_state = np.array([np.random.uniform(np.array(low1[i].cpu()), np.array(high1[i].cpu())) for i in range(len(low1))]).astype(np.float32)
#         self.ur_state = torch.tensor(ur_state).to(device).to(torch.double)
#         s_enc = enc_net2.ENC(self.ur_state)
#         self.sws = net.encode(s_enc)
#         self.state = s_enc
#         return np.array(self.state.detach().cpu(), dtype=np.float32), {}
# 
#     def step(self, action: np.ndarray):
#         U_enc = torch.Tensor(action).to(device).to(torch.double)
#         self.sws = net.forward(self.sws, U_enc)
#         s_enc=self.sws[:Nstate]
#         self.ur_state = dec_net2.DEC(s_enc)
#         s_enc = enc_net2.ENC(self.ur_state)
#         self.sws = net.encode(s_enc)
# 
#         self.state = s_enc
#         ur_state_array = np.array(self.ur_state.detach().cpu())
#         reward = 0
#         reward -= 10*np.linalg.norm(ur_state_array[:6] - middle_array_ur[:6])+1*np.linalg.norm(ur_state_array[6:])+ 0.1*np.linalg.norm(action)
#         terminated = False
# 
#         if self.render_mode == "human":
#             self.render()
#         return np.array(self.state.detach().cpu(), dtype=np.float32), reward, terminated, False, {}
# 
# 
#     def close(self):
#         if self.screen is not None:
#             import pygame
# 
#             pygame.display.quit()
#             pygame.quit()
#             self.isopen = False
# 
# 
# class BipedalWalkerHardcore:
#     def __init__(self):
#         raise error.Error(
#             "Error initializing BipedalWalkerHardcore Environment.\n"
#             "Currently, we do not support initializing this mode of environment by calling the class directly.\n"
#             "To use this environment, instead create it by specifying the hardcore keyword in gym.make, i.e.\n"
#             'gym.make("BipedalWalker-v3", hardcore=True)'
#         )