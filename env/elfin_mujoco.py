import gymnasium as gym
from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.interfaces.mujoco import Mujoco
from gymnasium.utils import seeding
from gymnasium.spaces import Tuple, Box
import numpy as np
from .env_untils import get_limits, generate_target_traj3

ANGLE_LIMIT = [[-3.14, 3.14], [-2.25, 2.35], [-2.61, 2.61], [-3.14, 3.14], [-2.56, 2.56],]
INIT_JOINTS = [0,  0,  0,  0,   0,  0]

class elfin5_mujoco(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", ],
        "render_fps": 60, }
    def __init__(self, env_config, ):
        super().__init__()
        self.dis_flag = env_config.dis if env_config.dis else 1.732
        self.init_joint, = INIT_JOINTS
        _, self.targets_traj = generate_target_traj3(dis=self.dis_flag)
        self.action_ratio = env_config.action_ratio
        self.dis = env_config.distance_threshold
        self.max_episode_steps = env_config.max_episode_steps
        self.regulation = env_config.regulation
        self.vis = env_config.visualize
        self.len = 27
        self.lows, self.highs = get_limits(ANGLE_LIMIT)
        assert len(self.lows) == env_config.n_action
        self.action_space = Box(low=self.lows, high=self.highs, shape=(env_config.n_action,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                     shape=(self.len,), dtype=np.float32, seed=829)

        self.robot_config = arm(env_config.robot_name)
        self.interface = Mujoco(self.robot_config,
                                display_frequency=self.vis, dt=0.001,
                                # render_params={'cameras': [1], 'resolution': [480, 848], 'depth': True},
                                )
        self.interface.connect()
        self.reset()

    def reset(self, seed=1, options=None):
        self.np_random, seed = seeding.np_random(seed)
        self.step_counter = 1
        self.terminated = False
        self.success = False
        self.angle_old = np.array(self.init_joint[:5])
        assert len(self.angle_old) == 5
        self.target_flag = 0
        self.target_pos = self.targets_traj[self.target_flag]
        self.interface.set_mocap_xyz(name="goal", xyz=self.target_pos)
        self.interface.set_mocap_xyz(name="obstacle", xyz=[0, 0, 10])
        self.interface.send_target_angles(np.array(self.init_joint))
        self.EE_old = np.array(self.interface.get_xyz(name="EE"))
        self.dis_out = np.linalg.norm(self.target_pos - self.EE_old)
        return self._get_obs(), {'EE': self.EE_old, 'goal':self.target_pos, 'dist':self.dis_out}

    def render(self):
        if self.vis > 0:
            self.interface.viewer.render()

    def _move_limit(self, ang):
        angle = ang.copy()
        for i in range(len(ang)):
            if ang[i] < self.lows[i]:
                angle[i] = self.lows[i]
            elif ang[i] > self.highs[i]:
                angle[i] = self.highs[i]
        return angle

    def step(self, action):
        if self.regulation:
            angle_new = self._move_limit(ang=np.array(action) * float(self.action_ratio) + self.angle_old)
        else:
            angle_new = self._move_limit(ang=np.array(action) * self.highs)

        self.interface.send_target_angles(np.concatenate((angle_new, [0]), axis=0).astype(np.float32))
        new_ee_pos = np.array(self.interface.get_xyz("EE")).astype(np.float32)
        # 基于末端坐标与步幅的基础奖励
        dis_old = np.linalg.norm(self.target_pos - self.EE_old)
        dis_new = np.linalg.norm(self.target_pos - new_ee_pos)  # 离目标的距离
        dis_r = 100 * (dis_old - dis_new)

        end_flag = 0
        if self.target_flag >= len(self.targets_traj)-1 and dis_new < self.dis:
            end_flag = 1
            re_flag = 0
        elif self.target_flag < len(self.targets_traj)-1 and dis_new < self.dis:
            self.target_flag += 1
            self.target_pos = self.targets_traj[self.target_flag]
            re_flag = 1
        else:
            re_flag = 0

        if self.step_counter >= self.max_episode_steps:
            reward = -20 + dis_r
            self.terminated = True
            self.success = False

        elif end_flag:
            reward = 50 + (self.max_episode_steps-self.step_counter)
            self.terminated = True
            self.success = True
        else:
            reward = dis_r + re_flag * 20
            self.terminated = False
            self.success = False

        self.step_counter += 1
        self.EE_old = new_ee_pos
        self.dis_out = dis_new
        info= {}
        return self._get_obs(), reward, self.terminated, self.success, info

    def close(self):
        self.interface.disconnect()
        print("Simulation terminated...")