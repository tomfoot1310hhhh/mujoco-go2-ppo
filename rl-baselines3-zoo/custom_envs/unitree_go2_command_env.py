import os
import gymnasium as gym
import mujoco.renderer
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium.spaces import Box
import math
from scipy.spatial.transform import Rotation as R
from mujoco import Renderer
import time


class UnitreeGo2CommandEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        model_path = os.path.join(os.path.dirname(__file__), "assets", "unitree_go2", "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        # obs_size = self.model.nq + self.model.nv + 2
        obs_size = self.model.nq + self.model.nv + 4
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)


        self.step_count = 0
        self.frame_skip = 10
        self.dt = self.model.opt.timestep * self.frame_skip
        self.step_limit = int(20.0 / self.dt)

        self.default_joint_pos = np.tile(
            np.array([0.0, 0.9, -1.8]), 4
        )

        self.action_scale = np.tile(
            np.array([0.25, 0.35, 0.45]), 4
        )

        self.kp = np.full(12, 35.0)
        self.kd = np.full(12, 0.8)
        self.forward_velocity = 0.0
        # updates on Sep 3
        self.lateral_velocity = 0.0
        self.yaw_rate = 0.0
        self.yaw_tracking_sigma = 0.25
        # ending
        # updates on Sep 4
        self.gait_period = 0.5
        # ending
        self.tracking_sigma = 0.25

        self.command_x = 0.5
        self.command_y = 0.0
        self.command_yaw = 0
        self.base_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "base",
        )
        self.floor_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "floor",
        )

        self.foot_geom_ids = {
            name: mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )
            for name in ("FL", "FR", "RL", "RR")
        }
        self.base_height_target = 0.3
        # self.tracking_sigma = 0.25

        # Set physics parameters to stabilize simulation
        # self.model.dof_damping[6:] = 4.0
        # self.model.dof_frictionloss[6:] = 1.0
        # self.model.dof_armature[6:] = 0.05

        # Map joint names to indices
        self.joint_indices = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name is not None:
                self.joint_indices[name] = i

        self.default_dof_pos = self._get_default_joint_pos()
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)

        self.logger = RewardLogger()

    # Update on Sep 4
    def _reward_foot_height_gait(self):
        # phase_angle = (
        #     2.0
        #     * np.pi
        #     * self.data.time
        #     / self.gait_period
        # )
        # phase_signal = np.sin(phase_angle)

        # # 两组对角腿的连续抬脚目标
        # target_pair_1 = max(0.0, phase_signal)
        # target_pair_2 = max(0.0, -phase_signal)

        # # 顺序：FL、FR、RL、RR
        # desired_heights = np.array(
        #     [
        #         target_pair_1,
        #         target_pair_2,
        #         target_pair_2,
        #         target_pair_1,
        #     ],
        #     dtype=np.float64,
        # )

        # foot_heights = np.array(
        #     [
        #         self.data.geom_xpos[
        #             self.foot_geom_ids[name]
        #         ][2]
        #         for name in ("FL", "FR", "RL", "RR")
        #     ],
        #     dtype=np.float64,
        # )

        # # 日志显示触地中心约0.01 m，
        # # 目标最高抬到约0.04 m
        # normalized_heights = np.clip(
        #     (foot_heights - 0.01) / 0.03,
        #     0.0,
        #     1.0,
        # )

        # height_error = np.mean(
        #     np.square(
        #         normalized_heights
        #         - desired_heights
        #     )
        # )

        # return -height_error
        phase = np.sin(
        2.0 * np.pi * self.data.time / self.gait_period
        )

        # FL + RR 与 FR + RL 交替抬腿
        pair_1_target = max(0.0, phase)
        pair_2_target = max(0.0, -phase)

        desired_lift = np.array(
            [
                pair_1_target,  # FL
                pair_2_target,  # FR
                pair_2_target,  # RL
                pair_1_target,  # RR
            ],
            dtype=np.float64,
        )

        base_position = self.data.xpos[self.base_body_id]
        base_rotation = self.data.xmat[
            self.base_body_id
        ].reshape(3, 3)

        relative_foot_z = []

        for foot_name in ("FL", "FR", "RL", "RR"):
            geom_id = self.foot_geom_ids[foot_name]

            world_offset = (
                self.data.geom_xpos[geom_id] - base_position
            )
            foot_in_base = base_rotation.T @ world_offset
            relative_foot_z.append(foot_in_base[2])

        relative_foot_z = np.asarray(
            relative_foot_z,
            dtype=np.float64,
        )

        # 相对躯干：-0.24 表示腿伸展，-0.18 表示腿抬起
        stance_z = -0.24
        swing_z = -0.18

        normalized_lift = np.clip(
            (relative_foot_z - stance_z) / (swing_z - stance_z),
            0.0,
            1.0,
        )

        return -np.mean(
            np.square(normalized_lift - desired_lift)
        )

    def _reward_knee_flexion_gait(self):
        phase = np.sin(
            2.0 * np.pi * self.data.time / self.gait_period
        )

        pair_1_target = max(0.0, phase)
        pair_2_target = max(0.0, -phase)

        desired_lift = np.array(
            [
                pair_1_target,  # FL
                pair_2_target,  # FR
                pair_2_target,  # RL
                pair_1_target,  # RR
            ],
            dtype=np.float64,
        )

        # FL、FR、RL、RR 的 calf/knee 关节
        calf_angles = self.data.qpos[
            [9, 12, 15, 18]
        ]

        # 支撑时接近默认角度，摆动时适度屈膝
        stance_calf = -1.8
        swing_calf = -2.1

        desired_calf = (
            stance_calf
            + (swing_calf - stance_calf) * desired_lift
        )

        normalized_error = (
            calf_angles - desired_calf
        ) / 0.3

        return -np.mean(
            np.square(normalized_error)
        )

    def _get_gait_clock(self):
        phase_angle = (
            2.0
            * np.pi
            * self.data.time
            / self.gait_period
        )

        return np.array(
            [
                np.sin(phase_angle),
                np.cos(phase_angle),
            ],
            dtype=np.float64,
        )

    def _penalty_foot_slip(self):
        contacts = self._get_foot_contacts()
        slip_speeds = self._get_foot_slip_speeds(contacts)

        penalty = 0.0

        for name in self.foot_geom_ids:
            if contacts[name]:
                # 限制极端值，避免一次异常接触支配整个奖励
                slip_speed = min(slip_speeds[name], 2.0)
                penalty += slip_speed ** 2

        return penalty

    def _get_foot_slip_speeds(self, foot_contacts):
        slip_speeds = {}

        for name, geom_id in self.foot_geom_ids.items():
            if not foot_contacts[name]:
                slip_speeds[name] = 0.0
                continue

            velocity_world = np.zeros(6, dtype=np.float64)

            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
                velocity_world,
                0,
            )

            linear_velocity = velocity_world[3:]

            # 脚接触地面时的世界坐标水平速度
            slip_speeds[name] = float(
                np.linalg.norm(linear_velocity[:2])
            )

        return slip_speeds

    @staticmethod
    def _score_trot_contacts(contacts):
        fl = float(contacts["FL"])
        fr = float(contacts["FR"])
        rl = float(contacts["RL"])
        rr = float(contacts["RR"])

        # 对角腿内部应保持同步：
        # FL 与 RR 相同，FR 与 RL 相同
        diagonal_sync_error = 0.5 * (
            abs(fl - rr)
            + abs(fr - rl)
        )

        # 两组对角腿应该处于不同接触状态
        diagonal_pair_1 = 0.5 * (fl + rr)
        diagonal_pair_2 = 0.5 * (fr + rl)
        pair_opposition = abs(
            diagonal_pair_1 - diagonal_pair_2
        )

        # 四脚全部腾空需要额外惩罚
        all_airborne = float(
            fl + fr + rl + rr == 0
        )

        return (
            pair_opposition
            - diagonal_sync_error
            - all_airborne
        )


    # def _reward_trot_gait(self):
    #     contacts = self._get_foot_contacts()
    #     fl = float(contacts["FL"])
    #     fr = float(contacts["FR"])
    #     rl = float(contacts["RL"])
    #     rr = float(contacts["RR"])

    #     phase_angle = (
    #         2.0
    #         * np.pi
    #         * self.data.time
    #         / self.gait_period
    #     )
    #     phase_signal = np.sin(phase_angle)

    #     diagonal_pair_1 = 0.5 * (fl + rr)
    #     diagonal_pair_2 = 0.5 * (fr + rl)

    #     # 正半周期奖励 FL+RR，负半周期奖励 FR+RL
    #     phase_alignment = phase_signal * (
    #         diagonal_pair_1 - diagonal_pair_2
    #     )

    #     diagonal_sync_error = 0.5 * (
    #         abs(fl - rr)
    #         + abs(fr - rl)
    #     )

    #     all_airborne = float(
    #         fl + fr + rl + rr == 0
    #     )

    #     return (
    #         phase_alignment
    #         - 0.5 * diagonal_sync_error
    #         - all_airborne
    #     )
    def _reward_trot_gait(self):
        contacts = self._get_foot_contacts()

        phase_angle = (
            2.0
            * np.pi
            * self.data.time
            / self.gait_period
        )

        if np.sin(phase_angle) >= 0:
            desired_contacts = {
                "FL": 1.0,
                "FR": 0.0,
                "RL": 0.0,
                "RR": 1.0,
            }
        else:
            desired_contacts = {
                "FL": 0.0,
                "FR": 1.0,
                "RL": 1.0,
                "RR": 0.0,
            }

        contact_error = np.mean(
            [
                (
                    float(contacts[name])
                    - desired_contacts[name]
                ) ** 2
                for name in ("FL", "FR", "RL", "RR")
            ]
        )

        return -contact_error
    # End


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        if self.viewer is None and self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.step_count = 0
        self.last_action.fill(0.0)
        self.forward_velocity = 0.0
        # Updates on Sep 3
        self.lateral_velocity = 0.0
        self.yaw_rate = 0.0
        # if self.step_count % 50 == 0:
        #     print(
        #         f"target_yaw={self.command_yaw:+.2f}, "
        #         f"actual_yaw={self.yaw_rate:+.2f}, "
        #         f"target_forward={self.command_x:+.2f}, "
        #         f"actual_forward={self.forward_velocity:+.2f}"
        #     )

        # 一半回合直行，一半回合左转
        # turn_left = bool(
        #     self.np_random.integers(0, 2)
        # )

        # if turn_left:
        #     self.command_x = 0.3
        #     self.command_yaw = -0.5
        # else:
        #     self.command_x = 0.5
        #     self.command_yaw = 0.0
        command_id = int(self.np_random.integers(0, 3))

        if command_id == 0:
            # 直行
            self.command_x = 0.5
            self.command_yaw = 0.0
        elif command_id == 1:
            # 左转
            self.command_x = 0.3
            self.command_yaw = 0.5
        else:
            # 右转
            self.command_x = 0.3
            self.command_yaw = -0.5
        # self.command_x = 0.3
        # self.command_yaw = -0.35
        # self.command_x = 0.5
        # self.command_yaw = 0
        # if self.np_random.random() < (1.0 / 3.0):
        #     # 保留足够的精确直行样本
        #     self.command_x = 0.5
        #     self.command_yaw = 0.0
        # else:
        #     # 连续、左右对称的转向角速度
        #     self.command_x = 0.3
        #     self.command_yaw = float(
        #         self.np_random.uniform(-0.5, 0.5)
        #     )
        # self.command_x = 0.5
        # self.command_yaw = 0.0
        return self._get_obs(), {}

    # def step(self, action):
    #     x_before = float(self.data.qpos[0])

    #     self.step_count += 1

    #     action = np.clip(action, -1.0, 1.0)
    #     target_pos = self.default_joint_pos + self.action_scale * action

    #     # 一个策略动作对应10个物理仿真步
    #     for _ in range(self.frame_skip):
    #         joint_pos = self.data.qpos[7:19]
    #         joint_vel = self.data.qvel[6:18]

    #         torque = (
    #             self.kp * (target_pos - joint_pos)
    #             - self.kd * joint_vel
    #         )

    #         ctrl_range = self.model.actuator_ctrlrange
    #         self.data.ctrl[:] = np.clip(
    #             torque,
    #             ctrl_range[:, 0],
    #             ctrl_range[:, 1],
    #         )

    #         mujoco.mj_step(self.model, self.data)

    #     # 以下内容必须在for循环结束以后执行
    #     obs = self._get_obs()
    #     reward = self._compute_reward(action)
    #     terminated = self._check_termination()
    #     truncated = self.step_count >= self.step_limit
    # def step(self, action):
    #     self.step_count += 1
    #     # x_before = float(self.data.qpos[0])
    #     self.forward_velocity = (
    #         float(self.data.qpos[0]) - x_before
    #     ) / self.dt

    #     action = np.clip(action, -1.0, 1.0)
    #     target_pos = self.default_joint_pos + self.action_scale * action

    #     for _ in range(self.frame_skip):
    #         joint_pos = self.data.qpos[7:19]
    #         joint_vel = self.data.qvel[6:18]

    #         torque = (
    #             self.kp * (target_pos - joint_pos)
    #             - self.kd * joint_vel
    #         )

    #         ctrl_range = self.model.actuator_ctrlrange
    #         self.data.ctrl[:] = np.clip(
    #             torque,
    #             ctrl_range[:, 0],
    #             ctrl_range[:, 1],
    #         )

    #         mujoco.mj_step(self.model, self.data)

    #     self._update_body_velocity()
    #     self.forward_velocity = (
    #         float(self.data.qpos[0]) - x_before
    #     ) / self.dt

    #     obs = self._get_obs()
    #     reward = self._compute_reward(action)
    #     terminated = self._check_termination()
    #     truncated = self.step_count >= self.step_limit
    def set_command(self, command_x, command_yaw):
        self.command_x = float(command_x)
        self.command_yaw = float(command_yaw)

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0)
        target_pos = (
            self.default_joint_pos
            + self.action_scale * action
        )

        for _ in range(self.frame_skip):
            joint_pos = self.data.qpos[7:19]
            joint_vel = self.data.qvel[6:18]

            torque = (
                self.kp * (target_pos - joint_pos)
                - self.kd * joint_vel
            )

            ctrl_range = self.model.actuator_ctrlrange
            self.data.ctrl[:] = np.clip(
                torque,
                ctrl_range[:, 0],
                ctrl_range[:, 1],
            )

            mujoco.mj_step(self.model, self.data)

        self._update_body_velocity()
        if self.step_count % 5 == 0:
            joint_angles = self.data.qpos[7:19].reshape(4, 3)

            parts = []

            for leg_name, angles in zip(
                ("FL", "FR", "RL", "RR"),
                joint_angles,
            ):
                hip, thigh, calf = angles

                parts.append(
                    f"{leg_name}:"
                    f"H={hip:+.3f}/"
                    f"T={thigh:+.3f}/"
                    f"C={calf:+.3f}"
                )

            # if self.step_count % 50 == 0:
            #     print(
            #         f"target_yaw={self.command_yaw:+.2f}, "
            #         f"actual_yaw={self.yaw_rate:+.2f}, "
            #         f"target_forward={self.command_x:+.2f}, "
            #         f"actual_forward={self.forward_velocity:+.2f}"
            #     )

            # print(
            #     f"step={self.step_count:04d} "
            #     + " ".join(parts)
            # )
        # foot_heights = {
        #     name: float(self.data.geom_xpos[geom_id][2])
        #     for name, geom_id in self.foot_geom_ids.items()
        # }

        # leg_actions = action.reshape(4, 3)
        # base_position = self.data.xpos[
        #     self.base_body_id
        # ].copy()

        # base_rotation = self.data.xmat[
        #     self.base_body_id
        # ].reshape(3, 3)

        # relative_foot_z = {}

        # for name, geom_id in self.foot_geom_ids.items():
        #     foot_world_position = self.data.geom_xpos[
        #         geom_id
        #     ]

        #     foot_body_position = (
        #         base_rotation.T
        #         @ (
        #             foot_world_position
        #             - base_position
        #         )
        #     )

        #     relative_foot_z[name] = float(
        #         foot_body_position[2]
        #     )

        # print(
        #     f"step={self.step_count:04d} "
        #     f"base_z={base_position[2]:.3f} "
        #     f"rel_FL={relative_foot_z['FL']:.3f} "
        #     f"rel_FR={relative_foot_z['FR']:.3f} "
        #     f"rel_RL={relative_foot_z['RL']:.3f} "
        #     f"rel_RR={relative_foot_z['RR']:.3f}"
        # )


        # print(
        #     f"step={self.step_count:04d} "
        #     f"h_FL={foot_heights['FL']:.3f} "
        #     f"h_FR={foot_heights['FR']:.3f} "
        #     f"h_RL={foot_heights['RL']:.3f} "
        #     f"h_RR={foot_heights['RR']:.3f} "
        #     f"a_FL={leg_actions[0].round(2)} "
        #     f"a_FR={leg_actions[1].round(2)} "
        #     f"a_RL={leg_actions[2].round(2)} "
        #     f"a_RR={leg_actions[3].round(2)}"
        # )
        # if self.step_count % 5 == 0:
        #     contacts = self._get_foot_contacts()
        #     slip = self._get_foot_slip_speeds(contacts)

        #     print(
        #         f"step={self.step_count:04d} "
        #         f"FL={int(contacts['FL'])}/{slip['FL']:.2f} "
        #         f"FR={int(contacts['FR'])}/{slip['FR']:.2f} "
        #         f"RL={int(contacts['RL'])}/{slip['RL']:.2f} "
        #         f"RR={int(contacts['RR'])}/{slip['RR']:.2f}"
        #     )
        # if self.step_count % 5 == 0:
        #     contacts = self._get_foot_contacts()
        #     slip = self._get_foot_slip_speeds(contacts)

        #     phase_angle = (
        #         2.0
        #         * np.pi
        #         * self.data.time
        #         / self.gait_period
        #     )

        #     if np.sin(phase_angle) >= 0:
        #         expected_pair = "FL+RR"
        #     else:
        #         expected_pair = "FR+RL"

        #     print(
        #         f"step={self.step_count:04d} "
        #         f"expected={expected_pair} "
        #         f"FL={int(contacts['FL'])}/{slip['FL']:.2f} "
        #         f"FR={int(contacts['FR'])}/{slip['FR']:.2f} "
        #         f"RL={int(contacts['RL'])}/{slip['RL']:.2f} "
        #         f"RR={int(contacts['RR'])}/{slip['RR']:.2f}"
        #     )
        #     print(
        #         f"step={self.step_count:04d} "
        #         f"FL={int(contacts['FL'])} "
        #         f"FR={int(contacts['FR'])} "
        #         f"RL={int(contacts['RL'])} "
        #         f"RR={int(contacts['RR'])}"
        #     )
        # if self.step_count % 50 == 0:
        #     print(
        #         f"target_yaw={self.command_yaw:+.2f}, "
        #         f"actual_yaw={self.yaw_rate:+.2f}, "
        #         f"forward={self.forward_velocity:+.2f}"
        #     )
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_termination()
        truncated = self.step_count >= self.step_limit

        info = {
            "x_position": float(self.data.qpos[0]),
            "x_velocity": float(self.data.qvel[0]),
            "simulation_time": float(self.data.time),
            "rew/vel": self._reward_tracking_velocity(),
            "rew/survive": self._reward_survival_bonus(),
            "rew/height": self._penalty_height(),
            "rew/posture": self._penalty_posture(),
            "rew/hip_limit": self._penalty_hip_duction(),
            "command_x": float(self.command_x),
            "command_yaw": float(self.command_yaw),
            "actual_yaw_rate": float(self.yaw_rate),
            "body_forward_velocity": float(
                self.forward_velocity
            ),
        }

        self.logger.update({
            "velocity": self._reward_tracking_velocity(),
            "height": self._penalty_height(),
            "head_height": self._penalty_head_height(),
            "y": self._penalty_lateral(),
            "head_y": self._penalty_head_lateral(),
            "posture": self._penalty_posture(),
            "torque": self._penalty_torque_effort(),
            "pose_penalty": self._penalty_pose_different(),
            "ang_velocity": self._reward_tracking_ang_vel(),
            "action_rate": self._penalty_action_rate(action),
            "hip_limit": self._penalty_hip_duction(),
            "survival bonus": self._reward_survival_bonus(),
        })

        self.last_action = np.copy(action)

        return obs, reward, terminated, truncated, info

    # Updates on Sep 3
    def _update_body_velocity(self):
        velocity_world = np.zeros(6, dtype=np.float64)

        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body_id,
            velocity_world,
            0,
        )


        # 前3项为角速度，后3项为线速度
        # self.yaw_rate = float(velocity[2])
        # self.forward_velocity = float(velocity[3])
        # self.lateral_velocity = float(velocity[4])
        angular_world = velocity_world[:3]
        linear_world = velocity_world[3:]

        # base 身体坐标系相对于世界坐标系的旋转矩阵
        body_rotation = self.data.xmat[
            self.base_body_id
        ].reshape(3, 3)

        # 世界坐标速度转换为机械狗自身坐标速度
        linear_body = body_rotation.T @ linear_world
        angular_body = body_rotation.T @ angular_world

        self.forward_velocity = float(linear_body[0])
        self.lateral_velocity = float(linear_body[1])
        self.yaw_rate = float(angular_body[2])
    # End
    # Update on Sep 26
    def _get_foot_contacts(self):
        foot_contacts = {
            name: False
            for name in self.foot_geom_ids
        }

        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1 = contact.geom1
            geom2 = contact.geom2

            for name, foot_geom_id in self.foot_geom_ids.items():
                foot_floor_contact = (
                    (
                        geom1 == foot_geom_id
                        and geom2 == self.floor_geom_id
                    )
                    or (
                        geom2 == foot_geom_id
                        and geom1 == self.floor_geom_id
                    )
                )

                if not foot_floor_contact:
                    continue

                contact_force = np.zeros(6)
                mujoco.mj_contactForce(
                    self.model,
                    self.data,
                    contact_index,
                    contact_force,
                )

                # contact_force[0] 是接触法向力
                if contact_force[0] > 1.0:
                    foot_contacts[name] = True

        return foot_contacts
    # End

    def _get_obs(self):
        # return np.nan_to_num(np.concatenate([self.data.qpos, self.data.qvel]), nan=0.0)
        command = np.array(
            [
                self.command_x,
                self.command_yaw,
            ],
            dtype=np.float64,
        )


        gait_clock = self._get_gait_clock()
        obs = np.concatenate(
        [
            self.data.qpos,
            self.data.qvel,
            command,
            gait_clock,
        ]
        )

        return np.nan_to_num(
            obs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    # def _check_termination(self):
    #     site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "head_tracker")
    #     head_pos = np.copy(self.data.site_xpos[site_id])
    #     y = self.data.qpos[1]
    #     z = self.data.qpos[2]

    #     # Extract rotation matrix
    #     R = self.data.xmat[0].reshape(3, 3)

    #     # Estimate roll and pitch from rotation matrix
    #     pitch = np.arcsin(-R[2, 0])         # around y-axis
    #     roll = np.arctan2(R[2, 1], R[2, 2]) # around x-axis

    #     # Too much side-to-side motion
    #     if np.abs(y) > 0.2 or np.abs(head_pos[1]) > 0.2:
    #         return True
    #     # Body too low or too high
    #     if z < 0.2 or z > 0.4:
    #         return True
    #     # Too much body rolling/rotation
    #     if abs(roll) > 0.2 or abs(pitch) > 0.2:
    #         return True
    #     return False
    def _check_termination(self):
        z = self.data.qpos[2]

        quat = self.data.qpos[3:7]
        rpy = R.from_quat(
            [quat[1], quat[2], quat[3], quat[0]]
        ).as_euler("xyz")

        roll, pitch, _ = rpy

        if not np.all(np.isfinite(self.data.qpos)):
            return True

        if z < 0.16 or z > 0.55:
            return True

        if abs(roll) > 0.8 or abs(pitch) > 0.8:
            return True

        return False

    def render(self):
        if self.render_mode == "rgb_array":
            if not hasattr(self, "_renderer"):
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data, camera="follow")
            return self._renderer.render()
        elif self.render_mode == "human" and self.viewer:
            self.viewer.sync()
            time.sleep(self.dt)


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.logger.write_averages()

    def _get_default_joint_pos(self):
        default_angles = {
            "FL_hip_joint": 0.0, "FR_hip_joint": 0.0, "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
        }
        joint_pos = np.copy(self.data.qpos)
        for name, angle in default_angles.items():
            idx = self.joint_indices.get(name)
            if idx is not None and idx < len(joint_pos):
                joint_pos[idx] = angle
        return joint_pos[:self.model.nu]

    # def _compute_reward(self, action):
    #     reward = (
    #         2 * self._reward_tracking_velocity()                # body moving forward
    #         - self._penalty_height()                            # penalize height deviation
    #         - self._penalty_head_height()                       # penalize height deviation
    #         - self._penalty_lateral()                           # penalize side-to-side body motion
    #         - self._penalty_head_lateral()                      # penalize side-to-side head motion
    #         - 10 * self._penalty_posture()                      # posture: penalize roll/pitch/yaw
    #         # - 0.000001 * self._penalty_torque_effort()        # penalize actuator strain
    #         - 0.05 * self._penalty_pose_different()             # pose penalty: penalize unnatural joint config
    #         + self._reward_tracking_ang_vel()                   # ang velocity close to target (0)
    #         - 0.00005 * self._penalty_action_rate(action)             # penalize changes in actions
    #         - 0.2 * self._penalty_hip_duction()                 # penalize hip joints folding inwards or outwards
    #         + self._reward_survival_bonus()                     # survival bonus
    #     )
    #     return reward
    def _compute_reward(self, action):
        vertical_velocity = self.data.qvel[2]
        roll_pitch_velocity = self.data.qvel[3:5]
        forward_progress = np.clip(
            self.forward_velocity
            / max(self.command_x, 0.1),
            -1.0,
            1.0,
        )

        reward = (
            4.0 * self._reward_tracking_velocity()

            # 必须产生正向净位移
            + 2.0 * np.clip(
                self.forward_velocity,
                -0.5,
                1.0,
            )
            # + 2.0 * forward_progress

            # 保持直立，但降低单纯活着的收益
            + 0.1 * self._reward_survival_bonus()
            + 4.0 * self._reward_tracking_ang_vel()

            # 姿态约束
            - 5.0 * self._penalty_posture()
            - 0.5 * self._penalty_height()
            #- 0.5 * self._penalty_height()
            - 2.0 * self._penalty_lateral()

            # 防止上下跳和剧烈摇晃
            - 1.0 * vertical_velocity ** 2
            - 0.1 * np.sum(
                np.square(roll_pitch_velocity)
            )

            # 平滑、节能
            - 0.05 * self._penalty_action_rate(action)
            - 0.0001 * np.mean(
                np.square(self.data.ctrl)
            )

            # 只轻微约束初始姿势
            - 0.005 * self._penalty_pose_different()
            - 0.1 * self._penalty_hip_duction()

            #关节速度
            - 0.001 * np.mean(np.square(self.data.qvel[6:18]))


            # 足端触地时不应水平滑动
            - 0.05 * self._penalty_foot_slip()

            # + 0.5 * self._reward_trot_gait()
            #  + 0.5 * self._score_trot_contacts(
            # self._get_foot_contacts()
            # )
            # 连续足端高度步态奖励
            + 2.0 * self._reward_foot_height_gait()
            # + 0.5 * self._reward_knee_flexion_gait()
            )

        return reward

    # def _reward_tracking_velocity(self):
    #     target = self.command_x  # 0.5 at the moment
    #     current = self.data.qvel[0]
    #     return np.exp(-((current - target) ** 2) / self.tracking_sigma)
    def _reward_tracking_velocity(self):
        error = self.forward_velocity - self.command_x
        return np.exp(
            -(error ** 2) / self.tracking_sigma
        )

    # def _reward_tracking_ang_vel(self):
    #     ang_vel_error = np.square(self.data.qvel[5] - self.command_yaw)
    #     return np.exp(-ang_vel_error / self.tracking_sigma)
    def _reward_tracking_ang_vel(self):
        error = self.yaw_rate - self.command_yaw

        return np.exp(
            -(error ** 2) / self.yaw_tracking_sigma
        )

    def _penalty_height(self):
        height_error = (
        self.data.qpos[2] - 0.27
        ) / 0.05
        return height_error** 2

    def _penalty_head_height(self):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "head_tracker")
        head_pos = np.copy(self.data.site_xpos[site_id])
        return (head_pos[2] - 0.37) ** 2

    def _penalty_lateral(self):
        # return (np.abs(self.data.qpos[1]))
        return self.lateral_velocity ** 2

    def _penalty_head_lateral(self):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "head_tracker")
        head_pos = np.copy(self.data.site_xpos[site_id])
        return (np.abs(head_pos[1]))

    def _penalty_posture(self):
        # quat = self.data.qpos[3:7]
        # rpy = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz", degrees=False)
        # return rpy[0] ** 2 + rpy[1] ** 2 + rpy[2] ** 2
        quat = self.data.qpos[3:7]

        rpy = R.from_quat(
            [quat[1], quat[2], quat[3], quat[0]]
        ).as_euler("xyz", degrees=False)

        roll, pitch, _ = rpy

        # 不惩罚yaw，因为之后需要学习转弯
        return roll ** 2 + pitch ** 2

    def _penalty_torque_effort(self):
        return np.sum(np.square(self.data.ctrl))

    def _penalty_action_rate(self,action):
        return np.mean(np.square(action - self.last_action))

    def _reward_survival_bonus(self):
        return 1.0

    def _penalty_hip_duction(self):
        """Penalize hip joints by how far they exceed the abduction/adduction limits (summed absolute violation)."""
        hip_angles = self.data.qpos[[7, 10, 13, 16]]
        return np.mean(np.square(hip_angles))

    # def _penalty_pose_different(self):
    #     # abduction, thigh joint, calf joint
    #     weights = [2, 1, 1]
    #     joint_pos = self.data.qpos[7:7 + self.model.nu]
    #     default_joint_pos = np.array([
    #         0, 0.9, -1.8,  # FL
    #         0, 0.9, -1.8,  # FR
    #         0, 0.9, -1.8,  # RL
    #         0, 0.9, -1.8   # RR
    #     ])
    #     abduction = np.sum(np.abs(joint_pos[[0, 3, 6, 9]] - default_joint_pos[[0, 3, 6, 9]]))
    #     thigh = np.sum(np.abs(joint_pos[[1, 4, 5, 10]] - default_joint_pos[[1, 4, 5, 10]]))
    #     calf = np.sum(np.abs(joint_pos[[2, 5, 8, 11]] - default_joint_pos[[2, 5, 8, 11]]))
    #     return np.dot(weights, [abduction, thigh, calf])

    def _penalty_pose_different(self):
        joint_pos = self.data.qpos[7:7 + self.model.nu]
        default_joint_pos = np.array([
            0, 0.9, -1.8,  # FL
            0, 0.9, -1.8,  # FR
            0, 0.9, -1.8,  # RL
            0, 0.9, -1.8   # RR
        ])
        return np.sum(np.abs(joint_pos - default_joint_pos))


class RewardLogger:
    def __init__(self):
        self.count = 0
        self.totals = {
            "velocity": 0.0,
            "height": 0.0,
            "head_height": 0.0,
            "y": 0.0,
            "head_y": 0.0,
            "posture": 0.0,
            "torque": 0.0,
            "pose_penalty": 0.0,
            "ang_velocity": 0.0,
            "action_rate": 0.0,
            "hip_limit": 0.0,
            "survival bonus": 0.0
        }

    def update(self, values):
        self.count += 1
        for key in self.totals:
            self.totals[key] += values[key]

    def write_averages(self, filename="reward_log.txt"):
        if self.count == 0:
            return
        with open(filename, "w") as f:
            f.write(f"Averages over {self.count} steps:\n")
            for key, total in self.totals.items():
                f.write(f"{key}: {total / self.count:.6f}\n")
