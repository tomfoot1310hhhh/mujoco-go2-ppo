import queue
import threading
from pathlib import Path

import gymnasium as gym
import numpy as np

import rl_zoo3.import_envs  # 注册 UnitreeGo2Command-v0

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "experiment_27"

MODEL_PATH = MODEL_DIR / "best_model.zip"
NORMALIZE_PATH = MODEL_DIR / "vecnormalize.pkl"


COMMANDS = {
    "直行": (0.5, 0.0),
    "前进": (0.5, 0.0),
    "左转": (0.3, -0.5),
    "左": (0.3, -0.5),
    "右转": (0.3, 0.5),
    "右": (0.3, 0.5),
}


command_queue = queue.Queue()


def read_commands():
    while True:
        text = input(
            "\n请输入：直行 / 左转 / 右转 / 退出\n> "
        ).strip()

        command_queue.put(text)

        if text == "退出":
            break


def make_env():
    return gym.make(
        "UnitreeGo2Command-v0",
        render_mode="human",
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型：{MODEL_PATH}"
        )

    if not NORMALIZE_PATH.exists():
        raise FileNotFoundError(
            f"找不到归一化文件：{NORMALIZE_PATH}"
        )

    vec_env = DummyVecEnv([make_env])

    vec_env = VecNormalize.load(
        NORMALIZE_PATH,
        vec_env,
    )
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(
        MODEL_PATH,
        device="cpu",
    )

    vec_env.reset()

    current_command = COMMANDS["直行"]

    input_thread = threading.Thread(
        target=read_commands,
        daemon=True,
    )
    input_thread.start()

    print("实验 27 已载入")
    print("当前命令：直行")
    print("可以在终端输入：直行、左转、右转、退出")

    running = True

    try:
        while running:
            while not command_queue.empty():
                text = command_queue.get()

                if text == "退出":
                    running = False
                    break

                if text not in COMMANDS:
                    print(f"无法识别命令：{text}")
                    continue

                current_command = COMMANDS[text]

                print(
                    f"切换命令：{text} "
                    f"(x={current_command[0]:.1f}, "
                    f"yaw={current_command[1]:+.1f})"
                )

            if not running:
                break

            command_x, command_yaw = current_command

            # 每一步都写入当前指令，防止回合重置后丢失
            vec_env.env_method(
                "set_command",
                command_x,
                command_yaw,
            )

            # set_command 后重新取得包含新指令的 observation
            raw_obs = np.asarray(
                vec_env.env_method("_get_obs"),
                dtype=np.float64,
            )
            obs = vec_env.normalize_obs(raw_obs)

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            vec_env.step(action)

    except KeyboardInterrupt:
        print("\n控制已停止")

    finally:
        vec_env.close()


if __name__ == "__main__":
    main()