# rollout_eval_no_norm.py
import upkie.envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

upkie.envs.register()

MODEL_PATH = "./models/pendulum_best/best_model.zip"  # or "./models/ppo_upkie_final.zip"
ENV_ID = "Upkie-Spine-Pendulum"
ENV_KWARGS = dict(frequency=200.0)
SEED = 0

def main():
    env = make_vec_env(ENV_ID, n_envs=1, env_kwargs=ENV_KWARGS, seed=SEED)
    model = PPO.load(MODEL_PATH, env=env, device="auto")

    obs = env.reset()
    ep_return = 0.0
    ep_len = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        ep_return += float(reward[0])
        ep_len += 1
        if bool(dones[0]):
            break
    print(f"[EVAL] return={ep_return:.3f}, length={ep_len} steps")

if __name__ == "__main__":
    main()
