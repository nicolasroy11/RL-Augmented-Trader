from RL.base_environment import BaseTradingEnv
from db.data_store import db_path


if __name__ == "__main__":
    env = BaseTradingEnv(db_path, window_size=10)
    obs = env.reset()
    print("Initial observation shape:", obs.shape)
    done = False
    prev_obs = None
    while not done:
        obs, reward, done, info = env.step(action=None)
        print("Step:", env.current_step, "Observation shape:", None if obs is None else obs.shape)
        if obs is not None:
            print("First row of obs:", obs[0])
            if prev_obs is not None:
                # Compare with previous obs last row
                print("Previous obs last row:", prev_obs[-1])
            prev_obs = obs
