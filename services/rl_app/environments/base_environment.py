from typing import List
import numpy as np
import pandas as pd
import torch
from RL.playground.stochastic.actor_critic import ActorCritic
import runtime_settings
from services.core.models import TickData, TrainingFields
from services.types import Actions


class BaseTradingEnvironment:

    def __init__(self, tick_df: pd.DataFrame = None, tick_list: List[TickData] = None):
        if not tick_df and not tick_list:
            raise ValueError('need to pass in at tick_df or tick_list!')
        self.window_size = runtime_settings.DATA_TICKS_WINDOW_LENGTH
        if tick_list:
            tick_df = TickData.list_to_env_dataframe(tick_list=tick_list)
        self.data = tick_df
        self.current_step = 0
        self.position: int = 0
        self.entry_price: float = None


    def get_ordered_ticks(self):
        ticks = TickData.objects.all().order_by("timestamp").values()
        return ticks
    

    def reset(self):
        self.current_step = self.window_size
        return self._get_observation()


    def _get_observation(self) -> np.ndarray:
        window = self.data.iloc[self.current_step - self.window_size : self.current_step]
        window = TickData.remove_non_training_fields_in_df(df=window)
        return window.to_numpy()


    def get_normalized_observation(self):
        obs_df = self.data.iloc[self.current_step - self.window_size : self.current_step].copy()
        input_features_vector = TrainingFields.ObservationFeatures.normalized_vector_from_obs_df(obs_df, self.position, self.entry_price)
        return input_features_vector


    def get_inference_action_probs(self):
        self.load_inference_policy()
        self.current_step = self.window_size
        obs_vector = self.get_normalized_observation()
        with torch.no_grad():
            state_tensor = torch.tensor(obs_vector, dtype=torch.float32, device=self.policy_device).flatten().unsqueeze(0)
            logits, _ = self.policy(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            probs = dist.probs.squeeze(0).cpu().numpy()
            return {
                Actions.SELL: float(probs[0]),
                Actions.HOLD: float(probs[1]),
                Actions.BUY: float(probs[2]),
            }
        

    def load_inference_policy(self):
        vector_size = TrainingFields.ObservationFeatures.get_vector_size()
        input_dim = vector_size
        action_dim = 3  # sell, hold, buy
        self.policy_device = torch.device("cpu")
        self.policy = ActorCritic(input_dim=input_dim, action_dim=action_dim).to(self.policy_device)

        state_dict = torch.load(runtime_settings.RL_POLICY_PATH, map_location=self.policy_device)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()


    def normalized_reset(self):
        self.current_step = self.window_size
        return self.get_normalized_observation()


    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data)
        obs = self._get_observation() if not done else None
        reward = 0.0
        info = {}
        return obs, reward, done, info
