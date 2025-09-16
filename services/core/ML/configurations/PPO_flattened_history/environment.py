from typing import List
import numpy as np
import pandas as pd
import torch
from RL.playground.stochastic.actor_critic import ActorCritic
import runtime_settings
from services.core.models import DerivedfeatureSetMapping, FeatureSet, TickData
from services.core.services.feature_service import DerivedFeatureMethods
from services.types import Actions


class Environment:

    def __str__():
        return 'A PPO with n-tick history flattened to a 1-D vector'

    def __init__(self, feature_set: FeatureSet, tick_df: pd.DataFrame = None, tick_list: List[TickData] = None, initial_cash=1000000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.feature_set = feature_set
        self.feature_set_flat_vector_size = feature_set.get_feature_vector_size()
        # if not tick_df and not tick_list:
        #     raise ValueError('need to pass in at tick_df or tick_list!')
        self.window_size = runtime_settings.DATA_TICKS_WINDOW_LENGTH
        if tick_list:
            tick_df = TickData.list_to_env_dataframe(tick_list=tick_list)
        self.data = tick_df
        self.current_step = 0
        self.position: int = 0
        self.entry_price: float = None
        self.inference_policy_loaded = False


    def get_ordered_ticks(self):
        ticks = TickData.objects.all().order_by("timestamp").values()
        return ticks
    

    def reset(self):
        self.current_step = self.window_size
        return self._get_observation()


    def get_normalized_observation(self):
        obs_df = self.data.iloc[self.current_step - self.window_size : self.current_step].copy()
        obs_df = TickData.remove_non_training_fields_in_df(df=obs_df)
        obs_df = obs_df.drop(columns=[col for col in obs_df.columns if obs_df[col].isna().all()])
        input_features_vector = self.normalized_vector_from_obs_df(obs_df, self.position, self.entry_price)
        return input_features_vector


    def normalized_vector_from_obs_df(self, obs_df: pd.DataFrame, position, entry_price):
            
            price = obs_df['price'].iloc[-1]
            
            # === Derived features ===
            derived_features = {}
            derived_feature_mappings = list(DerivedfeatureSetMapping.objects.filter(feature_set_id=self.feature_set.id))
            m: DerivedfeatureSetMapping
            for m in derived_feature_mappings:
                if m.derived_feature.method_name.startswith('position_'):
                    value = getattr(DerivedFeatureMethods, m.derived_feature.method_name)(position, entry_price, price)
                    derived_features[m.derived_feature.method_name] = value
                elif m.derived_feature.method_name.startswith('threshold_'):
                    value = getattr(DerivedFeatureMethods, m.derived_feature.method_name)(obs_df)
                    derived_features[m.derived_feature.method_name] = value
                else:
                    value = getattr(DerivedFeatureMethods, m.derived_feature.method_name)(obs_df)
                    derived_features[m.derived_feature.method_name] = value

            # === Normalize base features ===
            base_features = TickData.remove_non_training_fields_in_df(df=obs_df)
            base_features = base_features.to_numpy().astype(np.float32)
            base_mean = base_features.mean(axis=0)
            base_std = base_features.std(axis=0) + 1e-8
            base_features_norm = ((base_features - base_mean) / base_std).flatten()

            # === Normalize derived features ===
            derived_array = np.array(list(derived_features.values()), dtype=np.float32)
            derived_mean = derived_array.mean()
            derived_std = derived_array.std() + 1e-8
            derived_features_norm = (derived_array - derived_mean) / derived_std

            # === Final observation vector ===
            final_vector = np.concatenate([base_features_norm, derived_features_norm])
            return final_vector
    

    def get_inference_action_probs(self):
        self.current_step = self.feature_set.window_length
        obs_vector = self.get_normalized_observation()
        with torch.no_grad():
            state_tensor = self.get_state_tensor(obs_vector=obs_vector)
            logits, _ = self.policy(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            probs = dist.probs.squeeze(0).cpu().numpy()
            return {
                Actions.SELL: float(probs[0]),
                Actions.HOLD: float(probs[1]),
                Actions.BUY: float(probs[2]),
            }
        

    def get_state_tensor(self, obs_vector: np.ndarray):
        t = torch.tensor(obs_vector, dtype=torch.float32, device=self.policy_device).flatten().unsqueeze(0)
        return t
        

    def load_inference_policy(self):
        if not self.inference_policy_loaded: self.load_inference_policy()
        input_dim = self.feature_set_flat_vector_size
        action_dim = 3  # sell, hold, buy
        self.policy_device = torch.device("cpu")
        self.policy = ActorCritic(input_dim=input_dim, action_dim=action_dim).to(self.policy_device)
        state_dict = torch.load(runtime_settings.RL_POLICY_PATH, map_location=self.policy_device)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()


    def normalized_reset(self):
        self.current_step = self.window_size
        return self.get_normalized_observation()


    def step(self, action: int):
        """
        Actions:
            -1 = Sell
            0 = Hold
            1 = Buy

        Position state:
            self.position: 0 = flat, 1 = long
            self.entry_price: entry price if long, None if flat
        """
        price = self.data.iloc[self.current_step]['price']
        realized_pnl = 0.0

        # === Execute action ===
        if action == 1:  # BUY
            if self.position == 0:  # only open a new position if flat
                self.position = 1
                self.entry_price = price

        elif action == -1:  # SELL
            if self.position == 1:  # only sell if long
                realized_pnl = price - self.entry_price
                self.cash += realized_pnl  # accumulate realized gains/losses
                self.position = 0
                self.entry_price = None

        # HOLD (0) does nothing

        # === Advance environment ===
        self.current_step += 1
        done = self.current_step >= len(self.data)

        # === Get next observation ===
        obs = self.get_normalized_observation() if not done else None

        # === Reward: realized PnL only ===
        reward = realized_pnl

        # === Info dict for debugging/logging ===
        unrealized_pnl = (price - self.entry_price) if self.position == 1 else 0.0
        info = {
            "cash": self.cash,
            "position": self.position,
            "entry_price": self.entry_price if self.entry_price else 0.0,
            "price": price,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
        }

        return obs, reward, done, info
