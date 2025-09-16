from uuid import UUID
from services.core.ML.configurations.PPO_flattened_history.environment import Environment as PPO_flattened_history_env
from services.core.ML.configurations.PPO_temporal_tcn.environment import Environment as PPO_temporal_tcn_env


CONFIG_UUIDS = {
    PPO_flattened_history_env: UUID('9966a6fa-632b-4cbf-8af0-a7045c4584ee'),
    PPO_temporal_tcn_env: UUID('07dd9189-5b43-44d7-8077-5355b2d76e2a')
}

DEFAULT_FEATURE_SET_ID = UUID("33333333-3333-3333-3333-333333333333")
PPO_TCN_FUTURES_SET_ID = UUID("ae1a159c-72f8-4c48-aa36-08a567fdba15")