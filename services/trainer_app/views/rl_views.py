from django.core.handlers.wsgi import WSGIRequest
from django.http import JsonResponse
import runtime_settings
from services.core.ML.configurations.XGBoost.xgboost_strategy import XGBMultiClassModel, run_futures_xgb_pipeline, train_test_walk_forward_on_futures
from services.core.ML.configurations.fixture_config import PPO_TCN_FUTURES_SET_ID
from services.core.dtos.policy_gradient_results_dto import PPOTCNTrainingResults
from services.decorators.decorators.view_decorator import View
from services.decorators.decorators.view_class_decorator import ViewClass
from services.core.ML.configurations.PPO_flattened_history.train import RLRepository
from services.core.ML.configurations.PPO_temporal_tcn.train import RLRepository as PPO_TCN_Repo


@ViewClass(
    url='training/'
)
class RLViews:

    class Meta:
        app_label = 'services.trainer_app'
        label = 'training'


    @View(
        path='run_policy_gradient',
        http_method='GET',
        return_type=PPOTCNTrainingResults.Serializer(),
        description='Run policy gradient algorithm and return results',
        include_in_swagger=True
    )
    def run_policy_gradient(req: WSGIRequest):
        def exec():
            rl_repo = RLRepository()
            results = rl_repo.run_policy_gradient(window_size=runtime_settings.DATA_TICKS_WINDOW_LENGTH, num_episodes=100)
            dto = PPOTCNTrainingResults.Serializer(results).data
            return JsonResponse(dto)
        return exec()


    @View(
        path='run_ppo',
        http_method='POST',
        return_type=PPOTCNTrainingResults.Serializer(),
        description='Run policy gradient algorithm and return results',
        include_in_swagger=True
    )
    def run_ppo(req: WSGIRequest):
        feature_set = req.GET.get('feature_set')
        # num_episodes=100, gamma=0.99, lr=1e-4, clip_epsilon=0.2, ppo_epochs=4, batch_size=64
        def exec():
            rl_repo = RLRepository()
            results = rl_repo.run_ppo(window_size=runtime_settings.DATA_TICKS_WINDOW_LENGTH, num_episodes=100)
            dto = PPOTCNTrainingResults.Serializer(results).data
            return JsonResponse(dto)
        return exec()
    

    @View(
        path='run_ppo_futures',
        http_method='POST',
        return_type=PPOTCNTrainingResults.Serializer(),
        description='Run policy gradient algorithm and return results',
        include_in_swagger=True
    )
    def run_ppo(req: WSGIRequest):
        feature_set = req.GET.get('feature_set')
        # num_episodes=100, gamma=0.99, lr=1e-4, clip_epsilon=0.2, ppo_epochs=4, batch_size=64
        def exec():
            rl_repo = RLRepository()
            results = rl_repo.run_ppo(window_size=runtime_settings.DATA_TICKS_WINDOW_LENGTH, num_episodes=100)
            dto = PPOTCNTrainingResults.Serializer(results).data
            return JsonResponse(dto)
        return exec()
    
    
    @View(
        path='run_ppo_tcn_futures',
        http_method='POST',
        return_type=PPOTCNTrainingResults.Serializer(),
        description='Run policy gradient algorithm and return results',
        include_in_swagger=True
    )
    def run_ppo_tcn_futures(req: WSGIRequest):
        feature_set = req.GET.get('feature_set')
        # num_episodes=100, gamma=0.99, lr=1e-4, clip_epsilon=0.2, ppo_epochs=4, batch_size=64
        def exec():
            rl_repo = PPO_TCN_Repo()
            results = rl_repo.run_ppo(window_size=runtime_settings.DATA_TICKS_WINDOW_LENGTH, feature_set_id = PPO_TCN_FUTURES_SET_ID,num_episodes=100, is_futures=True)
            dto = PPOTCNTrainingResults.Serializer(results).data
            return JsonResponse(dto)
        return exec()
    

    @View(
        path='run_xgboost_futures',
        http_method='POST',
        return_type=PPOTCNTrainingResults.Serializer(),
        description='Run policy gradient algorithm and return results',
        include_in_swagger=True
    )
    def run_xgboost_futures(req: WSGIRequest):
        feature_set = req.GET.get('feature_set')
        def exec():

            result = run_futures_xgb_pipeline(
                horizon_ticks=6,
                fee_bps=2.0,
                min_edge_bps=1.0,
                xgb_params=None,          # or dict of overrides
                n_splits=5,
                min_confidence=0.40,
                fee_bps_roundtrip=4.0,
                artifact_name="futures_walkforward.json",
                # env=my_env,               # optional: run backtest on your env
                # env_results_path="/tmp/env_bt.csv",  # optional
            )
            print(result)

            # return JsonResponse(dto)
        return exec()