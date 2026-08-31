from pathlib import Path

from btc_forecasting.common.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_deep_training_protocol_v1_is_fully_frozen_for_e03() -> None:
    model = load_yaml(ROOT / 'configs' / 'models' / 'lstm.yaml')['model']
    training = load_yaml(ROOT / 'configs' / 'training.yaml')['training']
    experiment = load_yaml(ROOT / 'configs' / 'experiments' / 'e03.yaml')['experiment']

    assert experiment['seed'] == 42
    assert experiment['training_protocol'] == 'deep_training_v1'
    assert experiment['status'] == 'ready_for_training'

    assert model['hidden_size'] == 64
    assert model['num_layers'] == 1
    assert model['dropout'] == 0.20
    assert model['bidirectional'] is False
    assert model['output_size'] == 1
    assert model['readout'] == {
        'state': 'final_hidden_state_of_top_layer',
        'expression': 'h_n[-1]',
        'carry_state_between_batches': False,
    }
    assert model['regression_head'] == {
        'type': 'linear',
        'in_features': 'hidden_size',
        'out_features': 1,
        'activation': 'none',
        'additional_dropout': 'none',
    }

    assert training['protocol_version'] == 'deep_training_v1'
    assert training['device'] == {
        'official': 'cuda',
        'require_available': True,
        'run_metadata': ['actual_device', 'gpu_name'],
    }
    assert training['optimizer'] == {
        'type': 'adam',
        'learning_rate': 0.001,
        'betas': [0.9, 0.999],
        'eps': 1.0e-8,
        'weight_decay': 0.0,
        'amsgrad': False,
    }
    assert training['loss'] == {
        'type': 'torch.nn.HuberLoss',
        'delta': 0.01,
        'reduction': 'mean',
        'target_scale': 'unscaled_one_hour_log_return',
    }
    assert training['batch_size'] == 128
    assert training['data_loader'] == {
        'num_workers': 0,
        'generator_seed': 'experiment_seed',
        'train': {'shuffle': True, 'drop_last': False},
        'validation': {'shuffle': False, 'drop_last': False},
    }
    assert training['max_epochs'] == 30
    assert training['early_stopping'] == {
        'enabled': True,
        'patience': 5,
        'monitor': 'validation_huber_loss',
        'mode': 'min',
        'min_delta': 0.0,
        'improvement': 'strict',
        'equal_is_improvement': False,
        'bad_epoch_increment': 'every_non_improving_epoch',
        'stop_condition': 'bad_epochs_greater_than_or_equal_to_patience',
        'restore': 'best_validation_loss_checkpoint',
    }
    assert training['gradient_clipping'] == {'max_norm': 1.0, 'norm_type': 2.0}
    assert training['scheduler'] == {
        'type': 'torch.optim.lr_scheduler.CosineAnnealingLR',
        'T_max': 'configured_max_epochs',
        'eta_min': 0.0,
        'step': 'end_of_each_completed_training_epoch',
    }
    assert training['feature_scaler'] == {'type': 'robust', 'fit_scope': 'train_only'}
    assert training['target_scaler'] == {'type': 'none', 'fit_scope': 'not_applicable'}
    assert training['determinism'] == {
        'seed_sources': [
            'python_random',
            'numpy',
            'torch',
            'torch_cuda_manual_seed_all',
        ],
        'seed_value': 'experiment_seed',
        'torch_backends_cudnn_benchmark': False,
        'torch_backends_cudnn_deterministic': True,
        'stronger_deterministic_algorithms': False,
    }
    assert training['mixed_precision'] is False
    assert training['checkpoint_policy'] == 'best_validation_loss'
