import wandb
import os

sweep_config = {
    'program': 'run_ala_ddp_opt_gpu.py',
    'method': 'bayes'
    }
metric = {
    'name': 'val_loss',
    'goal': 'minimize'
    }

sweep_config['metric'] = metric

hyperparameters_dict = {
    'logdir': {
        'value': './RBCG-sarscov2'
    },
    'lr': {
        "distribution": "log_uniform_values",
        "max": 0.0002,
        "min": 6e-05,
    },
    'n_basis': {
        "distribution": "int_uniform",
        "min": 500,
        "max": 600
    },
    'n_rbf': {
        'value': 6
    },
    'activation': {
        'value': "swish"
    },
    'cg_method': {
        'value': 'cgae'
    },
    'atom_cutoff': {
        'distribution': 'uniform',
        'min': 20,
        'max': 50
    },
    'optimizer': {
        'value': 'adam'
    },
    'cg_cutoff': {
        'distribution': 'uniform',
        'min': 50,
        'max': 80
    },
    'enc_nconv': {
        'distribution': 'int_uniform',
        'min': 3,
        'max': 5
    },
    'dec_nconv': {
        'distribution': 'int_uniform',
        'min': 6,
        'max': 10
    },
    'batch_size': {
        'value': 2
    },
    'nepochs': {
        'value': 30
    },
    'nsamples': {
        'value': 40
    },
    'n_ensemble': {
        'value': 2
    },
    'nevals': {
        'value': 36
    },
    'edgeorder': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 3
    },
    'auxcutoff': {
        'value': 0.0
    },
    'beta': {
        'distribution': 'log_uniform_values',
        'min': 0.0001,
        'max': 0.1
    },
    'gamma': {
        'distribution': 'log_uniform_values',
        'min': 0.5,
        'max': 30.0
    },
    'threshold': {
        'value': 1e-3
    },
    'patience': {
        'value': 10
    },
    'factor': {
        'distribution': 'log_uniform_values',
        'min': 0.1,
        'max': 0.9
    },
    'cgae_reg_weight': {
        'value': 0.25
    },
    'dec_type': {
        'value': 'EquivariantDecoder'
    },
    'dataset': {
        'value': 'RBCG-sarscov2'
    },
    'n_cgs': {
        'value': 25
    },
    'n_data': {
        'value': 3000
    },
}

sweep_config['parameters'] = hyperparameters_dict
early_terminate = {
    'type': 'hyperband',
    'min_iter': 20,
    }
sweep_config['early_terminate'] = early_terminate
sweep_id = wandb.sweep(sweep_config, project="sequential_backmapping")
# save sweep_id to file
with open('sweep_id.txt', 'w') as f:
    f.write(sweep_id)