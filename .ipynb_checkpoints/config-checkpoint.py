from dataclasses import dataclass, field

@dataclass
class hyperparameter:
    Learning_rate: float = 0.00005
    Epoch: int = 100
    Batch_size: int = 64
    Patience: int = 50
    Decay_interval: int = 10
    Lr_decay: float = 1e-5
    weight_decay: float = 1e-4
    Loss_epsilon: float = 1.0

    char_dim: int = 512
    hidden_dim: int = 128
    protein_kernel: list = field(default_factory=lambda: [4, 6, 8])
    drug_kernel: list = field(default_factory=lambda: [3, 3, 3])
    conv: int = 128

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid hyperparameter.")

def FMCAargs(hp: hyperparameter):
    config = {
        'embed_d_size': hp.char_dim,
        'embed_p_size': hp.char_dim,
        'd_channel_size': [[19, hp.conv], [19, 256, hp.conv], [19, 128, 256, hp.conv]],
        'p_channel_size': [[181, hp.conv], [19, 256, hp.conv], [19, 128, 256, hp.conv]],
        'lstm_hidden': hp.hidden_dim,
        'num_lstm_layers': 2,
        'fc_size': [1024, 512, 256],
        'clip': {'enabled': True, 'value': 5.0},
        'max_drug_seq': {"celegans": [19, 11], "human": [20, 21]},
        'max_protein_seq': {"celegans": 181, "human": 184},
        'input_d_dim': {"celegans": [2184, 1804], "human": [3269, 2658]},
        'input_p_dim': {"celegans": 224, "human": 226},
    }
    return config