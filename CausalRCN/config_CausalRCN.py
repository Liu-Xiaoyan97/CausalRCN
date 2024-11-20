from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import Dict, Optional

logger = logging.get_logger(__name__)

class DatasetCategoriesMapping:
    def __init__(self) -> None:
        self.dcmapping = {
            "lm": 4096,
            "agnews": 4,
            "amazon": 2,
            "dbpedia": 14,
            "yelp": 2,
            "imdb": 2,
            "hyper": 2,
            "cola": 2,
            "mnli": 3,
            "qnli": 2,
            "rte": 2,
            "mrpc": 2,
            "qqp": 2,
            "sst": 2,
            "wnli": 2,
            "mr": 2,
            "mpqa": 2,
            "subj": 2,
            "cr": 2
        }

class CausalRCNConfig(PretrainedConfig):
    model_type = "CausalRCN_LightningLM"

    def __init__(self, 
                task:  str = "imdb",
                finetune: bool = False,
                n_layers: int=4,
                embedding_dim: int=32,
                projection_dim: int=128,
                drop_rate: float=0.3,
                # batch_size: int=768,
                batch_size: int=16,
                window_size: int=5,
                num_workers: int=8,
                lr: float=1e-4,
                weight_decy: float=0.1, 
                log_every_n_steps: int=4,
                val_check_interval: float=0.1,
                check_val_every_n_epoch: int=1,
                accelerator: str="auto",
                devices: str="0, 1, 2, 3, 4, 5, 6, 7",
                # devices: str="0, 1, 2, 3",
                # devices: str=1,
                max_epochs: int=9999,
                vocab_size: int=4096,
                activation: str="silu",
                ablation: str="cnn",
                use_hashembedding: bool = False,
                use_aux_loss: bool = False,
                use_rnn: bool = True,
                ffn_type: Optional[str] = None,
                pad_token_id: int = 0,
                unk_token_id: int = 4,
                bos_token_id: int = 1, 
                eos_token_id: int = 2,
                sep_token_id: int = 3,
                **kwargs
    ):
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.finetune = finetune
        self.n_layers =  n_layers
        self.embedding_dim = embedding_dim
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.activation = activation
        self.task = task
        dcmapping = DatasetCategoriesMapping()
        self.categories = dcmapping.dcmapping[task]
        self.use_hashembedding = use_hashembedding
        self.use_rnn = use_rnn
        self.ffn_type = ffn_type
        self.window_size = window_size
        self.use_aux_loss = use_aux_loss
        self.ablation = ablation 
        self.optimizer = {
            "lr": lr,
            "weight_decay": weight_decy
        }
        self.trainer = {
            "log_every_n_steps": log_every_n_steps,
            "accelerator": accelerator,
            "devices": devices,
            "max_epochs": max_epochs,
            "val_check_interval": val_check_interval if task == "lm" else 0.9,
            "check_val_every_n_epoch": check_val_every_n_epoch
        }
        super().__init__(
            pad_token_id=pad_token_id,
            unk_token_id=unk_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )

# if __name__ == "__main__":
#     CausalRCNconfig = CausalRCNConfig()
#     print(CausalRCNconfig.pad_token_id)
