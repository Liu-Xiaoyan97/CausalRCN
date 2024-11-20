import lightning as L
from lightning.pytorch.callbacks import ThroughputMonitor
from lightning.fabric.utilities.throughput import measure_flops
from CausalRCN.utils import CausalRCNDataModule, CausalRCNModule
import torch
from CausalRCN.config_CausalRCN import CausalRCNConfig
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
torch.set_float32_matmul_precision('medium')
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="funeturing from checkpoint")
    parser.add_argument("--ckpt", type=str, default=None, help="model ckpt path for funeturing")
    # parser.add_argument("--task", type=str, default="lm", help='''select dataset  from \n ["lm", "agnews, "amazon", "dbpedia", "hyper", "imdb", "yelp"] \n''')
    CausalRCNconfig = CausalRCNConfig()
    if CausalRCNconfig.task == "lm":
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", 
                                        filename='mixer-best-{epoch:04d}-{val_loss:.9f}',
                                        save_top_k=1,
                                        mode='min',
                                        save_last=True
                                        )
    else:
        checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", 
                                            filename='mixer-best-{epoch:04d}-{val_loss:.4f}-{val_f1score:.4f}-{val_accuracy:.4f}-{val_recall:.4f}',
                                            save_top_k=1,
                                            mode='max',
                                            save_last=True
                                            )
    # lr_scheduler_callback = CustomLRScheduler(CausalRCNconfig.optimizer["lr"], 1e-5, 10)
    args = parser.parse_args()
    # CausalRCNConfig.task = args.task
    # trainer = L.Trainer(**CausalRCNconfig.trainer, callbacks=[checkpoint_callback, lr_scheduler_callback])
    trainer = L.Trainer(**CausalRCNconfig.trainer, callbacks=[checkpoint_callback])
    dm = CausalRCNDataModule(CausalRCNconfig)
    if args.ckpt is not None:
        if CausalRCNconfig.finetune:
            model = CausalRCNModule.load_from_checkpoint(checkpoint_path=args.ckpt, config=CausalRCNconfig)
        else:
            model = CausalRCNModule.load_from_checkpoint(checkpoint_path=args.ckpt)
    else:
        model = CausalRCNModule(CausalRCNconfig)
    dm.setup("fit")

    trainer.fit(model, datamodule=dm)
        

