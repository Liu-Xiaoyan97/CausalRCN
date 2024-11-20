from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch import nn as nn
from CausalRCN.Mixers import CausalRCN, Causal_CNN
from torchmetrics import Accuracy, F1Score, Recall, Precision
from lightning import LightningModule, LightningDataModule
from .hashembedding import HashEmbedding
from transformers.configuration_utils import PretrainedConfig
import torch.nn.functional as F
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()
from torch.utils.data import DataLoader
from dataset_map import DatasetMap
import torch.utils.data as Data
from sru import SRU


class CausalRCNModule(LightningModule):
    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.ablation = config.ablation
        if self.ablation == "rcn":
            self.MHBAMixers = CausalRCN(
                n_layers=config.n_layers,
                hidden_dim=config.projection_dim,
                window_size=config.window_size,
                drop_rate=config.drop_rate,
                activation=config.activation,
                use_rnn=config.use_rnn,
                ffn_type = config.ffn_type
            )
        elif self.ablation == "sru":
            self.MHBAMixers = SRU(
                config.projection_dim,
                hidden_size=config.projection_dim,
                num_layers=config.n_layers,
                dropout=config.drop_rate,
                highway_bias=-2, 
                rescale=True,
                use_tanh=True
                )
        else:
            self.MHBAMixers = Causal_CNN(config.n_layers,
                                         config.projection_dim,
                                         config.window_size,
                                         config.activation,
                                         config.drop_rate)
        self.task = config.task
        self.finetune = config.finetune
        self.batch_size = config.batch_size,
        self.optimizerconfig = config.optimizer
        self.pad_token_id = config.pad_token_id
        self.token_shift = nn.ConstantPad1d((-1, 1), self.pad_token_id)
        if config.use_hashembedding:
            self.embeddings = HashEmbedding(config.vocab_size, config.embedding_dim, append_weight=False)
        else:
            self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.type_embedding = nn.Embedding(4, config.embedding_dim)
        self.postNorm = nn.LayerNorm(config.projection_dim)
        self.projection = nn.Linear(config.embedding_dim, config.projection_dim)
        self.drop_out = nn.Dropout(config.drop_rate)
        self.llmhead = nn.Linear(config.projection_dim, config.categories, bias=False)
        self.categories = config.categories
        self.embedding_dim = config.embedding_dim
        self.projection_dim = config.projection_dim
        self.use_aux_loss = config.use_aux_loss

        self.accuary = Accuracy(task="multiclass", num_classes=config.categories)
        self.recall = Recall(task="multiclass", average='macro', num_classes=config.categories)
        self.f1score = F1Score(task="multiclass", average='macro', num_classes=config.categories)
        self.precision = Precision(task="multiclass", average='macro', num_classes=config.categories)
        self.apply(self._init_weights)

    def on_load_checkpoint(self, checkpoint, *args, **kwargs):
        state_dict = checkpoint['state_dict']
        self.llmhead = nn.Linear(self.projection_dim, self.categories, bias=False)
        state_dict.update({'llmhead.weight': torch.normal(mean=0.0, std=0.1, size=(self.categories, self.projection_dim))})
        self.load_state_dict(state_dict, strict=False)
        del state_dict

    def _init_weights(self, module):
            if isinstance(module, (nn.Linear)):
                module.weight.data.normal_(mean=0.0, std=0.1)
            if isinstance(module, (nn.Parameter)):
                module.weight.data.normal_(mean=0.0, std=0.0005)
            if isinstance(module, (nn.Conv1d)):
                module.weight.data.normal_(mean=0.0, std=0.0005)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids: torch.tensor, type_ids: torch.tensor = None, attention_mask: torch.tensor = None):
        embeddings = self.embeddings(input_ids)
        if type_ids != None:
            type_embedding = self.type_embedding(type_ids)
            embeddings = embeddings + type_embedding
        if attention_mask != None:
            embeddings = embeddings * attention_mask[:, :, None].expand(embeddings.size())
        pn_embeddings = self.drop_out(embeddings)
        pn_embeddings =self.postNorm(F.silu(self.projection(pn_embeddings)))
        if self.ablation == "sru":
            pn_embeddings = pn_embeddings.transpose(0, 1)
            outputs, auxs = self.MHBAMixers(pn_embeddings)
            outputs = outputs.transpose(0, 1)
        else:
            outputs, auxs = self.MHBAMixers(pn_embeddings)

        llm_outputs = self.llmhead(outputs)
        if self.task != "lm":
            llm_outputs = llm_outputs[:, -1, :]
            if self.ablation == "sru":
                return F.log_softmax(llm_outputs, dim=-1), None
            else:
                if self.use_aux_loss:
                    aux_llm_outputs = [self.llmhead(aux)[:, -1, :] for aux in auxs]
                    return F.log_softmax(llm_outputs, dim=-1), aux_llm_outputs
                else:
                    return F.log_softmax(llm_outputs, dim=-1), None
        else:
            if self.ablation == "sru":
                return F.log_softmax(llm_outputs, dim=-1), None
            else:
                if self.use_aux_loss:
                    aux_llm_outputs = [self.llmhead(aux) for aux in auxs]
                    return F.log_softmax(llm_outputs, dim=-1), aux_llm_outputs
                else:
                    return F.log_softmax(llm_outputs, dim=-1), None
        
    
    
    
    '''  detecting non grad parameters '''
    # def on_after_backward(self) -> None:
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(f"{name}")
    #        if param.requires_grad:
    #             tensorboard.add_histogram(f'{name}_grad', param.grad)
    
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        if "type_ids" in batch.keys():
            type_ids = batch["type_ids"]
        if "attention_mask" in batch.keys():
            attention_mask = batch["attention_mask"]
        if "label" in batch.keys():
            target = batch["label"].cuda()
        else:
            target = self.token_shift(input_ids).cuda()
        output, aux_llm_outputs = self.forward(input_ids, type_ids, attention_mask)
        loss = F.cross_entropy(output.view(-1, self.categories), target.view(-1), reduction='mean')
        if self.use_aux_loss:
            aux_loss = sum(F.cross_entropy(aux_llm_output.view(-1, self.categories), target.view(-1), reduction='mean') for aux_llm_output in aux_llm_outputs)
            loss = (loss+aux_loss)/(len(aux_llm_outputs)+1)
        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch), sync_dist=True)
        return loss
    
    # def on_train_batch_end(self, output, batch, batch_idx) -> None:
    #     tensorboard = self.logger.experiment
    #     for name, param in self.named_parameters():
    #        if param.requires_grad:
    #             tensorboard.add_histogram(f'{name}_grad', param.grad, self.global_step)
        

    def _shared_eval_step(self, batch, batch_idx, stage):
        input_ids = batch["input_ids"]
        if "type_ids" in batch.keys():
            type_ids = batch["type_ids"]
        if "attention_mask" in batch.keys():
            attention_mask = batch["attention_mask"]
        if "label" in batch.keys():
            target = batch["label"].cuda()
        else:
            target = self.token_shift(input_ids).cuda()
        assert (target >= 0).all() and (target < self.categories).all(), f"Found invalid target: {target}"
        output, aux_llm_outputs = self.forward(input_ids, type_ids, attention_mask)
        loss = F.cross_entropy(output.view(-1, self.categories), target.view(-1), reduction='mean')
        if self.use_aux_loss:
            aux_loss = sum(F.cross_entropy(aux_llm_output.view(-1, self.categories), target.view(-1), reduction='mean') for aux_llm_output in aux_llm_outputs)
            loss = (loss+aux_loss)/(len(aux_llm_outputs)+1)
        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch), sync_dist=True)
        # loss = L2Wrap.apply(loss, output)
        logits = torch.argmax(output, dim=-1)
        metrics = {stage+'_loss': loss, 
                   stage+'_accuracy': self.accuary(logits, target),
                   stage+'_f1score': self.f1score(logits, target),
                   stage+'_recall': self.recall(logits, target),
                   stage+'_precision': self.precision(logits, target)
                   }
        return loss, logits, metrics
    

    def validation_step(self, batch, batch_idx):
        loss, logits, metrics = self._shared_eval_step(batch, batch_idx, "val")
        if self.task == "lm":
            self.log("val_loss", metrics["val_loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch), sync_dist=True)
        else:
            self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch), sync_dist=True)
        return loss
        
    def on_validation_epoch_end(self) -> None:
        if self.task == "lm":
            pass
        else:
            metrics = {
                    'val_accuracy_epoch': self.accuary.compute(),
                    'val_f1score_epoch': self.f1score.compute(),
                    'val_recall_epoch': self.recall.compute(),
                    'val_precision_epoch': self.precision.compute()
                    }
            self.log_dict(metrics)
    
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), **self.optimizerconfig)


class CausalRCNDataModule(LightningDataModule):
    def __init__(self, 
                 config: PretrainedConfig
                 ) -> None:
        super().__init__()
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.task = config.task
    
    def setup(self, stage: str):
        dm = DatasetMap()
        if self.task == "lm":
            datasets = dm.create([
                                    "text8", 
                                    "bookcorpus", 
                                    "wiki103",
                                    # "arxiv",
                                    ], 
                                    mode="train")
            train_dataset, val_dataset = Data.random_split(datasets, 
                                                        [int(0.9*len(datasets)), len(datasets) - int(0.9*len(datasets))],
                                                        generator=torch.Generator().manual_seed(0))
        if stage == "fit":
            if self.task == "lm":
                '''
                Datasets for Modeling language model must be selected from 
                ["text8", "wiki103", "bookcorpus", "c4", "cc", "arxiv", "github", "se", "wiki"]
                '''
                self.train_dataset = train_dataset
                self.val_dataset = val_dataset

            else:
                '''
                Datasets for Modeling classfication model must be selected from 
                ["agnews, "amazon", "dbpedia", "hyper", "imdb", "yelp"]
                '''
                self.train_dataset = dm.create(self.task, mode="train")
                # self.val_dataset = dm.create(self.task, mode="validation")
                # self.val_dataset = dm.create(self.task, mode="dev")
                self.val_dataset = dm.create(self.task, mode="test")
    
    def train_dataloader(self) -> torch.Any:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)



