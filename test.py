import sys
sys.path.append('./')
import torch
import torch.nn as nn
import torch.nn.functional as F
from CausalRCN.Mixers import CausalRCN, Causal_CNN
from tokenizers import Tokenizer
from transformers.configuration_utils import PretrainedConfig
from CausalRCN.config_CausalRCN import CausalRCNConfig
from CausalRCN.hashembedding import HashEmbedding
from torchmetrics import Accuracy, F1Score, Recall, Precision
from torch.utils.data import DataLoader
from dataset_map import DatasetMap
from sru import SRU
from thop import profile
import time
torch.manual_seed(1)

@staticmethod
def beam_search(inputs: torch.Tensor, top_k: int, top_p: float, temperature: float):
    inputs = F.softmax(inputs.view(-1), dim=-1)
    inputs_values, inputs_index = torch.topk(inputs, top_k)
    inputs_values = F.softmax(inputs_values, dim=-1).cumsum(-1)
    index = torch.nonzero(inputs_values > top_p, as_tuple=False)[0]
    top_p_values = inputs_values[: index]
    top_p_index = inputs_index[:index]
    temperature_numerator = torch.exp(top_p_values/temperature)
    temperature_denominator = temperature_numerator.sum(-1)
    temperature_norm_res = temperature_numerator/temperature_denominator
    index = torch.multinomial(temperature_norm_res, 1, replacement=False)
    sample_res_index = top_p_index[index]
    return sample_res_index


class CausalRCNForGeneration(nn.Module):
    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        super().__init__()
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
        self.batch_size = config.batch_size
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
        self.use_aux_loss = config.use_aux_loss

        self.accuary = Accuracy(task="multiclass", num_classes=config.categories)
        self.recall = Recall(task="multiclass", average='macro', num_classes=config.categories)
        self.f1score = F1Score(task="multiclass", average='macro', num_classes=config.categories)
        self.precision = Precision(task="multiclass", average='macro', num_classes=config.categories)

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
    
    
    @torch.inference_mode()
    def predict_step(self, input_ids, max_length, top_k, top_p, temperature):
        for i in range(max_length):
            output, aux_llm_outputs = self.forward(input_ids)
            output = output[:,-1]
            next_token = beam_search(output, top_k, top_p, temperature)
            if next_token.item == self.eos_token_id:
                print(next_token, self.eos_token_id)
                break
            else:
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        return input_ids.squeeze(0)
    
    def test(self, ckpt, mode):
        checkpoint = torch.load(ckpt, map_location=torch.device('cuda:5'))
        model_weights = checkpoint["state_dict"]
        self.load_state_dict(model_weights)
        dm = DatasetMap()
        test_dataset = dm.create(self.task, mode=mode)
        tl = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=4)
        avg_loss = 0
        for step, batch in enumerate(tl):
            input_ids = batch["input_ids"].to('cuda:5')
            if "type_ids" in batch.keys():
                type_ids = batch["type_ids"].to('cuda:5')
            if "attention_mask" in batch.keys():
                attention_mask = batch["attention_mask"].to('cuda:5')
            if "label" in batch.keys():
                target = batch["label"].to('cuda:5')
            else:
                target = self.token_shift(input_ids).to('cuda:5')
            if step == 1:
                torch.cuda.reset_peak_memory_stats()  # 重置显存统计
                initial_memory = torch.cuda.memory_allocated()
                start_time = time.time()
            output, aux_llm_outputs = self.forward(input_ids, type_ids, attention_mask)
            if step == 1:
                memory_allocated = torch.cuda.memory_allocated() - initial_memory
                peak_memory_allocated = torch.cuda.max_memory_allocated()
                total_time = time.time() - start_time
                Flops, params = profile(self, inputs=(input_ids, type_ids, attention_mask))
            loss = F.cross_entropy(output.view(-1, self.categories), target.view(-1), reduction='mean')
            if self.use_aux_loss:
                aux_loss = sum(F.cross_entropy(aux_llm_output.view(-1, self.categories), target.view(-1), reduction='mean') for aux_llm_output in aux_llm_outputs)
                loss = (loss+aux_loss)/(len(aux_llm_outputs)+1)
            # loss = L2Wrap.apply(loss, output)
            logits = torch.argmax(output, dim=-1)
            avg_loss = avg_loss + loss.item()
            self.accuary(logits, target)
            self.f1score(logits, target)
            self.recall(logits, target)
            self.precision(logits, target)
            # print(torch.where(logits == target, 1, 0).sum()/target.shape[0])
        metrics = {
                    'test_loss': '%.2f' % (avg_loss/len(tl)),
                    'test_accuracy': '%.2f' % (100 * self.accuary.compute().item()),
                    'test_f1score': '%.2f' % (100 * self.f1score.compute().item()),
                    'test_recall': '%.2f' % (100 * self.recall.compute().item()),
                    'test_precision': '%.2f' % (100 * self.precision.compute().item()),
                    'model_memory_step': '%.2f MB' % (memory_allocated / (1024 ** 2)),
                    'max_memory': '%.2f MB' % (peak_memory_allocated / (1024 ** 2)),
                    'thoughput': '%.2f Samples/s' % (logits.shape[0]/total_time),
                    'FLOPs': "% .4fG" % (Flops / 1000000000),
                    'Params': "% .4f M" % (params / 1000000)
                    }
        return metrics
    
    

def predict(model, tokenizer, ckpt, inputs, max_length, top_k, top_p, temperature):
    checkpoint = torch.load(ckpt, map_location=torch.device('cuda:5'))
    model_weights = checkpoint["state_dict"]
    model.load_state_dict(model_weights)
    model.eval()
    input_ids = torch.LongTensor(tokenizer.encode(inputs).ids).unsqueeze(0)
    outputs = model.predict_step(input_ids.to('cuda:5'), max_length, top_k, top_p, temperature).cpu().numpy()
    outputs_sentence = tokenizer.decode(outputs)
    return outputs_sentence


if __name__ == "__main__":
    M2C = CausalRCNConfig()
    M2G = CausalRCNForGeneration(M2C)
    M2G = M2G.to('cuda:5')
    M2G.eval()
    ckpt = "lightning_logs/version_10/checkpoints/mixer-best-epoch=0216-val_loss=0.3127-val_f1score=0.5379-val_accuracy=0.8822-val_recall=0.5116.ckpt"
    mode = "test"
    # mode = "validation"
    metrics = M2G.test(ckpt, mode)
    print(metrics)
