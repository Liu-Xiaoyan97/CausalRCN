import sys
sys.path.append('./')
import torch
import torch.nn as nn
import torch.nn.functional as F
from MHBAMixerV2.Mixers import MHBAMixerV2
from tokenizers import Tokenizer
from transformers.configuration_utils import PretrainedConfig
from MHBAMixerV2.config_MHBAMixerV2 import MHBAMixerV2Config
from MHBAMixerV2.hashembedding import HashEmbedding



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


class MHBAMixerV2ForGeneration(nn.Module):
    def __init__(self, config: PretrainedConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.MHBAMixers = MHBAMixerV2(
            n_layers=config.n_layers,
            hidden_dim=config.embedding_dim,
            window_size=config.window_size,
            drop_rate=config.drop_rate,
            activation=config.activation,
            use_rnn=config.use_rnn,
            ffn_type = config.ffn_type
        )
        self.task = config.task
        self.finetune = config.finetune
        self.batch_size = config.batch_size
        self.eos_token_id = config.eos_token_id
        self.optimizerconfig = config.optimizer
        self.pad_token_id = config.pad_token_id
        self.token_shift = nn.ConstantPad1d((-1, 1), self.pad_token_id)
        if config.use_hashembedding:
            self.embeddings = HashEmbedding(config.vocab_size, config.embedding_dim, append_weight=False)
        else:
            self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.type_embedding = nn.Embedding(4, config.embedding_dim)
        self.postNorm = nn.LayerNorm(config.embedding_dim)
        self.drop_out = nn.Dropout(config.drop_rate)
        self.llmhead = nn.Linear(config.embedding_dim, config.categories, bias=False)
        self.categories = config.categories
        self.embedding_dim = config.embedding_dim

    def forward(self, input_ids: torch.tensor, type_ids: torch.tensor = None, attention_mask: torch.tensor = None):
        embeddings = self.embeddings(input_ids)
        if type_ids != None:
            type_embedding = self.type_embedding(type_ids)
            embeddings = embeddings + type_embedding
        if attention_mask != None:
            embeddings = embeddings * attention_mask[:, :, None].expand(embeddings.size())
        pn_embeddings = self.postNorm(F.silu(embeddings))
        outputs, auxs = self.MHBAMixers(pn_embeddings)
        llm_outputs = self.llmhead(outputs)
        if self.task != "lm":
            llm_outputs = llm_outputs[:, -1, :]
            aux_llm_outputs = [self.llmhead(aux)[:, -1, :] for aux in auxs]
        else:
            aux_llm_outputs = [self.llmhead(aux) for aux in auxs]
        return F.log_softmax(llm_outputs, dim=-1), aux_llm_outputs
    
    
    @torch.inference_mode()
    def predict_step(self, input_ids, max_length, top_k, top_p, temperature):
        for i in range(max_length):
            output, aux_llm_outputs = self.forward(input_ids)
            del aux_llm_outputs
            output = output[:,-1]
            next_token = beam_search(output, top_k, top_p, temperature)
            if next_token.item == self.eos_token_id:
                print(next_token, self.eos_token_id)
                break
            else:
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        return input_ids.squeeze(0)
    

M2C = MHBAMixerV2Config()
M2G = MHBAMixerV2ForGeneration(M2C)
M2G = M2G.to("cuda")

tokenizer = Tokenizer.from_file("wikitext.json")

def predict(ckpt, inputs, max_length, top_k, top_p, temperature):
    checkpoint = torch.load(ckpt, map_location=torch.device('cuda'))
    model_weights = checkpoint["state_dict"]
    M2G.load_state_dict(model_weights)
    M2G.eval()
    input_ids = torch.LongTensor(tokenizer.encode(inputs).ids).unsqueeze(0)
    outputs = M2G.predict_step(input_ids.to('cuda'), max_length, top_k, top_p, temperature).cpu().numpy()
    outputs_sentence = tokenizer.decode(outputs)
    return outputs_sentence



