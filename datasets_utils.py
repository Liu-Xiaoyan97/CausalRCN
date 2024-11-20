from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import os
import torch
from typing import Dict
import yaml

os.environ["RED_PAJAMA_DATA_DIR"] = "/share/home/liuxiaoyan/togethercomputer/RedPajama-Data-1T/"

class CausalRCNDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = None
        self.tokenizer = Tokenizer.from_file("wikitext.json")
        self.label_map = {
            "true": 1,
            "false": 0,
            "True": 1,
            "False": 0
        }

    def processing(self, field):
        field = self.tokenizer.encode(field['text'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "type_ids": type_ids
        }
    
    def __getitem__(self, index):
        results = self.processing(self.data[index])
        return results

    def __len__(self):
        return len(self.data)


class Text8(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/afmck/text8-chunked1024", split=mode)
    
    def processing(self, field):
        field = self.tokenizer.encode(field['text'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids)
        }
    
class Wikitext103(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/liuyanchen1015/VALUE_wikitext103_been_done", split=mode)

    def processing(self, field):
        field = self.tokenizer.encode(field['sentence'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids)
        }
    
class Bookcorpus(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/bookcorpus", split=mode)
    
    def processing(self, field):
        field = self.tokenizer.encode(field['text'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids)
        }

class C4(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/togethercomputer/RedPajama-Data-1T/", "c4", split=mode, trust_remote_code=True)


class Arxiv(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/togethercomputer/RedPajama-Data-1T/", 'arxiv', split=mode, trust_remote_code=True)

class CC(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/togethercomputer/RedPajama-Data-1T/", "common_crawl", split=mode, trust_remote_code=True)

class Github(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/togethercomputer/RedPajama-Data-1T/", "github", split=mode, trust_remote_code=True)

class Stackexchange(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/togethercomputer/RedPajama-Data-1T/","stackexchange", split=mode, trust_remote_code=True)

class Wikipedia(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/togethercomputer/RedPajama-Data-1T/", "wikipedia", split=mode, trust_remote_code=True)

class AGNews(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        # mode in ['train', 'test'], content = "text", label = "label", categories=4
        self.data = load_dataset("/share/home/liuxiaoyan/fancyzhx/ag_news", split=mode)
    
    def processing(self, field):
        label = field['label']
        field = self.tokenizer.encode(field['text'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }

class AmazonPolarity(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        # mode in ["train", "test"], content = "content", label = "label", categories=2
        self.data = load_dataset("/share/home/liuxiaoyan/fancyzhx/amazon_polarity", split=mode, trust_remote_code=True)

    def processing(self, field):
        label = field['label']
        field = self.tokenizer.encode(field['content'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }

class DBpedia(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        # mode in ["train", "test"], content = "content", label = "label", categories=14
        self.data = load_dataset("/share/home/liuxiaoyan/fancyzhx/dbpedia_14", split=mode, trust_remote_code=True)
    
    def processing(self, field):
        label = field['label']
        field = self.tokenizer.encode(field['content'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }

class YelpPolarity(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        # mode in ["train", "test"], content = "content", label = "label", categories=2 "1, 2"
        self.data = load_dataset("/share/home/liuxiaoyan/fancyzhx/yelp_polarity", split=mode, trust_remote_code=True)
    
    def processing(self, field):
        label = field['label']
        field = self.tokenizer.encode(field['text'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }

class Imdb(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        # mode in ["train", "test"], content = "text", label = "label", categories=2 "0, 1"
        self.data = load_dataset("/share/home/liuxiaoyan/stanfordnlp/imdb", split=mode, trust_remote_code=True)
    
    def processing(self, field):
        label = field['label']
        field = self.tokenizer.encode(field['text'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }

class MR(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        # mode in ["train", "test"], content = "text", label = "label", categories=2 "0, 1"
        self.data = load_dataset("/share/home/liuxiaoyan/cornell-movie-review-data/rotten_tomatoes", split=mode, trust_remote_code=True)
    
    def processing(self, field):
        label = field['label']
        field = self.tokenizer.encode(field['text'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }

class CR(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        # mode in ["train", "test"], content = "text", label = "label", categories=2 "0, 1"
        self.data = load_dataset("/share/home/liuxiaoyan/SetFit/CR", split=mode, trust_remote_code=True)
    
    def processing(self, field):
        label = field['label']
        field = self.tokenizer.encode(field['text'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }

class Hyperpartisan(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        # mode in ["train", "validation"], content = "text", label = "hyperpartisan", categories=2 "true, false"
        self.data = load_dataset("/share/home/liuxiaoyan/pietrolesci/hyperpartisan_news_detection", split=mode, trust_remote_code=True)

    def processing(self, field):
        label = field['hyperpartisan']
        field = self.tokenizer.encode(field['text'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label, dtype=torch.long)
        }

class Cola(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/nyu-mll/glue", "cola", split=mode)
    
    def processing(self, field):
        label = field["label"]
        field = self.tokenizer.encode(field['sentence'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }

class SST(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/nyu-mll/glue", "sst2", split=mode)
    
    def processing(self, field):
        label = field["label"]
        field = self.tokenizer.encode(field['sentence'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }
    
class MPQA(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/jxm/mpqa", split=mode)
    
    def processing(self, field):
        label = field["label"]
        field = self.tokenizer.encode(field['sentence'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }

class SUBJ(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/SetFit/subj", split=mode)
    
    def processing(self, field):
        label = field["label"]
        field = self.tokenizer.encode(field['text'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }

class MNLI(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/nyu-mll/glue", "mnli", split=mode)
    
    def processing(self, field):
        label = field["label"]
        field = self.tokenizer.encode(field['premise'], field['hypothesis'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }
    
class MRPC(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/nyu-mll/glue", "mrpc", split=mode)
    
    def processing(self, field):
        label = field["label"]
        field = self.tokenizer.encode(field['sentence1'], field['sentence2'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }

class RTE(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/nyu-mll/glue", "rte", split=mode)
    
    def processing(self, field):
        label = field["label"]
        field = self.tokenizer.encode(field['sentence1'], field['sentence2'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }

class QNLI(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/nyu-mll/glue", "qnli", split=mode)
    
    def processing(self, field):
        label = field["label"]
        field = self.tokenizer.encode(field['question'], field['sentence'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }

class WNLI(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/nyu-mll/glue", "wnli", split=mode)
    
    def processing(self, field):
        label = field["label"]
        field = self.tokenizer.encode(field['sentence1'], field['sentence2'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }

class QQP(CausalRCNDataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/nyu-mll/glue", "qqp", split=mode)
    
    def processing(self, field):
        label = field["label"]
        field = self.tokenizer.encode(field['question1'], field['question2'])
        input_ids = field.ids
        attention_mask = field.attention_mask
        type_ids = field.type_ids
        if isinstance(label, str):
            label = self.label_map[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "type_ids": torch.tensor(type_ids),
            "label": torch.tensor(label)
        }

__all__=[AGNews, 
         MR,
         AmazonPolarity, 
         Cola,
         MNLI,
         MRPC,
         QNLI,
         QQP,
         RTE,
         SST,
         WNLI,
         Arxiv, 
         Bookcorpus, 
         CC, 
         C4, 
         DBpedia, 
         Github, 
         Hyperpartisan, 
         Imdb, 
         Wikipedia, 
         Wikitext103, 
         Stackexchange,
         Text8, 
         YelpPolarity,
         MPQA,
         SUBJ,
         CR]


if __name__ == "__main__":
    agnews = AGNews("test")
    dl = DataLoader(agnews, 4)
    for step, batch in enumerate(dl):
        print(batch["input_ids"].shape)