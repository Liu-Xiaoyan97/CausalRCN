from datasets_utils import (AGNews, 
         AmazonPolarity, 
         MR,
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
         CR)

from typing import Union, List
from torch.utils.data import ConcatDataset

class DatasetMap:
    def __init__(self) -> None:
        self.dataset_map = {
             "agnews": AGNews,
             "amazon": AmazonPolarity,
             "arxiv": Arxiv,
             "bookcorpus": Bookcorpus,
             "c4": C4,
             "cc": CC,
             "dbpedia": DBpedia,
             "github": Github,
             "hyper": Hyperpartisan,
             "imdb": Imdb,
             "se": Stackexchange,
             "text8": Text8,
             "wiki": Wikipedia,
             "wiki103": Wikitext103,
             "yelp": YelpPolarity,
             "cola": Cola,
             "mnli": MNLI,
             "qnli": QNLI,
             "rte": RTE,
             "mrpc": MRPC,
             "qqp": QQP,
             "sst": SST,
             "wnli": WNLI,
             "mr": MR,
             "mpqa": MPQA,
             "subj": SUBJ,
             "cr": CR
        }


    def create(self, dataset_name: Union[List, str], mode: str):
        if isinstance(dataset_name, List):
            tmp_list = []
            for dn in dataset_name:
                tmp_list.append(self.dataset_map[dn](mode))
            return ConcatDataset(tmp_list)
        if isinstance(dataset_name, str):
            return self.dataset_map[dataset_name](mode)
        
# if __name__ == "__main__":
#     dm = DatasetMap()
#     ds = dm.create(dataset_name=["imdb", "agnews"], mode="test")
#     print(ds)