from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset
from tokenizers.pre_tokenizers import Metaspace
from typing import Dict, List, Optional
from torch.utils.data import ConcatDataset
from tokenizers.processors import TemplateProcessing


class MHBAMixerTokenizer:
    def __init__(self, 
                    special_tokens: List[str], 
                    min_frequency: int, 
                    show_progress: bool, 
                    vocab_size: int,
                    max_length: int,
                    strategy: str,
                    direction: str,
                    replacement: str,
                    *args, **kwargs):
        self.tokenizer = Tokenizer(BPE(unk_token="<|unkown|>"))
        self.trainer = BpeTrainer(special_tokens=special_tokens,
                                    min_frequency=min_frequency,
                                    show_progress=show_progress,
                                    vocab_size=vocab_size,
                                )
        self.tokenizer.pre_tokenizer = Metaspace(replacement = replacement)
        self.tokenizer.enable_padding(pad_id=0, length=max_length, pad_token=special_tokens[0])
        self.tokenizer.enable_truncation(max_length=max_length, strategy=strategy, direction=direction)
        self.tokenizer.post_processor = TemplateProcessing(
                                                single=f"{special_tokens[1]}:0 $A:0 {special_tokens[2]}:0",
                                                pair=f"{special_tokens[1]}:0 $A:0 {special_tokens[3]}:0 $B:1 {special_tokens[2]}:1",
                                                special_tokens=[
                                                    (special_tokens[1], 1),
                                                    (special_tokens[2], 2),
                                                    (special_tokens[3], 3),
                                                ],
                                            )

    def build(self, corpus_info: List[Dict], save: bool, path_to_save: Optional[str]) -> None:
        if save:
            assert (path_to_save is not None), f"Please check the args, your save is {save}, but path_to_save is {path_to_save}"
        corpus = []
        for dataset in corpus_info:
            # 需要加入train/test/val的名字尽兴循环
            for split in dataset["split"]:
                
                current_dataset = load_dataset(dataset["path"], dataset["name"], split=split)[dataset["feature"]]
                corpus.append(current_dataset)
        corpus = ConcatDataset(corpus)
        self.tokenizer.train_from_iterator(iterator=corpus, trainer=self.trainer)
        if save:
            self.tokenizer.save(path_to_save)
        
    def gettokenizer(self):
        return self.tokenizer


if __name__ == "__main__":
    special_tokens=[
        "<|pad|>",
        "<|startoftext|>",
        "<|endoftext|>",
        "<|separation|>",
        "<|unkown|>",
        "<|im_start|>",
        "<|im_end|>",
        "<repo_name>",
        "<reponame>",
        "<file_sep>",
        "<filename>",
        "<gh_stars>",
        "<issue_start>",
        "<issue_comment>",
        "<issue_closed>",
        "<jupyter_start>",
        "<jupyter_text>",
        "<jupyter_code>",
        "<jupyter_output>",
        "<jupyter_script>",
        "<empty_output>"
    ]
    mmtokenizer = MHBAMixerTokenizer(special_tokens=special_tokens,
                                    min_frequency=1, 
                                    show_progress=True, 
                                    vocab_size=4096,
                                    replacement="_",
                                    max_length=512,
                                    strategy="longest_first",
                                    direction="right"
                                    )
    corpus_info = [{
        "path": "wikitext", 
        "name": "wikitext-2-raw-v1",  
        "feature": "text",
        "split": ["train", "test", "validation"]
    }]
    # corpus_info: List[Dict], save: bool, path_to_save: Optional[str]
    mmtokenizer.build(corpus_info, save=True, path_to_save="wikitext.json")
    test_meg = "hello, world!"
    test_tokenizer = Tokenizer.from_file("wikitext.json")
    result = test_tokenizer.encode(test_meg)
    print(f"tokens: {result.tokens} \n ids: {result.ids} \n attention_mask: {result.attention_mask}")

