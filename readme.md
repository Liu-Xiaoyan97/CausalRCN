# Causal RCN SLM
## Causal RCN 
![Causal RCN](/imgs/architecture.png)
## Performance
![performance](/imgs/results.png)
## News
- [2024-11-20] Release source code.

## ToDo
- [x] Release source code.
- [] Train SLM.
- [] SFT.

## Start train
```shell
python train.py --ckpt[option] CHECKPOINT_PATH
```
For *LM* task, modify the *task* field to "lm". For *classification* task, modify the *task* field to [dataset].

## Test
```shell
python test.py
```

## Inference with LM
```shell
python app_inference.py
```

## Citation
```shell

```
## Other projects
[MHBAMixer](https://github.com/Liu-Xiaoyan97/MHBA-Mixer)
```latex
@article{TANG2023119076,
title = {Pay attention to the hidden semanteme},
journal = {Information Sciences},
volume = {640},
pages = {119076},
year = {2023},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2023.119076},
url = {https://www.sciencedirect.com/science/article/pii/S0020025523006618},
author = {Huanling Tang and Xiaoyan Liu and Yulin Wang and Quansheng Dou and Mingyu Lu},
```
[TCAMixer](https://github.com/Liu-Xiaoyan97/TCAMixer)
```latex
@Article{Liu2023,
  author    = {Xiaoyan Liu and Huanling Tang and Jie Zhao and Quansheng Dou and Mingyu Lu},
  journal   = {Eng. Appl. Artif. Intell.},
  title     = {TCAMixer: {A} lightweight Mixer based on a novel triple concepts attention mechanism for {NLP}},
  year      = {2023},
  number    = {Part {C}},
  pages     = {106471},
  volume    = {123},
  bibsource = {dblp computer science bibliography, https://dblp.org},
  biburl    = {https://dblp.org/rec/journals/eaai/LiuTZDL23.bib},
  doi       = {10.1016/J.ENGAPPAI.2023.106471},
}
```
