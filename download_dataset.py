import os
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
print("start to load dataset")
# os.environ["RED_PAJAMA_DATA_DIR"] = "/share/home/liuxiaoyan/togethercomputer/RedPajama-Data-1T/"
# ds = load_dataset("/share/home/liuxiaoyan/togethercomputer/RedPajama-Data-1T/", 'github', split='train', trust_remote_code=True)
# ds = load_dataset("/share/home/liuxiaoyan/togethercomputer/RedPajama-Data-1T/","wikipedia", split='train', trust_remote_code=True)
ds = load_dataset("SetFit/CR")

print(ds)
dl = DataLoader(ds, batch_size=5)

for batch in dl:
    print(batch)