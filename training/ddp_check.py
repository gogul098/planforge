"""Quick 2-GPU communication check (all_reduce + barrier)."""
import datetime
import os

import torch
import torch.distributed as dist

dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
r = dist.get_rank()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
t = torch.ones(1024, 1024, device="cuda") * (r + 1)
for _ in range(5):
    dist.all_reduce(t)
dist.barrier()
torch.cuda.synchronize()
if r == 0:
    print("DDP_OK", float(t[0, 0]), flush=True)
dist.destroy_process_group()
