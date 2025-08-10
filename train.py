import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import argparse
import subprocess
from model import BERT, BERTLM
from scheduler import LinearWarmupLinearDecay
from dataset import WikipediaDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
import torch
from transformers import BertTokenizer
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import login

# torchrun --standalone --nproc-per-node=4 train.py

# Config argparser
parser = argparse.ArgumentParser(description="Pre-train BERT on Wikipedia with distributed processes")
parser.add_argument("--hf_token", type=str, help="Hugging Face token to access private dataset")
args = parser.parse_args()

# Initialize distributed processing
distributed = int(os.environ.get("RANK", -1)) != -1

if distributed:
    assert torch.cuda.is_available(), "CUDA required for distributed processing."
    init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_process = rank == 0
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
else:
    rank = 0
    local_rank = 0
    world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.set_float32_matmul_precision("high")

# Choose tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size

# Config model
bert = BERT(
    vocab_size=vocab_size,
    n_segments=2,
    max_len=512,
    attn_heads=12,
    embed_size=768,
    ff_size=3072,
    n_layers=12,
    dropout=0.1
)

model = BERTLM(bert=bert, vocab_size=vocab_size)
model.to(device)

# Display number of parameters
if master_process:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

# Wrap the model with DistributedDataParallel if distributed training is enabled
if distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
print(f"Initialized model on {device}")
time.sleep(0.5)

# Calculate gradient accumulation steps
batch_size = 32
max_len = 512
total_batch_size = 256 * 512
assert total_batch_size % (batch_size * max_len * world_size) == 0, "total_batch_size must be divisible by (batch_size * max_len * world_size)"
grad_accum_steps = total_batch_size // (batch_size * max_len * world_size)
if master_process:
    print("Gradient accumulation steps set to", grad_accum_steps)

# Load dev dataset
login(token=args.hf_token)
wikipedia = load_dataset("rplacucci/wiki-sentences", split="train")
time.sleep(0.5)

dataset = WikipediaDataset(
    dataset=wikipedia,
    tokenizer=tokenizer,
    max_len=max_len,
    world_size=world_size
)

sampler = DistributedSampler(
    dataset=dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=seed
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=0,
    prefetch_factor=None
)

dataiter = iter(dataloader)

# Define number of iterations to train
total_steps = 1000000
if master_process:
    print(f"Total training steps set to {total_steps:,}")

# Define optimizier and learning rate schedule
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.01
lr = 1e-4
warmup_steps = total_steps // 10

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=True)
scheduler = LinearWarmupLinearDecay(optimizer=optimizer, lr_max=lr, warmup_steps=warmup_steps, total_steps=total_steps)

# Define loss function
criterion = torch.nn.NLLLoss(ignore_index=-100)

# Config tensorboard and directories for logging/saving
out_dir = "./models/bert-110M-wikipedia-en-v2"
run_dir = "./runs/bert-110M-wikipedia-en-v2"
pfs_dir = "/lambda/nfs/lambda-fs/"

if master_process:
    writer = SummaryWriter(run_dir)
    os.makedirs(out_dir, exist_ok=True)
else:
    writer = None

if distributed:
    dist.barrier()

# Merge directories in the root disk to the persistent file system
if master_process:
    subprocess.run([
        "rsync", "-av", "--info=progress2", 
        "./models/", 
        os.path.join(pfs_dir, "models/")
    ], check=True)

    subprocess.run([
        "rsync", "-av", "--info=progress2", 
        "./runs/", 
        os.path.join(pfs_dir, "runs/")
    ], check=True)

if distributed:
    dist.barrier()

# Resume from checkpoint if available
checkpoint_path = os.path.join(out_dir, "ckpt.pt")  
start_step = 0  
loss = float("inf")  

if os.path.isfile(checkpoint_path):  
    if master_process:  
        print(f"Found checkpoint at {checkpoint_path}")  

    ckpt = torch.load(checkpoint_path, map_location=device)  

    if distributed:  
        model.module.load_state_dict(ckpt["model"])  
    else:  
        model.load_state_dict(ckpt["model"])  

    # load optimizer, scheduler & sampler
    optimizer.load_state_dict(ckpt["optimizer"])  
    scheduler.load_state_dict(ckpt["scheduler"])  
    sampler.load_state_dict(ckpt["sampler"])

    # restore bookkeeping
    start_step = ckpt["step"] + 1  
    loss = ckpt.get("loss", loss)  

    if master_process:  
        print(f"Resuming from step {start_step} with loss = {loss:.4f}")  

    # if using DDP, broadcast loaded weights to all ranks
    if distributed:  
        for param in model.parameters():  
            dist.broadcast(param.data, src=0)
        dist.barrier()

# Train
ckpt_steps = 1000
save_steps = 100000

for step in range(start_step, total_steps):
    
    # train model
    model.train()
    start = time.time()
    optimizer.zero_grad()
    
    loss_accum = 0.0
    for accum_step in range(grad_accum_steps):
        try:
            data = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            data = next(dataiter)

        data = {key: value.to(device) for key, value in data.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
            nsp_out, mlm_out = model(data["input_ids"], data["segment_ids"], data["attention_mask"])

        nsp_loss = criterion(nsp_out, data["nsp_label"])
        mlm_loss = criterion(
            mlm_out.view(-1, mlm_out.size(-1)),      # [batch_size * seq_len, vocab_size]
            data["mlm_labels"].view(-1)              # [batch_size * seq_len]
        )
        loss = nsp_loss + mlm_loss

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if distributed:
            model.require_backward_grad_sync = (accum_step == grad_accum_steps - 1)
        loss.backward()

    if distributed:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    torch.cuda.synchronize()
    elapsed = time.time() - start
    tokens_per_sec = batch_size * max_len * grad_accum_steps * world_size / elapsed if elapsed > 0 else 0

    loss = loss_accum.item()
    lr = scheduler.get_lr()[0]

    if master_process:
        print(f"(train) step: {step:6d}/{total_steps} | loss: {loss:.4f} | lr: {lr:.4e} | norm: {norm:.4f} | tok/sec: {tokens_per_sec:.0f}")
        writer.add_scalar("loss", loss, step)
        writer.add_scalar("lr", lr, step)
    
    # checkpoint model
    if step % ckpt_steps == 0 and step > 0:
        if master_process:
            checkpoint = {
                    "step": step,
                    "model": model.module.state_dict() if distributed else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "sampler": sampler.state_dict(),
                    "loss": loss,
                }
            print(f"Saving checkpoint to {out_dir}")
            torch.save(checkpoint, checkpoint_path)

            # merge
            subprocess.run([
                "rsync", "-av", "--info=progress2", 
                "./models/", 
                os.path.join(pfs_dir, "models/")
            ], check=True)

            subprocess.run([
                "rsync", "-av", "--info=progress2", 
                "./runs/", 
                os.path.join(pfs_dir, "runs/")
            ], check=True)

        if distributed:
            dist.barrier()

    # save model
    if step % save_steps == 0 and step > 0:
        if master_process:
            print(f"Saving model to {out_dir}")
            torch.save(model.module.state_dict() if distributed else model.state_dict(), os.path.join(out_dir, f"model_{step:06d}.pth"))
            
            # merge
            subprocess.run([
                "rsync", "-av", "--info=progress2", 
                "./models/", 
                os.path.join(pfs_dir, "models/")
            ], check=True)

            subprocess.run([
                "rsync", "-av", "--info=progress2", 
                "./runs/", 
                os.path.join(pfs_dir, "runs/")
            ], check=True)

        if distributed:
            dist.barrier()            

# Save final model
if master_process:
    print(f"Saving final model to {out_dir}")
    torch.save(model.module.state_dict() if distributed else model.state_dict(), os.path.join(out_dir, "model_final.pth"))

if distributed:
    dist.barrier()

# Cleanup distributed processing
if master_process:
    print("Cleaning up...")

if writer is not None:
    writer.flush()
    writer.close()

del model, optimizer, scheduler
del dataset, sampler, dataloader, dataiter
del tokenizer, writer

torch.cuda.empty_cache()

if distributed:
    torch.cuda.set_device(local_rank)
    dist.barrier(device_ids=[local_rank])
    destroy_process_group()

if master_process:
    print("Goodbye!")

time.sleep(1)