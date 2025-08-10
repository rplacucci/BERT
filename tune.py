import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import time
import torch
import evaluate
import itertools
from transformers import BertTokenizer
from dataset import GLUEDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from model import BERT, BERTLM, BERT4GLUE
from scheduler import LinearLRwithWarmup
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# torchrun --standalone --nproc-per-node=4 tune.py

# Config argparser
parser = argparse.ArgumentParser(description="Fine-tune BERT on a GLUE task with distributed training")

parser.add_argument(
    "--task_name",
    type=str,
    default="qqp",
    help="Name of the GLUE task to run (mnli, qqp, qnli, sst2, cola, stsb, mrpc, rte, wnli)"
)

parser.add_argument(
    "--lr",
    type=float,
    default=2e-5,
    help="Constant learning rate for the optimizer"
)

parser.add_argument(
    "--n_epochs",
    type=int,
    default=3,
    help="Number of training epochs"
)

args = parser.parse_args()

# Set task name
task_name = args.task_name

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
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")

# Choose tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
if master_process:
    print("vocab size:", vocab_size)

# Load pretrained model
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

bertlm = BERTLM(bert, vocab_size)

model_dir = "models/bert-110M-wikipedia-en"
checkpoint_path = os.path.join(model_dir, "ckpt.pt") 

ckpt = torch.load(checkpoint_path, map_location=device)
bertlm.load_state_dict(ckpt["model"])
bert.load_state_dict(bertlm.bert.state_dict())

# Config model
model = BERT4GLUE(bert, task_name)
model.to(device)

# Display number of parameters
if master_process:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded BERT for {task_name} with")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

# Wrap the model with DistributedDataParallel if distributed training is enabled
if distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
print(f"Initialized model on {device}")
time.sleep(0.1)

# Config training hyperparameters
batch_size = 32
max_len = 512

# Load dev dataset and metric
glue = load_dataset("glue", task_name)
time.sleep(0.1)

primary_metric = (
    "spearmanr" if task_name == "stsb" else
    "f1" if task_name in("qqp", "mrpc") else
    "matthews_correlation" if task_name == "cola" else
    "accuracy"
) 

metric = evaluate.load(primary_metric)
time.sleep(0.1)

train_dataset = GLUEDataset(
    dataset=glue["train"],
    tokenizer=tokenizer,
    max_len=max_len,
    task_name=task_name
)

train_sampler = DistributedSampler(
    dataset=train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=0,
    prefetch_factor=None
)

valid_dataset = GLUEDataset(
    dataset=glue["validation_matched" if task_name == "mnli" else "validation"],
    tokenizer=tokenizer,
    max_len=max_len,
    task_name=task_name
)

valid_sampler = DistributedSampler(
    dataset=valid_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=False
)

valid_dataloader = DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    sampler=valid_sampler,
    num_workers=0,
    prefetch_factor=None
)

# Define optimizier and learning rate schedule
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.01
lr = args.lr

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

# Define loss function
criterion = (
    torch.nn.MSELoss() if task_name == "stsb" else 
    torch.nn.CrossEntropyLoss(torch.tensor([6.0, 1.0], device=device)) if task_name == "cola" else
    torch.nn.CrossEntropyLoss() 
)

# Config tensorboard and directory for logging/saving
writer = SummaryWriter(f"runs/bert-110M-wikipedia-en-glue-{task_name}")
out_dir = f"models/bert-110M-wikipedia-en-glue-{task_name}"
os.makedirs(out_dir, exist_ok=True)

# Train
n_epochs = args.n_epochs
best_loss = float("inf")
best_score = float("-inf")
step = 0

for epoch in range(n_epochs):
    # train
    if distributed:
        dist.barrier(device_ids=[local_rank])
    train_sampler.set_epoch(epoch)
    model.train()

    for batch in train_dataloader:
        start = time.time()
        data = {key: value.to(device) for key, value in batch.items()}

        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=False):
            logits = model(data["input_ids"], data["segment_ids"], data["attention_mask"])

        # logits = logits.float()
        if task_name == "stsb":
            logits = logits.squeeze(-1)
        
        loss = criterion(logits, data["label"])
        loss.backward()
        if distributed:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        elapsed = time.time() - start
        tokens_per_sec = batch_size * max_len * world_size / elapsed if elapsed > 0 else 0
        
        loss = loss.item()

        if master_process:
            print(f"(train) epoch: {epoch:2d} | step: {step:4d} | loss: {loss:.4f} | lr: {lr:.4e} | norm: {norm:.4f} | tok/sec: {tokens_per_sec:.0f}")
            writer.add_scalar("loss/train", loss, step)
            writer.add_scalar("lr", lr, step)
        
        step += 1

    # validate
    if distributed:
        dist.barrier(device_ids=[local_rank])

    model.eval()
    with torch.no_grad():
        local_preds, local_refs = [], []
        total_valid_loss = torch.tensor(0.0, device=device)
        n_valid_batches = torch.tensor(0, device=device)

        start = time.time()
        for batch in valid_dataloader:
            data = {key: value.to(device) for key, value in batch.items()}
            labels = data["label"]

            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=False):
                logits = model(data["input_ids"], data["segment_ids"], data["attention_mask"])

            # logits = logits.float()
            if task_name == "stsb":
                logits = logits.squeeze(-1)

            loss = criterion(logits, labels)
            preds = (
                logits.cpu().tolist() if task_name == "stsb" else
                torch.argmax(logits, dim=-1).cpu().tolist()
            )
            refs = labels.cpu().tolist()
            local_preds.extend(preds)
            local_refs.extend(refs)

            total_valid_loss += loss.detach()
            n_valid_batches += 1

        if distributed:
            dist.all_reduce(total_valid_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_valid_batches, op=dist.ReduceOp.SUM)
            all_preds = [None for _ in range(world_size)]
            all_refs = [None for _ in range(world_size)]
            dist.all_gather_object(all_preds, local_preds)
            dist.all_gather_object(all_refs, local_refs)
        else:
            all_preds = [local_preds]
            all_refs = [local_refs]

        total_valid_loss = total_valid_loss.item()
        n_valid_batches = n_valid_batches.item()

        avg_valid_loss = total_valid_loss / n_valid_batches if n_valid_batches > 0 else 0.0

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        tokens_per_sec = batch_size * max_len * world_size * n_valid_batches / elapsed if elapsed > 0 else 0

        if master_process:
            flat_preds = list(itertools.chain.from_iterable(all_preds))
            flat_refs = list(itertools.chain.from_iterable(all_refs))
            metric.add_batch(predictions=flat_preds, references=flat_refs)
            results = metric.compute()
            score = results.get(primary_metric, sum(results.values()) / len(results))

            print(f"(valid) epoch: {epoch:2d} | step: {step:4d} | loss: {avg_valid_loss:.4f} | {primary_metric}: {score:.4f} | tok/sec: {tokens_per_sec:.0f}")
            writer.add_scalar("loss/valid", avg_valid_loss, epoch)
            writer.add_scalar(f"metric/{primary_metric}", score, epoch)

# Save final model
if master_process:
    print(f"Saving final model to {out_dir}")
    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pth"))

# Cleanup distributed processing
if master_process:
    print("Cleaning up...")

if writer is not None:
    writer.flush()
    writer.close()

del model, optimizer
del train_dataset, valid_dataset
del tokenizer, writer

torch.cuda.empty_cache()

if distributed:
    torch.cuda.set_device(local_rank)
    dist.barrier(device_ids=[local_rank])
    destroy_process_group()

if master_process:
    print("Goodbye!")

time.sleep(0.1)