
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import yaml
import time
import pandas as pd
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from src.model import BERT, BERT4GLUE
from src.dataset import GLUEDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertTokenizer

# torchrun --standalone --nproc-per-node=4 test.py

# Config argparser
parser = argparse.ArgumentParser(description="Fine-tune BERT on a GLUE task with distributed training")

parser.add_argument(
    "--bert_config",
    type=str,
    default="tiny",
    help="Size of BERT model (tiny, mini, small, medium, base, or large)"
)

parser.add_argument(
    "--task_name",
    type=str,
    default="qqp",
    help="Name of the GLUE task to run (ax, mnli-m, mli-mm, mnli, qqp, qnli, sst2, cola, stsb, mrpc, rte, wnli)"
)

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

# Choose task
task_name = args.task_name

# Config tokenizer and hyperparameters
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
max_len = 512
batch_size = 32

# Load finetuned model
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

bert = BERT(vocab_size=vocab_size, **config[args.bert_config])

model = BERT4GLUE(
    bert=bert,
    task_name="mnli" if task_name in ("mnli-m", "mnli-mm") else task_name
)
model.to(device)

# Wrap the model with DistributedDataParallel if distributed training is enabled
if distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
print(f"Initialized model on {device}")
time.sleep(0.1)

# Load finetuned weights
model_dir = f"models/bert-{args.bert_config}-wikipedia-en-glue-{"mnli" if task_name in ("mnli-m", "mnli-mm", "ax") else task_name}"
state_dict = torch.load(os.path.join(model_dir, "model_final.pth"), map_location=device)
model.load_state_dict(state_dict)

# Display number of parameters
if master_process:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded BERT4GLUE from {model_dir} with {trainable_params:,} trainable parameters")

# Load dataset
glue = load_dataset(
    "glue", 
    "mnli" if task_name in ("mnli-m", "mnli-mm") else task_name
)

test_dataset = GLUEDataset(
    dataset=glue[
        "test_matched" if task_name == "mnli-m" else 
        "test_mismatched" if task_name == "mnli-mm" else 
        "test"
    ],
    tokenizer=tokenizer,
    max_len=max_len,
    task_name="mnli" if task_name in ("mnli-m", "mnli-mm") else task_name
)

test_sampler = DistributedSampler(
    dataset=test_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=False
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    sampler=test_sampler,
    num_workers=0,
    prefetch_factor=None
)

# Test model
model.eval()
with torch.no_grad():
    idxs, preds = [], []
    for step, batch in enumerate(test_dataloader):
        start = time.time()
        data = {key: value.to(device) for key, value in batch.items()}

        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=False):
            logits = model(data["input_ids"], data["segment_ids"], data["attention_mask"])

        if task_name == "stsb":
            logits = logits.squeeze(-1).float()

        idxs.extend(data["idx"].cpu().tolist())
        preds.extend(
            logits.cpu().tolist() if task_name == "stsb" else
            torch.argmax(logits, dim=-1).cpu().tolist()
        )

        torch.cuda.synchronize()
        elapsed = time.time() - start
        tokens_per_sec = batch_size * max_len * world_size / elapsed if elapsed > 0 else 0

        if master_process:
            print(f"(test) step: {step:4d}/{len(test_dataloader)-1:4d} | tok/sec: {tokens_per_sec:.0f}")
    
    if distributed:
        all_idxs = [None] * world_size
        all_preds = [None] * world_size
        dist.all_gather_object(all_idxs, idxs)
        dist.all_gather_object(all_preds, preds)
    else:
        all_idxs = [idxs]
        all_preds = [preds]

if master_process:
    # Flatten
    flat_idxs = [i for replica in all_idxs  for i in replica]
    flat_preds = [p for replica in all_preds for p in replica]

    # Format submission according to task
    if task_name in ("ax", "mnli-m", "mnli-mm"):
        flat_preds = [
            "entailment" if pred == 0 else
            "neutral" if pred == 1 else
            "contradiction"
            for pred in flat_preds
        ]

    if task_name in ("qnli", "rte"):
        flat_preds = [
            "entailment" if pred == 0 else
            "not_entailment"
            for pred in flat_preds
        ]

    if task_name == "stsb":
        flat_preds = [min(max(pred, 0), 5) for pred in flat_preds]
        flat_preds = [
            f"{pred:.3f}" for pred in flat_preds
        ]

    # Save results to file
    out_dir = f"submission-bert-{args.bert_config}"
    os.makedirs(out_dir, exist_ok=True)
    fname = {
        "cola": "CoLA.tsv",
        "mnli-m": "MNLI-m.tsv",
        "mnli-mm": "MNLI-mm.tsv",
        "mrpc": "MRPC.tsv",
        "qnli": "QNLI.tsv",
        "qqp": "QQP.tsv",
        "rte": "RTE.tsv",
        "sst2": "SST-2.tsv",
        "stsb": "STS-B.tsv",
        "wnli": "WNLI.tsv",
        "ax": "AX.tsv"
    }[task_name]

    out_path = os.path.join(out_dir, fname)

    df = pd.DataFrame({
        "index": flat_idxs,
        "prediction": flat_preds
    })

    df = df.sort_values("index").drop_duplicates(subset="index", keep="first").reset_index(drop=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(df)} rows to {out_path}")

# Cleanup distributed processing
if master_process:
    print("Cleaning up...")

torch.cuda.empty_cache()

if distributed:
    torch.cuda.set_device(local_rank)
    dist.barrier(device_ids=[local_rank])
    destroy_process_group()

if master_process:
    print("Goodbye!")

time.sleep(0.1)