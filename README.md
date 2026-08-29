<div align="center">

# 🔐 PyTorch → SafeTensors Converter

**Convert `.bin` / `.pth` checkpoints to SafeTensors — single-file or sharded — with verification on every tensor**

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SafeTensors](https://img.shields.io/badge/🤗_SafeTensors-FFD21E?style=for-the-badge&logoColor=black)](https://github.com/huggingface/safetensors)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

</div>

---

## 🎯 Why convert at all?

PyTorch's native checkpoint format is a **pickle**. Loading one executes arbitrary Python — `torch.load()` on an untrusted checkpoint is remote code execution, not a file read. That's why HuggingFace built SafeTensors: a format that stores nothing but a JSON header and raw tensor bytes, with no code path to execute.

The benefits stack up:

| | PyTorch `.bin` / `.pth` | SafeTensors |
|:--|:--|:--|
| **Safety** | ⚠️ Arbitrary code execution on load | ✅ Data only — structurally inert |
| **Load speed** | Full deserialization | ✅ Zero-copy memory mapping |
| **Lazy loading** | Whole file into RAM | ✅ Read individual tensors without loading the rest |
| **Ecosystem** | Increasingly legacy | ✅ Default across HF Hub |

> 💡 **Interesting fact:** the converter's first real job is to go looking for tensors that **share the same memory**. Modern LLMs almost always tie their input embedding and output projection to the same weight matrix — `lm_head.weight` and `model.embed_tokens.weight` are frequently two names pointing at one tensor. Pickle stores that as a single object with two references. SafeTensors has no concept of aliasing, so a naive conversion silently *duplicates* the tensor — inflating a 7B checkpoint by a full embedding matrix (hundreds of MB) and, worse, decoupling two weights the model expects to stay tied. `shared_pointers()` detects the aliasing via `v.data_ptr()` and drops all but the first name before saving.

---

## 🔬 What it actually does

```
                    model directory
                          │
              ┌───────────┴───────────┐
              │  pytorch_model.bin    │
              │  .index.json present? │
              └───────────┬───────────┘
                   no ────┴──── yes
                    │            │
                    ▼            ▼
            convert_single   convert_multi
             model.pth        shard-by-shard,
                │             rewrite index
                └──────┬──────┘
                       ▼
        ┌──────────────────────────────────┐
        │  convert_file()                   │
        │                                   │
        │  1. torch.load(map_location=cpu)  │
        │  2. unwrap "state_dict" if nested │
        │  3. shared_pointers() → dedupe    │  ◄── aliasing guard
        │  4. .contiguous().half()          │  ◄── layout + dtype
        │  5. save_file(metadata={"pt"})    │
        │  6. size delta < 1%?              │  ◄── sanity check
        │  7. reload + torch.equal() all    │  ◄── proof of correctness
        └──────────────────────────────────┘
```

### The three correctness guards

**1 · Shared-tensor deduplication**
```python
ptrs = defaultdict(list)
for k, v in tensors.items():
    ptrs[v.data_ptr()].append(k)     # group names by memory address
```
Any address with more than one name is an alias group; all names after the first are dropped.

**2 · Contiguity, and a dtype caveat**
```python
loaded = {k: v.contiguous().half() for k, v in loaded.items()}
```
SafeTensors requires contiguous memory — a tensor produced by `.transpose()` or `.permute()` is a strided *view* over another tensor's buffer and cannot be serialized directly. `.contiguous()` forces a real copy in row-major order.

> ⚠️ **`.half()` is opinionated.** Every tensor is cast to **fp16**. That halves file size and matches how most inference stacks serve models — but it is *lossy* for an fp32 checkpoint and will **downcast bf16 to fp16**, which have different exponent ranges. If you're converting a checkpoint for continued training, or one trained in bf16, edit this line to preserve the source dtype.

**3 · Verification, not assumption**
```python
if (sf_size - pt_size) / pt_size > 0.01:      # size sanity
    raise RuntimeError(...)

reloaded = load_file(sf_filename)              # round-trip proof
for k in loaded:
    if not torch.equal(loaded[k], reloaded[k]):
        raise RuntimeError(f"The output tensors do not match for key {k}")
```
The file is **read back off disk and every tensor compared element-wise** against what was written. Conversion only "succeeds" once the round trip is proven exact — which is the right bar, since a subtly corrupted checkpoint is far more expensive to discover later than to catch here.

### Sharded checkpoints

For multi-file checkpoints, the tool reads `pytorch_model.bin.index.json`, converts each shard referenced in the `weight_map`, then writes a **new `model.safetensors.index.json`** with every path remapped (`pytorch_model-00001-of-00004.bin` → `model-00001-of-00004.safetensors`). The `rename()` helper handles both the extension swap and the `pytorch_model` → `model` prefix convention that HuggingFace uses.

---

## 🚀 Usage

```bash
pip install torch safetensors tqdm

python convert.py --model /path/to/model_dir
```

| Flag | Default | Purpose |
|:--|:--|:--|
| `-m`, `--model` | *(required)* | Path to the model directory |
| `-d`, `--delete` | `False` | Remove the original PyTorch files after conversion is **verified** |

Single-file vs. sharded is auto-detected — the script scans for `model.pth` and falls through to the sharded path otherwise. Deletion only happens after the round-trip check passes, so a failed conversion never destroys the source.

```bash
# convert and reclaim disk space once verified
python convert.py --model ./my-7b-model --delete True
```

---

## 📁 Files

| Path | Purpose |
|:--|:--|
| `convert.py` | The converter — CLI, sharding logic, verification |
| `model.pth` | Sample input checkpoint (~1.9 MB) |
| `model.safetensors` | Sample output (~967 KB — fp16 halving in action) |

---

## 📝 Attribution

The core single-tensor conversion routine — shared-pointer detection, size sanity check, round-trip verification — follows the approach used in HuggingFace's own SafeTensors conversion tooling. This project's contribution is the CLI wrapper around it: automatic single-vs-sharded detection, sharding-index rewriting, and the verified-then-delete cleanup flag.

A hosted Gradio version of this tool is deployed at [**Model_Converter_BIN-SafeTensors**](https://huggingface.co/spaces/Syed-Hasan-8503/Model_Converter_BIN-SafeTensors) on HuggingFace Spaces.
