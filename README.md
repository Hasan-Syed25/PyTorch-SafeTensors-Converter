# PyTorch → SafeTensors Converter

A CLI utility for converting PyTorch model checkpoints (`.bin`/`.pth`) into the [SafeTensors](https://github.com/huggingface/safetensors) format — handling both single-file checkpoints and sharded multi-file checkpoints (the `pytorch_model.bin.index.json` case common with larger HuggingFace models) in one command.

## Why

SafeTensors is safer (no arbitrary code execution on load, unlike pickle-based `.bin`/`.pth`) and faster to load than PyTorch's native checkpoint format, so it's the preferred format for sharing and serving models. Converting a checkpoint correctly means more than just re-saving it: tensors that share memory (`shared_pointers`) need to be de-duplicated before saving, half-precision tensors need to be made contiguous, and the conversion needs to be verified — this tool does all three and re-derives the sharding index for multi-file checkpoints.

## What it does

- **Single-file checkpoints:** converts `model.pth` → `model.safetensors` directly.
- **Sharded checkpoints:** reads `pytorch_model.bin.index.json`, converts every shard referenced in the weight map, and writes a new `model.safetensors.index.json` with the weight map remapped to the converted filenames.
- **Correctness checks on every conversion:** detects and deduplicates tensors that share underlying memory, verifies the converted file's size is within 1% of the original, and reloads the SafeTensors file to confirm every tensor round-trips byte-for-byte identical to the source.
- **Optional cleanup:** pass `--delete` to remove the original PyTorch files once conversion is verified.

## Usage

```bash
pip install torch safetensors tqdm

python convert.py --model /path/to/model_dir [--delete True]
```

The script auto-detects whether the directory holds a single `model.pth` or a sharded checkpoint (via the presence of `pytorch_model.bin.index.json`) and converts accordingly.

## Note

The core single-tensor conversion routine (shared-pointer detection, size/round-trip verification) follows the approach used in HuggingFace's own `safetensors` conversion tooling; the CLI wrapper — auto-detecting single vs. sharded checkpoints, rewriting the sharding index, and the cleanup flag — is this project's contribution.
