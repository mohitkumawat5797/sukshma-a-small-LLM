# Sukshma — A Small LLM

**Sukshma** (Sanskrit for "subtle" / "small") is a from-scratch pretraining run of a small language model, built as a single, self-contained Google Colab notebook. It trains a randomly-initialized [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)-style architecture on streamed [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) text, with checkpointing to Google Drive so training can survive Colab disconnects.

> This is a learning project — "attempted to make a small LLM from not-so-scratch." Expect rough, early-stage generations rather than a polished chat model.

## What's in this repo

| File | Description |
|---|---|
| `sukshma.ipynb` | The entire project: model definition, data pipeline, training loop, eval, plotting, and generation — run top to bottom in Colab. |

There is currently no separate `model.py`/`train.py`/`requirements.txt` — everything lives in the notebook.

## Model architecture

The model is a `LlamaForCausalLM` from 🤗 Transformers, initialized with **random weights** (not loaded from a pretrained checkpoint) and configured to match SmolLM2-135M:

| Param | Value |
|---|---|
| Hidden size | 576 |
| Intermediate size | 1536 |
| Layers | 30 |
| Attention heads | 9 |
| KV heads (GQA) | 3 |
| Max sequence length | 1024 |
| Tied embeddings | Yes |
| Tokenizer | `HuggingFaceTB/SmolLM2-135M` |

## Training setup

- **Data:** `HuggingFaceFW/fineweb-edu` (`sample-10BT` config), streamed directly from the Hugging Face Hub and packed into fixed-length 1024-token blocks — no local dataset download required.
- **Optimizer:** AdamW (β = 0.9, 0.95), weight decay 0.1, cosine LR schedule with warmup.
- **Defaults:** micro-batch 4, gradient accumulation ×32 (effective batch 128), peak LR `6e-4`, 5000 steps, mixed precision (fp16 + `GradScaler`, tuned for a T4 GPU).
- **Checkpointing:** saves to `/content/drive/MyDrive/smol_llm/checkpoints` every 30 minutes (or every 200 steps, whichever comes first), keeping the last 3 checkpoints. Re-running the notebook automatically resumes from the latest checkpoint.
- **Eval:** a small fixed held-out set of batches (never trained on) is used to track eval loss/perplexity alongside training loss.

## How to run it

1. Open the notebook in Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohitkumawat5797/sukshma-a-small-LLM/blob/main/sukshma.ipynb)
2. Select a GPU runtime (Runtime → Change runtime type → T4/A100).
3. Run the first cells to install dependencies and mount your own Google Drive — checkpoints and the final model will be saved there under `smol_llm/`.
4. Run the remaining cells in order: model init → (optional) resume from checkpoint → build the streaming dataset → train → plot loss/perplexity → generate a sample → export the final model.

Because checkpoints are written to *your* Drive, there's no pretrained weights file bundled with or linked from this repo yet — each run trains its own model from step 0 (or resumes from whatever you've saved in your Drive).

## Notes / caveats

- This is a smoke-test-scale setup by default (5000 steps on a streamed dataset) — not enough tokens for coherent, general-purpose generations. Treat it as a base to scale up (more steps, bigger GPU, longer context) rather than a finished model.
- On an A100 you can likely switch from fp16/GradScaler to bf16 and drop the scaler for simpler, faster training (noted as a TODO in the notebook).
- If you do want to share pretrained weights, the notebook's final export step (`model.save_pretrained(...)`) writes a standard 🤗 Transformers checkpoint folder you can zip and host (e.g. on Google Drive or the Hugging Face Hub) — add a real link here once you have one.

## Training Status & Weights

Please note that the model training is currently highly under-done. The weights, including checkpoints and log files for the progress saved so far, can be accessed below:

* 📦 **[Model Checkpoints and Logs (Google Drive)](https://drive.google.com/drive/folders/1veKARZfGG5IwvllcQcDEd0odEQvmfpKv?usp=sharing)**

## License

No license file is currently included. Please add a license (e.g., [MIT](https://choosealicense.com/licenses/mit/) or [Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)) to the repository if you want others to reuse this code.
