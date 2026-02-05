# Importing Hugging Face Models into NeMo Format

This guide describes how to import **any model** from Hugging Face into **NeMo format** using Slurm, the NeMo
Singularity container, and authenticated access with a Hugging Face token.

The workflow is designed to:

- Support **multiple model imports** in the same repository
- Keep caches and outputs isolated and reproducible
- Work automatically inside Slurm (no interactive login required)
- Use a consistent directory and script structure
- Allow adding per-model import scripts and documentation later

---

# 1. Overview (General)

Many large language models (LLMs) on Hugging Face (e.g., Llama, Mistral, Qwen, Falcon, Mixtral) can be imported into
NeMo format using the command:

```bash
nemo llm import
model=<nemo-model-type>
source="hf://<hf-org>/<hf-model-name>"
output_path=<output-directory>
--yes
```

This repository provides:

- A **general import workflow**
- A **reusable Slurm script template**
- A place to add **model-specific notes** and examples

---

# 2. Requirements (General)

## 2.1 NeMo Singularity Container

Ensure you have access to a NeMo container.

## 2.2 Hugging Face Account & Model Access

To download restricted/gated models, you must:

1. Log in to Hugging Face
2. Visit the model page
3. Accept terms & request access if required

## 2.3 Create a Hugging Face Access Token

1. Log in to Hugging Face
2. Go to: **Profile → Settings → Access Tokens**
3. Click **New Token**
4. Choose:
    - Name: `nemo-import`
    - Permission: **Read**
5. Copy the token (`hf_xxx...`)

## 2.4 **A100 GPU Node (required)**

NeMo model import **requires** an A100 GPU.  
Imports will **fail** on V100 nodes due to missing CUDA, BF16, and Tensor Core features needed by NeMo’s model
conversion pipeline.

### GPU Support Matrix

| GPU Type | Import Works? | Notes                                                          |
|----------|---------------|----------------------------------------------------------------|
| **A100** | ✅ Yes         | Fully supported. **Required for all imports.**                 |
| **H100** | ✅ Yes         | Supported, but not available on Ibex.                          |
| **V100** | ❌ No          | Import will fail — lacks needed CUDA/BF16/Tensor Core support. |

All import scripts in this repository include:

```bash
#SBATCH --constraint=a100,4gpus
```
---

# 3. Authentication Before Running the Job (General)

Set your token in the environment before submitting:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
```

---

# 4. General Slurm Import Script Template

This [template](template/template_impot_model.slurm) works for any model.
Just modify the `model=<NEMO_MODEL_NAME>` and `source="hf://<hf-org>/<hf-model-name>"` fields.

Replace the placeholders in the script as follows:

| Placeholder         | Description                                                               |
|---------------------|---------------------------------------------------------------------------|
| `<NEMO_MODEL_NAME>` | The NeMo model identifier (e.g., `llama31_8b`, `mistral_7b`, `qwen25_7b`) |
| `<hf-org>`          | The Hugging Face organization or user (e.g., `meta-llama`, `mistralai`)   |
| `<hf-model-name>`   | The model name on Hugging Face (e.g., `Llama-3.1-8B`)                     |

---

# 5. Verifying Success (General)

After job completion, run:

```commandline
tree model
```

Expected:

- model.nemo

- tokenizer files

- config files

- weight shards (if applicable)

Presence of model.nemo ⇒ import succeeded.

example:

```commandline
model/
├── context
│   ├── artifacts
│   │   └── generation_config.json
│   ├── io.json
│   ├── model.yaml
│   └── nemo_tokenizer
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── tokenizer.json
└── weights
    ├── __0_0.distcp
    ├── __0_1.distcp
    ├── common.pt
    └── metadata.json
```

# 6. Running the Import Jobs

Before running any import script, make sure you:

1. Have access to the model on Hugging Face
2. Have exported your Hugging Face token (`HF_TOKEN`)
3. Are requesting an **A100 GPU node** — *V100 nodes cannot be used for NeMo*
    - NeMo import requires newer GPU capabilities (tensor cores, BF16 support, and CUDA features not available on V100s)
    - Attempting to import on a V100 node will fail, even for smaller models

---

## 6.1 Importing Llama-3.1-8B

### Step 1 — Ensure you have access

Visit the model page:

https://huggingface.co/meta-llama/Llama-3.1-8B

Log in and:

- Accept the license terms
- Click **“Access repository”** if shown
- Wait for approval by mail (if it is gated)

### Step 2 — Export your Hugging Face token

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Step 3 — Submit the import job

```commandline
cd llama31_8b/
sbatch import_llama31_8b.slurm
```

### Step 4 — Check logs & output

Logs:

```commandline
llama31_8b/logs/l-imp-<JOBID>.out
```

Imported NeMo model:

```commandline
tree llama31_8b/model
```

## 6.2 Importing Mixtral-8x7B

### Step 1 — Ensure you have access

Visit the model page:

https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1

Log in and:

- Accept the license terms
- Click **“Access repository”** if shown
- Wait for approval by mail (if it is gated)

### Step 2 — Export your Hugging Face token

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Step 3 — Submit the import job

```commandline
cd mixtral_8x7b/
sbatch import_mixtral_8x7b.slurm
```

### Step 4 — Check logs & output

Logs:

```commandline
mixtral_8x7b/logs/m-imp-<JOBID>.out
```

Imported NeMo model:

```commandline
tree mixtral_8x7b/model
```

