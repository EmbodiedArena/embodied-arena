# Model installation and configuration guide

This document records installation dependencies, common issues, and recommended configuration for each model in the Embodied Arena evaluation framework. Different models may require different LLaVA versions or additional dependencies; some conflict with each other. Install on demand or use separate conda environments.

---

## Contents

- [1. General dependencies](#1-general-dependencies)
  - [flash-attention](#flash-attention)
  - [LLaVA-Next](#llava-next)
  - [Qwen2.5-VL](#qwen25-vl)
- [2. Model installation and configuration](#2-model-installation-and-configuration)
  - [EmbodiedBrain-7B](#embodiedbrain-7b)
  - [Qwen3-VL-8B-Instruct](#qwen3-vl-8b-instruct)
  - [VILA](#vila)
  - [RoboAnnotatorX](#roboannotatorx)
  - [robopoint](#robopoint)
  - [PhysVLM](#physvlm)
  - [EmbodiedGPT](#embodiedgpt)
  - [wall_oss](#wall_oss)
  - [mimo_embodied](#mimo_embodied)
  - [Wall-Brain](#wall-brain)

---

## 1. General dependencies

The following dependencies are shared by multiple models; install as needed.

### flash-attention

```
# Directly
pip install flash-attn --no-build-isolation

# [Option] install the per-commit wheel built by that PR, "https://github.com/Dao-AILab/flash-attention/releases"
pip install flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### LLaVA-Next

```
pip install -e git+https://github.com/LLaVA-VL/LLaVA-NeXT@b42941ceba259d5df18f8df8193a3897296a0449#egg=llava

# [Optional] If you want to use the latest version of LLaVA-Next, you can install it from the main branch.
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install -e . --no-deps # llava 1.7.0.dev0 
```

### Qwen2.5-VL

```
pip install qwen-vl-utils
```

---

## 2. Model installation and configuration

### EmbodiedBrain-7B

EmbodiedBrain-7B is a model fine-tuned on Qwen2.5-VL, specifically optimized for embodied intelligence tasks.

**Dependencies:**
```bash
# Same dependencies as Qwen2.5-VL
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

**Model download:**
Model weights should be placed under the `embodied_eval/data/EmbodiedBrain-7B/` directory.

**Usage:**
```bash
# Basic evaluation
bash example/vqa/embodied_brain.sh

# UniEQA evaluation
bash embodied_eval/tasks/unieqa/scripts/embodied_brain.sh
```

**Model characteristics:**
- Fine-tuned from Qwen2.5-VL-7B
- Uses Swift SFT full-parameter fine-tuning (`freeze_llm=False`)
- Freezes the vision encoder and aligner (`freeze_vit=True`, `freeze_aligner=True`)
- Supports multi-frame video input (recommended `max_num_frames=32`)
- Optimized for embodied intelligence–related tasks

### Qwen3-VL-8B-Instruct

```
pip install transformers==4.57.0
```

### VILA

need to install `llava` from VILA repo instead of LLaVA or LLaVA-Next.
```
pip uninstall llava
git clone https://github.com/NVlabs/VILA.git
cd VILA
pip install -e . --no-deps # vila-2.0.0
```
(1) cannot import name 'Qwen2FlashAttention2' from 'transformers.models.qwen2.modeling_qwen2'
```
pip install transformers==4.46.0
```
(2) Error during evaluation: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!
add `chat_template` to tokenizer_config.json of VILA1.5.
```json
"chat_template": "{% if messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{% for message in messages if message['content'] is not none %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
"chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + ' ' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
```

### RoboAnnotatorX

need to install `llava` from LLaVA repo.
```
pip uninstall llava
pip install --no-deps llava@git+https://github.com/haotian-liu/LLaVA.git@1619889c712e347be1cb4f78ec66e7cf414ac1a6 # llava-1.1.1
```
then you need to install `roboannotator` from the repo.
```
git clone https://github.com/LongXinKou/RoboannotatorX.git
cd RoboannotatorX
pip install -e . --no-deps # roboannotatorx-1.0
```
We recommend users to download the pretrained weights from the following link [EVA-ViT-G](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth)
and put them in `model_zoo`.

### robopoint

```
git clone https://github.com/wentaoyuan/RoboPoint.git
cd RoboPoint
pip install -e . --no-deps 
```

### PhysVLM

```
git clone https://github.com/unira-zwj/PhysVLM.git
cd PhysVLM/physvlm-main
pip install -e . --no-deps # physvlm-1.1.0 
```

### EmbodiedGPT

```
git clone https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch.git
cd EmbodiedGPT_Pytorch
pip install -e . --no-deps # robohusky-0.1.0
```
(1) ImportError: cannot import name 'is_flash_attn_available' from 'transformers.utils'
```python
from transformers.utils import is_flash_attn_2_available
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
```
(2) Error during evaluation: wo was specified in the `_keep_in_fp32_modules` list, but is not part of the modules in HuskyVisionModel.
modeling_husky_embody2.py(line 392): 
```python
# _keep_in_fp32_modules = ["wo"]
```
(3) Error during evaluation: 'HuskyQFormerFlashAttention2' object has no attribute 'embed_size'.
replace embed_size with embed_dim in modeling_husky_embody2.py line 838.
```python
context_layer = attn_output.reshape(bsz, tgt_len, embed_dim).contiguous()
```

### wall_oss

```
git clone https://github.com/X-Square-Robot/wall-x.git
pip install --no-build-isolation --verbose .
```

### mimo_embodied

(1) OSError: Can't load image processor for '/path/to/XiaomiMiMo/MiMo-Embodied-7B/'.
```
Download the file `preprocessor_config.json` from https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main
Put the file into /path/to/XiaomiMiMo/MiMo-Embodied-7B/
```

### Wall-Brain

Wall-Brain is an embodied VLM fine-tuned from Qwen3.5. It reuses the `qwen3_5` model class
in `embodied_eval/models/qwen3_5.py` for inference — set `--model qwen3_5` when running.

**Key dependencies:**

| Component     | Requirement           |
|---------------|-----------------------|
| PyTorch       | > 2.4                 |
| transformers  | ≥ 5.6.2 (recommended) |

Other dependencies follow the standard Qwen3.5 setup. See `embodied_eval/models/qwen3_5.py`
for the full list.
