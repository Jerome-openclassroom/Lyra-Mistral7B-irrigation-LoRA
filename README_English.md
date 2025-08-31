<img src="https://upload.wikimedia.org/wikipedia/en/c/c3/Flag_of_France.svg" width="100px" height="auto" />

# 🌱 Lyra-Mistral7B-irrigation-LoRA

![SouverainAI](https://img.shields.io/badge/🇫🇷%20SouverainAI-oui-success)
![EUstack](https://img.shields.io/badge/🇪🇺%20EUstack-ready-blue)

## 📌 Description
This repository documents the methodology used to perform a **LoRA fine-tuning** on the **Mistral-7B-Instruct-v0.3** model, specialized for agricultural irrigation (short answers in French giving water input recommendations in mm according to soil, water tension, and phenological stage).

The model is publicly available on Hugging Face:  
👉 [Lyra-Mistral7B-irrigation-LoRA](https://huggingface.co/jeromex1/Lyra-Mistral7B-irrigation-LoRA)

---

## ⚙️ Methodology

### Base model
- **Name**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Parameters**: 7.27 billion
- **Language**: multilingual, excellent in French

### Fine-tuning
- **Technique**: QLoRA (4-bit quantization, bitsandbytes)
- **LoRA target modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `down_proj`
- **Hyperparameters**:
  - `r=16`
  - `lora_alpha=32`
  - `lora_dropout=0.05`
  - `batch_size=2`
  - `gradient_accumulation_steps=4`
  - `seq_length=1024`
  - `epochs=3`
- **GPU used**: A100 (40 GB)

### Trainable parameters
```
trainable params: 23,068,672
all params: 7,271,092,224
trainable%: 0.3173
```

---

## 📉 Training results

### Loss by steps
```
Step   Loss
10     3.3845
20     2.2109
30     1.2643
40     0.7636
50     0.5695
60     0.4991
70     0.3880
80     0.3390
90     0.3037
100    0.2860
110    0.2766
Final avg loss (epoch 3): 0.9117
```

---

## 🔎 Prompt tests

### Prompt 1
```
Context: agriculture. Phenological stage: Flowering. Soil: clay. Soil water tension: 48 cbar.
Question: what irrigation water input is required? Provide a short answer.
```
- **Base Mistral-7B**: vague answer, unrealistic values (1000 m³/ha).  
- **LoRA**: `20 mm drip irrigation`.

---

### Prompt 2
```
Phenological stage: Growth. Soil: loam. Soil water tension: 32 cbar. Recent rainfall: 5 mm.
```
- **Base**: generic answer (10 mm).  
- **LoRA**: `20 mm irrigation recommended to maintain tension at 30 cbar`.

---

### Prompt 3
```
Phenological stage: Fruiting. Soil: sandy loam. Soil water tension: 67 cbar. Rain forecast: 22 mm.
```
- **Base**: "no need to irrigate".  
- **LoRA**: `10 mm irrigation recommended to maintain tension around 60 cbar`.

---

### Prompt 4
```
Phenological stage: Maturation. Soil: sand. Soil water tension: 72 cbar. Rain forecast: 4 mm.
```
- **Base**: off-topic answer.  
- **LoRA**: `2 mm is enough to maintain tension around 72 cbar`.

---

### Prompt 5
```
Phenological stage: Growth. Soil: sandy clay loam. Soil water tension: 55 cbar.
```
- **Base**: incoherent answer (27 mm/day).  
- **LoRA**: `20 mm drip irrigation`.

---

## 📦 Usage

Example of loading the model with PEFT:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = "mistralai/Mistral-7B-Instruct-v0.3"
lora_model = "jeromex1/Lyra-Mistral7B-irrigation-LoRA"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, load_in_4bit=True, device_map="auto")
model = PeftModel.from_pretrained(model, lora_model)

prompt = "Context: agriculture. Soil: sandy, tension 70 cbar, stage: Growth. What irrigation is required?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0], skip_special_tokens=True))
```
---

## 💾 Folder Structure

```
lyra_transformer/
├── README.md                          # French version
├── README_fr.md                       # English version
│
├── datasets/                          # my datasets in JSONL format, structured for training Mistral AI models
│   ├── lyra_irrigation_train_mistral.jsonl
│   └── lyra_irrigation_valid_mistral.jsonl
│
└── learning_curve/                    # training curve
    └── loss_LoRA_Mistral_7B.xlsx

```
---

## 📜 License
MIT
