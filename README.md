<img src="https://upload.wikimedia.org/wikipedia/en/c/c3/Flag_of_France.svg" width="100px" height="auto" />


# 🌱 Lyra-Mistral7B-irrigation-LoRA

![EUstack](https://img.shields.io/badge/🇪🇺%20EUstack-ready-blue)
![SouverainAI](https://img.shields.io/badge/🇫🇷%20SouverainAI-oui-success)

## 📌 Description
Ce dépôt décrit la méthodologie employée pour réaliser un **fine-tuning LoRA** sur le modèle **Mistral-7B-Instruct-v0.3**, spécialisé pour l'irrigation agricole (réponses courtes en français donnant des apports d'eau en mm selon le sol, la tension hydrique et le stade phénologique).

Le modèle est disponible publiquement sur Hugging Face :  
👉 [Lyra-Mistral7B-irrigation-LoRA](https://huggingface.co/jeromex1/Lyra-Mistral7B-irrigation-LoRA)

---

## ⚙️ Méthodologie

### Base model
- **Nom** : `mistralai/Mistral-7B-Instruct-v0.3`
- **Paramètres** : 7,27 milliards
- **Langue** : multilingue, excellent en français

### Fine-tuning
- **Technique** : QLoRA (quantization 4 bits, bitsandbytes) dans Google Colab Pro avec processeur A100
- **Modules LoRA** : `q_proj`, `k_proj`, `v_proj`, `o_proj`, `down_proj`
- **Hyperparamètres** :
  - `r=16`
  - `lora_alpha=32`
  - `lora_dropout=0.05`
  - `batch_size=2`
  - `gradient_accumulation_steps=4`
  - `seq_length=1024`
  - `epochs=3`
- **GPU utilisé** : A100 (40 Go)

### Paramètres entraînables
```
trainable params: 23,068,672
all params: 7,271,092,224
trainable%: 0.3173
```

---

## 📉 Résultats d'entraînement

### Loss par steps
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

## 🔎 Tests de prompts

### Prompt 1
```
contexte : agriculture. stade phénologique Floraison, sol argileux, tension de l'eau du sol 48 cbar,
question portant sur l'irrigation : quel apport d'eau est nécessaire ? donner une réponse courte
```
- **Base Mistral-7B** : réponse vague, valeurs aberrantes (1000 m³/ha).
- **LoRA** : `20 mm en goutte à goutte`.

---

### Prompt 2
```
stade phénologique Croissance, sol limoneux, tension hydrique 32 cbar, 5 mm pluie récente
```
- **Base** : réponse générique (10 mm).
- **LoRA** : `20 mm d'irrigation recommandée pour maintenir la tension à 30 cbar`.

---

### Prompt 3
```
stade phénologique Fructification, sol limono-sableux, tension 67 cbar, 22 mm de pluie prévue
```
- **Base** : "pas nécessaire d'irriguer".
- **LoRA** : `10 mm d'irrigation recommandée pour maintenir la tension autour de 60 cbar`.

---

### Prompt 4
```
stade phénologique Maturation, sol sableux, tension hydrique 72 cbar, 4 mm de pluie prévue
```
- **Base** : réponse hors sujet.
- **LoRA** : `2 mm suffisent pour maintenir la tension autour de 72 cbar`.

---

### Prompt 5
```
stade phénologique Croissance, sol argilo-sableux, tension hydrique 55 cbar
```
- **Base** : réponse incohérente (27 mm/jour).
- **LoRA** : `20 mm en goutte à goutte`.

---

## 📦 Utilisation

Exemple de chargement du modèle avec PEFT :
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = "mistralai/Mistral-7B-Instruct-v0.3"
lora_model = "jeromex1/Lyra-Mistral7B-irrigation-LoRA"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, load_in_4bit=True, device_map="auto")
model = PeftModel.from_pretrained(model, lora_model)

prompt = "contexte : agriculture. sol sableux, tension 70 cbar, stade Croissance, quel apport d'eau ?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0], skip_special_tokens=True))
```

---

## 📜 Licence
MIT
