# Mental-Support-Chat-DeepSeek-R1-Distill-Llama-8B

This repository contains a PEFT adapter fine-tuned for supportive mental health conversations, built on top of [`unsloth/DeepSeek-R1-Distill-Llama-8B`](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B). The model is optimized for empathetic, evidence-based, and emotionally aware responses in text-based interactions.

---

## 🤗 Hugging Face Model

**🔗 Model Link**: [arafatanam/Mental-Support-Chat-DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/arafatanam/Mental-Support-Chat-DeepSeek-R1-Distill-Llama-8B)

---

## 📦 Files Included

| File Name                   | Description                          |
| --------------------------- | ------------------------------------ |
| `.gitattributes`            | Git LFS file tracking config         |
| `adapter_config.json`       | PEFT adapter configuration           |
| `adapter_model.safetensors` | Adapter weights (safe tensor format) |
| `special_tokens_map.json`   | Special token mappings               |
| `tokenizer.json`            | Tokenizer vocabulary and settings    |
| `tokenizer_config.json`     | Tokenizer configuration              |

---

## 🧠 Model Highlights

- **Base Model:** `unsloth/DeepSeek-R1-Distill-Llama-8B`
- **Adapter Format:** PEFT (Parameter-Efficient Fine-Tuning)
- **Training Dataset:** [`Amod/mental_health_counseling_conversations`](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)
- **Purpose:** Supportive, empathetic AI assistant for mental wellness
- **License:** Apache 2.0

---

## 🚀 How to Use

```python
!pip install -U torch transformers peft bitsandbytes

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

base_model = "deepseek-ai/deepseek-llm-8b-r1"
adapter = "arafatanam/Mental-Support-Chat-DeepSeek-R1-Distill-Llama-8B"

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    add_bos_token=True,
    trust_remote_code=True,
    padding_side='left'
)

config = PeftConfig.from_pretrained(adapter)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_4bit=True,
    device_map='auto',
    torch_dtype='auto'
)
model = PeftModel.from_pretrained(model, adapter)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Example prompt
user_message = [{"role": "user", "content":
    "I'm feeling really overwhelmed lately. I don’t know how to handle everything. Can you help me?"
}]

input_ids = tokenizer.apply_chat_template(
    conversation=user_message,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors='pt'
).to(device)

output_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.2,
    top_k=20,
    top_p=0.9,
    repetition_penalty=1.3,
    typical_p=0.95,
    num_beams=3,
    no_repeat_ngram_size=2,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

generated_ids = output_ids[:, input_ids.shape[1]:]
response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

print("Generated Response:")
print(response)
```

---

## 📊 Training Summary

| Metric         | Value         |
| -------------- | ------------- |
| Training Steps | 327           |
| Final Loss     | 1.9309        |
| Total Runtime  | 17,184.85 sec |
| Samples/sec    | 0.613         |
| Steps/sec      | 0.019         |
| Total FLOPs    | 2.49e+17      |

---

## ⚠️ Disclaimer & Limitations

- **Not a replacement for therapy** or professional psychological help.
- **Not for crisis use** – always direct high-risk users to emergency services.
- May reflect **biases** present in training data.
- Use with **human oversight** in sensitive settings.

---

## 👨‍💻 Author

Developed by **Arafat Anam Chowdhury**

For feedback, collaborations, or improvements, feel free to connect!

---

## 📄 License

Licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
