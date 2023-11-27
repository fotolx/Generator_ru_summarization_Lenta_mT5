# Generator_ru_summarization_Lenta_mT5

This model trained to summarize news post. Trained on data grabbed from russian news site Lenta.ru.

Модель обучена суммаризации новостных статей. Обучение проводилось на данных, полученных с русского новостного сайта Lenta.ru.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->
- **Developed by:** [i-k-a](https://huggingface.co/i-k-a)
- **Shared by [optional]:** [i-k-a](https://huggingface.co/i-k-a)
- **Model type:** Transformer Text2Text Generation
- **Language(s) (NLP):** Russian
- **Finetuned from model [optional]:** mT5-base

### Model Sources [optional]

<!-- Provide the basic links for the model. -->
- **Repository:** [link](https://huggingface.co/i-k-a/ru_summarization_lenta_model_mt5-base_7_epochs_1024/tree/main)

## How to Get Started with the Model

Use code below to infer model.

Используйте код ниже для запуска модели.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
MAX_NEW_TOKENS=400
MODEL_DIR='i-k-a/i-k-a/ru_summarization_lenta_model_mt5-base_7_epochs_1024'
text = input('Введите текст:')
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
inputs = tokenizer(text, return_tensors="pt").input_ids
outputs = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Резюме от нейросети: "{result}"\n\nИсходный текст: "{text}"')
```

## Training Details

Model trained 7 epochs. Length of input text is cut to 1024 tokens. Output is 400 tokens.
Trained using Google Colab resources.

## Technical Specifications

### Model Architecture and Objective
google/mt5-base

### Compute Infrastructure
Google Colab

#### Hardware
Google Colab T4 GPU

#### Software
Python