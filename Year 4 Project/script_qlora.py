#!/usr/bin/env python3

import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
import torch
import os, re
from datetime import datetime


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")

# Load dataset
file_path = "medquad.csv"
# file_path = "Format Sample.csv"
df = pd.read_csv(file_path)
print(f"Dataset loaded. Number of rows: {len(df)}")
df = df[["question", "answer"]]

# Shuffling the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Limiting to the first 1000 rows
df = df.head(20000)


def preprocess_text(text):
    """
    Basic text cleaning function.
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    text = re.sub(r"\@\w+|\#", "", text)
    return text


def preprocess_dataframe(df, columns):
    """
    Preprocess the dataframe by applying text cleaning.
    Args:
    - df (pd.DataFrame): DataFrame with text data.
    - columns (list): List of columns to preprocess.
    Returns:
    - pd.DataFrame: Preprocessed DataFrame.
    """
    # Remove duplicates
    df = df.drop_duplicates()

    # Drop rows with any NaN values
    df = df.dropna()

    # Clean the text
    for col in columns:
        df[col] = df[col].apply(preprocess_text)

    return df


df = preprocess_dataframe(df, ["question", "answer"])


# Preprocess and tokenize text data
# Load model directly
print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token

# ChatML Template
tokenizer.chat_template = """
{% for message in messages %}
{{ message['role'] + ':\n' + message['content'] + '\n' }}
{% endfor %}
"""


def format_qa(conv):
    messages = [
        {"role": "user", "content": conv["question"]},
        {"role": "assistant", "content": conv["answer"]},
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False).strip()
    return {"text": chat}


def split_dataset(df):
    # Split the DataFrame first
    # Assuming df is your DataFrame containing the dataset
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)  # Adjust the test_size as needed
    train_df, val_df = train_test_split(train_val_df, test_size=0.18, random_state=42)  # Adjust to get ~10-15% validation data

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Testing set size: {len(test_df)}")

    train_dataset = train_df.apply(format_qa, axis=1)
    val_dataset = val_df.apply(format_qa, axis=1)
    test_dataset = test_df.apply(format_qa, axis=1)

    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = split_dataset(df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-chat-hf",
    quantization_config=BitsAndBytesConfig(
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    ),
    torch_dtype=torch.bfloat16,
)

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    # target_modules=[
    #     "q_proj",
    #     "k_proj",
    #     "down_proj",
    #     "v_proj",
    #     "gate_proj",
    #     "o_proj",
    #     "up_proj",
    # ],
    lora_dropout=0.1,
    bias="none",
    # modules_to_save = ["lm_head", "embed_tokens"],        # needed because we added new tokens to tokenizer/model
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.config.use_cache = False


def tokenize(element):
    return tokenizer(
        element["text"],
        truncation=True,
        max_length=2048,
        add_special_tokens=False,
    )


train_dataset_tokenized = train_dataset.map(tokenize)

val_dataset_tokenized = val_dataset.map(tokenize)

# train_dataset_tokenized = train_dataset.map(
#     tokenize,
#     batched=True,
#     num_proc=os.cpu_count(),  # multithreaded
#     remove_columns=[
#         "text"
#     ],  # don't need the strings anymore, we have tokens from here on
# )

print("Datasets tokenized.")


# collate function - to transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to single batch dictionary { input_ids: [..], labels: [..], attention_mask: [..] }
def collate(elements):
    tokenlist = [e["input_ids"] for e in elements]
    tokens_maxlen = max([len(t) for t in tokenlist])  # length of longest input

    input_ids, labels, attention_masks = [], [], []
    for tokens in tokenlist:
        # how many pad tokens to add for this sample
        pad_len = tokens_maxlen - len(tokens)

        # pad input_ids with pad_token, labels with ignore_index (-100) and set attention_mask 1 where content, otherwise 0
        input_ids.append(tokens + [tokenizer.pad_token_id] * pad_len)
        labels.append(tokens + [-100] * pad_len)
        attention_masks.append([1] * len(tokens) + [0] * pad_len)

    batch = {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_masks),
    }
    return batch


bs = 16  # batch size
ga_steps = 1  # gradient acc. steps
epochs = 10
steps_per_epoch = len(train_dataset_tokenized) // (bs * ga_steps)

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    evaluation_strategy="steps",
    logging_steps=25,
    eval_steps=steps_per_epoch,  # eval and save once per epoch
    save_steps=25,
    save_strategy="epoch",
    gradient_accumulation_steps=ga_steps,
    num_train_epochs=epochs,
    lr_scheduler_type="constant",
    optim="paged_adamw_32bit",
    learning_rate=0.0002,
    group_by_length=True,
    fp16=False,
    bf16=False,
    ddp_find_unused_parameters=False,  # needed for training with accelerate
    weight_decay=0.001,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=collate,
    train_dataset=train_dataset_tokenized,
    eval_dataset=val_dataset_tokenized,
    args=args,
)

print("Training started...")
trainer.train()
print("Training completed.")

# Get current date and time
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
print(f"Date-Time: {timestamp}")


model_save_path = f"./model/{timestamp}"
# Create the directory if it does not exist
os.makedirs(model_save_path, exist_ok=True)


model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model Saved at: {model_save_path}")
