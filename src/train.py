import os
from pathlib import Path
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

BACKSPACE_TOKEN = "<|backspace|>"
CURSOR_TOKENS = {
    "<|cursor_left_1|>": -1,
    "<|cursor_right_1|>": 1,
    "<|cursor_left_10|>": -10,
    "<|cursor_right_10|>": 10,
    "<|cursor_left_100|>": -100,
    "<|cursor_right_100|>": 100,
    "<|cursor_left_1000|>": -1000,
    "<|cursor_right_1000|>": 1000,
}

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append({'text': line})
    return {'train': data}


def main():
    data = load_data('data/train.txt')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    special_tokens = [BACKSPACE_TOKEN] + list(CURSOR_TOKENS.keys())
    to_add = [t for t in special_tokens if t not in tokenizer.get_vocab()]
    if to_add:
        tokenizer.add_special_tokens({'additional_special_tokens': to_add})
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    def tokenize(example):
        return tokenizer(example['text'], truncation=True)

    # Convert to Dataset object
    from datasets import Dataset
    dataset = Dataset.from_list(data['train'])
    dataset = dataset.map(tokenize, batched=False)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir='model',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    try:
        from torch.distributed.tensor import DTensor  # type: ignore
    except Exception:
        import torch
        import importlib
        tdt = importlib.import_module('torch.distributed.tensor')
        if not hasattr(tdt, 'DTensor'):
            tdt.DTensor = type('DTensor', (torch.Tensor,), {})
        import transformers.modeling_utils as mu
        mu.DTensor = tdt.DTensor

    model.save_pretrained('model', safe_serialization=True)
    tokenizer.save_pretrained('model')

if __name__ == '__main__':
    main()
