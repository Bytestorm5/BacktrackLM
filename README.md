# BacktrackLM

This repository demonstrates a simple language model based on GPT-2 that can use a special `<|backspace|>` token to delete previously generated tokens. It also supports moving a text cursor during generation so the model can insert or remove text in place.

## Training

Install the required packages:

```bash
pip install torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers datasets
```

Run training (the script downloads the `wikitext-2-raw-v1` dataset from HuggingFace and automatically creates a mix of normal and correction samples):

```bash
python src/train.py
```

The trained model and tokenizer will be saved to the `model` directory.

## Inference

Use `src/inference.py` to generate text. The script implements a small decoding loop that interprets the following special tokens:

- `<|backspace|>` – delete the character before the cursor
- `<|cursor_left_1|>` / `<|cursor_right_1|>` – move the cursor left or right by one character
- `<|cursor_left_10|>` / `<|cursor_right_10|>` – move ten characters
- `<|cursor_left_100|>` / `<|cursor_right_100|>` – move one hundred characters
- `<|cursor_left_1000|>` / `<|cursor_right_1000|>` – move one thousand characters

After generation finishes the script prints both the raw tokens and the final edited text.

```bash
python src/inference.py
```

## Dataset composition

During preprocessing, 75% of the training examples are kept as-is while the remaining 25% are turned into correction tasks. For a correction task a random line from the dataset is treated as an incorrect answer and then deleted with enough `<|backspace|>` tokens before inserting the correct text. This gives the model experience removing wrong output and replacing it with the right text.
