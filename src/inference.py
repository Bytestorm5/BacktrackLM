import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

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


def generate(text: str, model_path: str = "model", max_steps: int = 50):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    special_tokens = [BACKSPACE_TOKEN] + list(CURSOR_TOKENS.keys())
    missing = [t for t in special_tokens if t not in tokenizer.get_vocab()]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})

    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))

    token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in special_tokens}

    input_ids = tokenizer.encode(text, return_tensors="pt")
    generated = input_ids[0].tolist()

    out_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    cursor = len(out_text)

    for _ in range(max_steps):
        with torch.no_grad():
            logits = model(torch.tensor([generated])).logits[0, -1]
        next_token_id = int(torch.argmax(logits))
        generated.append(next_token_id)

        if next_token_id == token_ids[BACKSPACE_TOKEN]:
            if cursor > 0:
                out_text = out_text[: cursor - 1] + out_text[cursor:]
                cursor -= 1
            continue

        move = None
        for tok, offset in CURSOR_TOKENS.items():
            if next_token_id == token_ids.get(tok):
                move = offset
                break
        if move is not None:
            cursor = max(0, min(len(out_text), cursor + move))
            continue

        token_str = tokenizer.decode([next_token_id], clean_up_tokenization_spaces=False)
        out_text = out_text[:cursor] + token_str + out_text[cursor:]
        cursor += len(token_str)

        if next_token_id == tokenizer.eos_token_id:
            break

    raw = tokenizer.decode(generated, skip_special_tokens=True)
    return raw, out_text

if __name__ == '__main__':
    text = input('Prompt: ')
    raw, corrected = generate(text)
    print('Raw output:', raw)
    print('Corrected:', corrected)
