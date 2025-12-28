import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch 
    import sys
    import circuitsvis as cv 
    from formerlens import HookedOneLayerTransformer
    return HookedOneLayerTransformer, cv, torch


@app.cell
def _(HookedOneLayerTransformer):
    model = HookedOneLayerTransformer.from_checkpoint("checkpoints/best_model.pt")
    model.eval()

    print(f"Model: {model.model_dim}d, {model.num_heads} heads, vocab={model.vocab_size}")
    EOS_TOKEN = 11
    return EOS_TOKEN, model


@app.cell
def _(EOS_TOKEN):
    def num_to_input(n, num_digits=2):
        """Convert number to model input tokens."""
        digits = []
        for _ in range(num_digits):
            digits.append(n % 10)
            n //= 10
        return digits + [EOS_TOKEN]

    def tokens_to_labels(tokens):
        """Convert tokens to string labels."""
        return [str(t) if t < 10 else "EOS" if t == 11 else "PAD" for t in tokens]
    return num_to_input, tokens_to_labels


@app.cell
def _(num_to_input, tokens_to_labels, torch):

    # %%
    # Test: 42 + 7 = 49
    test_num = 42
    tokens = torch.tensor([num_to_input(test_num)])
    token_labels = tokens_to_labels(tokens[0].tolist())
    print(f"Input: {test_num} → {token_labels}")

    return test_num, token_labels, tokens


@app.cell
def _(model, test_num, tokens):

    # Get attention patterns
    logits, cache = model.run_with_cache(tokens)
    attn_pattern = cache["atn.hook_pattern"]  # (batch, heads, query, key)

        # Prediction
    pred = logits[0, -1].argmax().item()
    expected = (test_num + 7) % 10
    print(f"Predicts: {pred} (expected: {expected}) {'✓' if pred == expected else '✗'}")


    return (attn_pattern,)


@app.cell
def _(attn_pattern, cv, token_labels):
    # CircuitsVis attention pattern visualization
    cv.attention.attention_patterns(
        tokens=token_labels,
        attention=attn_pattern[0],  # Remove batch dim: (heads, query, key)
    )
    return


@app.cell
def _(attn_pattern, cv, token_labels):
    # ## Attention Heads View

    # %%
    # Attention heads visualization (shows what each head attends to)
    cv.attention.attention_heads(
        tokens=token_labels,
        attention=attn_pattern[0],
    )
    return


@app.cell
def _(cv, display, model, num_to_input, tokens_to_labels, torch):
    test_cases = [12, 42, 73, 99]

    for num in test_cases:
        tokens = torch.tensor([num_to_input(num)])
        labels = tokens_to_labels(tokens[0].tolist())

        logits, cache = model.run_with_cache(tokens)
        attn = cache["atn.hook_pattern"]

        pred = logits[0, -1].argmax().item()
        expected = (num + 7) % 10
        status = "✓" if pred == expected else "✗"

        print(f"\n{num} + 7 = {num + 7} | Pred: {pred} {status}")
        display(cv.attention.attention_patterns(tokens=labels, attention=attn[0]))
    return (tokens,)


@app.cell
def _(EOS_TOKEN, cv, display, model, num_to_input, tokens_to_labels, torch):
    # ## Attention During Generation

    # %%
    def generate_with_viz(model, input_num, max_steps=4):
        """Generate and visualize attention at each step."""
        tokens = torch.tensor([num_to_input(input_num)])
        generated = []

        for step in range(max_steps):
            labels = tokens_to_labels(tokens[0].tolist())
            logits, cache = model.run_with_cache(tokens)
            attn = cache["atn.hook_pattern"]

            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)

            print(f"\nStep {step + 1}: seq={labels}, next={next_token}")
            display(cv.attention.attention_patterns(tokens=labels, attention=attn[0]))

            if next_token == EOS_TOKEN:
                break
            tokens = torch.cat([tokens, torch.tensor([[next_token]])], dim=1)

        return generated

    # Generate 42 + 7 = 49
    print("Generating output for 42 + 7:")
    result = generate_with_viz(model, 42)
    print(f"\nGenerated: {result} (expected: [9, 4, 0, 11])")
    return


@app.cell
def _(cv, model, num_to_input, tokens_to_labels, torch):
    # Visualize attention for head analysis
    tokens = torch.tensor([num_to_input(42)])
    labels = tokens_to_labels(tokens[0].tolist())
    _, cache = model.run_with_cache(tokens)
    attn = cache["atn.hook_pattern"]

    # Text-based explanation alongside visualization
    cv.attention.attention_heads(
        tokens=labels,
        attention=attn[0],
        attention_head_names=[f"Head {i}" for i in range(model.num_heads)],
    )
    return (tokens,)


if __name__ == "__main__":
    app.run()
