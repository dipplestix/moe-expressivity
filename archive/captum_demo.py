import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Captum Interpretability Demo

    This notebook demonstrates how to use Captum for model interpretability.
    We'll train a simple neural network on a classification task and then
    analyze which input features are most important for predictions.
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import matplotlib.pyplot as plt
    import numpy as np
    return DataLoader, TensorDataset, nn, np, optim, plt, torch


@app.cell
def _(mo):
    mo.md("""
    ## 1. Generate Synthetic Data
    """)
    return


@app.cell
def _(np, torch):
    # Generate synthetic classification data
    # The target depends primarily on features 0, 1, 2 (to make interpretability interesting)
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 1000
    n_features = 10

    # Generate random features
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Target depends on first 3 features with some noise
    # Class 1 if: 2*x0 + 1.5*x1 - x2 > 0
    decision = 2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + 0.5 * np.random.randn(n_samples)
    y = (decision > 0).astype(np.int64)

    # Convert to tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    # Split into train/test
    train_size = 800
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Class distribution (train): {y_train.sum().item()} positive, {len(y_train) - y_train.sum().item()} negative")
    return X_test, X_train, n_features, y_test, y_train


@app.cell
def _(mo):
    mo.md("""
    ## 2. Define and Train a Simple Model
    """)
    return


@app.cell
def _(nn):
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )

        def forward(self, x):
            return self.net(x)
    return (SimpleClassifier,)


@app.cell
def _(
    DataLoader,
    SimpleClassifier,
    TensorDataset,
    X_train,
    n_features,
    nn,
    optim,
    y_train,
):
    # Create model
    model = SimpleClassifier(input_dim=n_features, hidden_dim=32)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Create dataloader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training loop
    n_epochs = 50
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    print("Training complete!")
    return losses, model


@app.cell
def _(X_test, model, torch, y_test):
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        predictions = test_outputs.argmax(dim=1)
        accuracy = (predictions == y_test).float().mean().item()
        print(f"Test Accuracy: {accuracy:.2%}")
    return


@app.cell
def _(losses, plt):
    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Captum Interpretability Analysis

    Now let's use Captum to understand which features the model considers important.
    We'll use several attribution methods:
    - **Integrated Gradients**: Attributes importance by integrating gradients along a path
    - **Saliency**: Simple gradient-based attribution
    - **Feature Ablation**: Measures importance by removing features
    """)
    return


@app.cell
def _():
    from captum.attr import IntegratedGradients, Saliency, FeatureAblation
    return FeatureAblation, IntegratedGradients, Saliency


@app.cell
def _(FeatureAblation, IntegratedGradients, Saliency, X_test, model):
    # Set model to eval mode
    model.eval()

    # Select a few test samples for analysis
    n_samples_to_analyze = 50
    sample_inputs = X_test[:n_samples_to_analyze]

    # Define a wrapper to get class 1 probability
    def model_forward(x):
        return model(x)[:, 1]  # Probability of class 1

    # Integrated Gradients
    ig = IntegratedGradients(model_forward)
    ig_attributions = ig.attribute(sample_inputs, n_steps=50)

    # Saliency
    saliency = Saliency(model_forward)
    saliency_attributions = saliency.attribute(sample_inputs)

    # Feature Ablation
    fa = FeatureAblation(model_forward)
    fa_attributions = fa.attribute(sample_inputs)

    # Average attributions across samples
    avg_ig = ig_attributions.abs().mean(dim=0).detach().numpy()
    avg_saliency = saliency_attributions.abs().mean(dim=0).detach().numpy()
    avg_fa = fa_attributions.abs().mean(dim=0).detach().numpy()

    print("Attribution analysis complete!")
    print(f"Analyzed {n_samples_to_analyze} samples")
    return avg_fa, avg_ig, avg_saliency, ig


@app.cell
def _(mo):
    mo.md("""
    ## 4. Visualize Feature Importance
    """)
    return


@app.cell
def _(avg_fa, avg_ig, avg_saliency, n_features, np, plt):
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    feature_names = [f"Feature {i}" for i in range(n_features)]
    x_pos = np.arange(n_features)

    # Integrated Gradients
    axes[0].bar(x_pos, avg_ig, color="steelblue")
    axes[0].set_xlabel("Feature")
    axes[0].set_ylabel("Attribution (abs mean)")
    axes[0].set_title("Integrated Gradients")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([str(i) for i in range(n_features)])

    # Saliency
    axes[1].bar(x_pos, avg_saliency, color="coral")
    axes[1].set_xlabel("Feature")
    axes[1].set_ylabel("Attribution (abs mean)")
    axes[1].set_title("Saliency")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([str(i) for i in range(n_features)])

    # Feature Ablation
    axes[2].bar(x_pos, avg_fa, color="seagreen")
    axes[2].set_xlabel("Feature")
    axes[2].set_ylabel("Attribution (abs mean)")
    axes[2].set_title("Feature Ablation")
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([str(i) for i in range(n_features)])

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Compare with Ground Truth

    Remember, our synthetic data was generated with the rule:
    **Class 1 if: 2*x₀ + 1.5*x₁ - x₂ > 0**

    So we expect Features 0, 1, and 2 to have the highest importance,
    with Feature 0 being most important (coefficient 2), followed by
    Feature 1 (coefficient 1.5), then Feature 2 (coefficient -1).
    """)
    return


@app.cell
def _(avg_fa, avg_ig, avg_saliency, np, plt):
    # Ground truth importance (absolute coefficients)
    ground_truth = np.array([2.0, 1.5, 1.0, 0, 0, 0, 0, 0, 0, 0])

    # Normalize all for comparison
    def normalize(x):
        return x / x.max()

    fig2, ax = plt.subplots(figsize=(10, 5))

    width = 0.2
    x = np.arange(10)

    ax.bar(x - 1.5 * width, normalize(ground_truth), width, label="Ground Truth", color="black", alpha=0.7)
    ax.bar(x - 0.5 * width, normalize(avg_ig), width, label="Integrated Gradients", color="steelblue")
    ax.bar(x + 0.5 * width, normalize(avg_saliency), width, label="Saliency", color="coral")
    ax.bar(x + 1.5 * width, normalize(avg_fa), width, label="Feature Ablation", color="seagreen")

    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Normalized Importance")
    ax.set_title("Attribution Methods vs Ground Truth")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Single Sample Analysis

    Let's look at attributions for a single sample to see how the model
    reasons about individual predictions.
    """)
    return


@app.cell
def _(X_test, ig, model, plt, torch):
    # Analyze a single sample
    single_sample = X_test[0:1]

    # Get prediction
    with torch.no_grad():
        pred = model(single_sample)
        pred_class = pred.argmax(dim=1).item()
        pred_probs = torch.softmax(pred, dim=1)[0]

    # Get attributions for this sample
    single_ig = ig.attribute(single_sample, n_steps=50)

    print(f"Prediction: Class {pred_class}")
    print(f"Probabilities: Class 0: {pred_probs[0]:.3f}, Class 1: {pred_probs[1]:.3f}")
    print(f"\nInput features: {single_sample[0].numpy().round(2)}")

    # Visualize
    fig3, ax3 = plt.subplots(figsize=(10, 4))

    attrs = single_ig[0].detach().numpy()
    colors = ["green" if a > 0 else "red" for a in attrs]

    ax3.bar(range(10), attrs, color=colors)
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax3.set_xlabel("Feature Index")
    ax3.set_ylabel("Attribution")
    ax3.set_title(f"Integrated Gradients for Single Sample (Predicted: Class {pred_class})")
    ax3.set_xticks(range(10))

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    This demo showed how to use Captum's attribution methods:

    1. **Integrated Gradients** - Accumulates gradients along a path from baseline to input
    2. **Saliency** - Computes gradients of output w.r.t. input
    3. **Feature Ablation** - Measures change in output when features are removed

    All three methods correctly identified Features 0, 1, and 2 as the most important,
    which matches our ground truth data generation process.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
