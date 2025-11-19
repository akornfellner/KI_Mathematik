# AI and Mathematics — Overview

This repository contains slides about core ideas in artificial intelligence and the mathematics behind them. The original slides are available at [slides/slides.md](slides/slides.md). The project is licensed under the MIT License: [LICENSE](LICENSE).

Presentation can be found [here](https://akornfellner.github.io/KI_Mathematik/)

## Topics covered

- Overview of AI, Machine Learning (ML) and Deep Learning (DL).
- Neural networks: model structure, weighted sums and activation functions.
  - Neuron output: $$\text{Output} = \sigma\left(\sum_{i=1}^{n} w_i \cdot x_i\right)$$
  - Example activation (Sigmoid): $ \sigma(x) = \dfrac{1}{1+e^{-x}} $
- Training algorithms:
  - Loss functions (e.g., MSE) and gradient-based optimization.
  - Gradient descent update: $w \leftarrow w - \alpha \cdot \dfrac{\partial L}{\partial w}$
  - Backpropagation: computing partial derivatives via the chain rule to update weights.
- Matrix notation and benefits (compact representation, parallelism, GPU speedups): e.g., $$\mathbf{h} = \sigma(\mathbf{W_1}\mathbf{x})$$
- Modern network types: CNNs, RNNs, LSTMs, and Transformers.
- Vector embeddings:
  - Converting items (words, sentences, images) to high-dimensional vectors.
  - Similarity metrics: Euclidean distance, Manhattan distance, cosine similarity, dot product.
- Large Language Models (LLMs) and Transformers:
  - Self-attention as the core mechanism enabling long-context understanding.
  - Training by next-token prediction and concerns like hallucinations.
- Prompt engineering:
  - Techniques for clearer, more controllable prompts (role assignment, specificity, structured requests).

## Use cases and classroom ideas

- Use small, hand-calculated network examples to teach the chain rule and derivatives.
- Demonstrate embeddings with visual tools (e.g., TensorFlow Projector).
- Discuss ethical considerations, limitations (hallucinations), and verification.

## Where to look in this repo

- Main slides: [slides/slides.md](slides/slides.md)
- License: [LICENSE](LICENSE)

Special thanks to Dominik Freinhofer — some parts of these slides are based on his presentation. More at: https://www.dominikfreinhofer.com/