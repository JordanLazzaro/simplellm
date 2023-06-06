# 🌵 **simplellm**

🌵 **simplellm** is a (in-progress) library for developing and training Transformer Language Models
with the goal of being as simple as possible.

## Project Goals

Build super simple abstractions for:
 - Performant layer and model definitions
 - Performant training loops
 - Performant data pipelines
 - Parameter Efficient Fine-Tuning Techniques

The first model to replicate in simplellm will be [MPT-7B](https://www.mosaicml.com/blog/mpt-7b)
which has a few interesting architecture details:
 - Attention with Linear Biases ([ALiBi](https://arxiv.org/abs/2108.12409))
 - [FlashAttention](https://arxiv.org/abs/2205.14135) (requires custom Triton kernel for ALiBi)
 - Embedding Layer Gradient Shrink (page 7 of [GLM-130B](https://arxiv.org/abs/2210.02414) paper)

Will also most likely be adopting some concepts from MosaicML's [Composer library](https://github.com/mosaicml/composer) at some point.