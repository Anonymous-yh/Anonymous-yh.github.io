---
layout: post
title: "Reading Notes: DiffusionGemma Technical Report"
date: 2026-08-15
categories: [paper-reading]
tags: [diffusion-llm, paper-reading]
description: "Notes and questions from reading the DiffusionGemma technical report."
---

> **Paper:** *DiffusionGemma Technical Report*  
> **Organization:** Google DeepMind  
> **Model:** DiffusionGemma-26B-A4B-it  
> **Base Model:** Gemma 4 26B A4B  
> **Paradigm:** Discrete Diffusion Language Model / Block-Autoregressive Diffusion  
> **Release:** 2026

DiffusionGemma 是 Google DeepMind 发布的开放权重离散扩散语言模型。它是直接从 Gemma 4 26B A4B 的 post-trained checkpoint 出发，通过两阶段训练将一个 autoregressive model 转换成能够进行 bidirectional denoising 的 text diffusion model。论文报告其总参数量约 25.2B，每次实际激活约 3.85B 参数，并使用长度为 256 tokens 的 diffusion canvas。

其中引人注目的结果：
- 一个 diffusion forward 同时处理 **256-token canvas**；
- 高效：adaptive stopping 后平均只需要约 **12 effective denoising steps**；最终达到约 **19.7 Tokens Per Forward（TPF）**；单张 NVIDIA H100、FP8、batch size 1 下达到约 **1450–1500 output tokens/s**；相比 Gemma 4 AR + MTP 的约 303 tokens/s，解码速度接近 **4.8×**。
- 解决以 **较低训练成本** 将大规模、具有reasoning/multimodal能力的 AR 模型迁移到能够在 few-forward regime 下工作的离散 diffusion generation system

![Figure 1]({{ "/images/blog/DG-report/image1.png" | relative_url }})

## 离散扩散框架与理论基础

在 学习DiffusionGemma 架构之前，有必要对 离散扩散框架以及dLLM的理论基础有一个整体性的理解。与传统的AR模型相比，diffusion lm的优势主要在于：**双向注意力** 和 **并行解码**。双向注意力让模型能够根据未来token来双向修改token，并行解码让模型大幅度提升解码速度，提高生成效率。

将diffusion相关理论应用到文本生成上，需要处理自然语言的离散性。一些早期的工作(2022-2024)尝试将**连续（即基于 Gaussian 加噪去噪的）的扩散模型直接应用到语言模态**上，可以理解为将离散的token（离散数据）映射到连续嵌入空间中。例如 Diffusion-LM(Li et al.,2022) 和 SED (Strudel et al.,2022), 产生连续向量再投影或取整回离散token，或 CDCD(Dieleman et al.,2022) 直接预测分类logits到token。由于 continous diffusion 的光滑几何 和 categorical token 的离散语义不是匹配的，连续扩散模型在语言模态上的效果存疑。

不同于直接应用连续扩散模型，**离散扩散模型**直接在离散空间上进行扩散建模，和图像生成的motivation一样，我们在离散扩散空间(长度为L的序列，序列中每个token从一个有限大的词表中取值)定义一个前向马尔可夫过程(加噪)，将干净的文本数据逐步corrupt成一个特殊的stationary distribution。在离散扩散模型中，有两种常用的加噪过程和对应stationary distribution:

- **Masked Diffusion Model(掩码扩散模型)**: 前向过程每个干净token有一定概率保持不动，有概率置为[MASK]状态，并且mask后续不会有变化；
- **Uniform Diffusion Model**: 平稳分布为词表上的均匀分布，前向过程每个token有概率保持不动，有概率被替换为词表随机token。Uniform Diffusion Model显然更类似于传统的连续扩散模型，当然 The Diffusion Duality 这篇工作也证明了这一点。

由于很多早期的工作在相同scaling下发现 MDM 的表现相比uniform 更好，后续主流工作更多采用 MDM 作为离散扩散前向过程，包括LLaDA(scaling到7B),Dream(fine-tune from AR)都follow了MDM的技术路线。

关于 masked diffusion model 和 uniform diffusion model 优劣，经过近期很多工作的分析和实验，也有了更深入一些的认识：masked diffusion 和 uniform diffusion 的差异实际上是 **“可辨识性”(identifiability)** 和 **可逆修正能力(reversibility)** 之间的权衡。
- masked diffusion 通过用 [MASK] 明确标识哪些token被破坏，因此模型只需要完成缺失token的修复；缺点是所有高噪声状态都会坍缩到同一个确定性的 fully-masked state，teminal entropy 为0，在 few-step decoding 下表现不佳（参考 Multi-Mask Diffusion(2026)）；
- uniform diffusion 中的token可能是真实token可能是随机token，需要结合sampler得到对应的token的entropy，这导致uniform diffusion model更难训练；但是所有token都可以被修改，因此模型有更强的self-correction能力，这对于few-step generation比较重要。事实上 DiffusionGemma 也是考虑了这一点，才选用 uniform diffusion 作为其扩散建模方式。

关于 Diffusion Language Model 的理论基础，可以参考博客：[How to Build A Diffusion Language Model](https://kuleshov-group.github.io/blog/blog/2026/how-to-build-a-diffusion-language-model/)

## DiffusionGemma 架构

reference: [A Visual Guide to Diffusiongemma](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-diffusiongemma)

![Figure 2]({{ "/images/blog/DG-report/pipeline.png" | relative_url }})

### Encoder-Decoder

DiffusionGemma 基于 Gemma 4 26B A4B 训练，但由于 Gemma 4 是一个 decoder-only 模型，解决方案是 **Encoder-Denoiser patch**: 让Gemma 4动态切换 denoiser mode 和 encoder mode 实现。在 denoiser mode 下，模型类似于编码器对canvas进行去噪；在 encoder mode 下，模型类似于解码器

**Denoiser Mode: Act like an Encoder**

回顾一下自回归模型，首先将文本转换为token embeddings，随后这些token embeddings在 LLM 中不断被处理和更新，最终得到模型的隐藏状态(hidden states)并被投影为logits, 代表词表中每个词的置信度分数。最终仅使用最后一个hidden state来预测下一个token，其他所有logit都被丢弃。仅使用最后一个hidden state的原因是在自回归框架下，每个token只能“看到/关注"它之前的token，因此最后一个hidden state包含了所有之前token的信息。

基于这个过程，如果我们将输入序列替换为**待去噪的token canvas**，因果注意力替换为**双向注意力**，通过双向注意力，每个token都可以关注到canvas中其他token的信息，这样我们就会利用所有位置的logits了。对于canvas中每个位置，可以根据logits选择最匹配的token：如果概率较高，则被接受，如果概率较低，就被替换为其他的随机token(re-noised)。

![Figure 3]({{ "/images/blog/DG-report/image3.png" | relative_url }})

**Encoder Mode: Act like a Decoder**

以 noisy canvas 作为输入，并使用双向注意力，可以用Gemma 4B 作为 denoiser 迭代更新 canvas，但是如何填充 canvas 需要 Encoder Mode 提供指导。







### Block-Autoregressive Diffusion




### Self-Conditioning



### The Entropy Bounded Sampler



## 两阶段训练：SFT + Sampler Distillation & RL



## 实验结果、优势与局限性分析