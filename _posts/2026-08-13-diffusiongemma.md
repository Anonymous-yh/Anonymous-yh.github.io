---
layout: post
title: "Reading Notes: DiffusionGemma Technical Report"
date: 2026-08-13
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
> **link:** [DiffusionGemma Technical Report](https://arxiv.org/abs/2608.00146)

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

DiffusionGemma 基于 Gemma 4 26B A4B 训练，但由于 Gemma 4 是一个 decoder-only 模型，解决方案是 **Encoder-Denoiser patch**: 让Gemma 4动态切换 denoiser mode 和 encoder mode 实现。在 denoiser mode 下，模型类似于编码器对canvas进行去噪；在 encoder mode 下，模型类似于解码器。

由于 DiffusionGemma 和 Gemma 4 有相同的transformer 架构，因此**保留了自回归生成的能力**，可以使用因果注意力进行标准的AR生成，性能得分落在 文本扩散模式 和 baseline Gemma 4 checkpoint 之间。

**Denoiser Mode: Act like an Encoder**

回顾一下自回归模型，首先将文本转换为token embeddings，随后这些token embeddings在 LLM 中不断被处理和更新，最终得到模型的隐藏状态(hidden states)并被投影为logits, 代表词表中每个词的置信度分数。最终仅使用最后一个hidden state来预测下一个token，其他所有logit都被丢弃。仅使用最后一个hidden state的原因是在自回归框架下，每个token只能“看到/关注"它之前的token，因此最后一个hidden state包含了所有之前token的信息。

基于这个过程，如果我们将输入序列替换为**待去噪的token canvas**，因果注意力替换为**双向注意力**，通过双向注意力，每个token都可以关注到canvas中其他token的信息，这样我们就会利用所有位置的logits了。对于canvas中每个位置，可以根据logits选择最匹配的token：如果概率较高，则被接受，如果概率较低，就被替换为其他的随机token(re-noised)。

![Figure 3]({{ "/images/blog/DG-report/image3.png" | relative_url }})

**Encoder Mode: Act like a Decoder**

以 noisy canvas 作为输入，并使用双向注意力，可以用Gemma 4B 作为 denoiser 迭代更新 canvas，但是如何填充 canvas 需要 Encoder Mode 处理输入query，为后续denoising提供条件信息。

DiffusionGemma 复用 Gemma 4 作为 Encoder，保持原本的 causal attention。由于这里的目的不是预测下一个token，因此不会使用最后的 LM Head，复用模型的 KV Cache 作为 Encoder 输出，即 DiffusionGEmma 在 Encoder Mode 和 Denoiser Mode 下共享 KV-Cache。**Encoder KV** 相当于整个 diffusion trajectory 中固定的条件信息，在一个 canvas 的多次 denoising steps 中保持不变并反复复用。

### Self-Conditioning

DiffusionGemma 的生成过程是：canvas 1 -> canvas 2 -> ... -> final output，如果每一步只能看到当前的noisy canvas，模型的预测能力受限。如果能知道模型在前一步所做的预测，这样能在上一轮判断的基础上进行refine。具体实现的方式是 **Self-Conditioning**: 将 Softmax 之后的 logits 乘以所有token 的 embedding matrix，再过一个小型的 FFNN，并在下一个 denoising step 加到 canvas 的 token embedding 上。

结合 Self-Conditioning，我们得到完整的 DiffusionGemma 架构：

![Figure 4]({{ "/images/blog/DG-report/image4.png" | relative_url }})

*Figure 4. DiffusionGemma 完整架构*

### Block-Autoregressive Diffusion

此前的讨论都是生成长度为 256 token 的定长序列，针对长文本生成情形，DiffusionGemma 采用 **Block-Autoregressive Diffusion**。实现思路也并不复杂，在原来的单个canvas的基础上，当我们完成一个canvas的去噪过程后，将这 256 个 token 添加到 Encoder 的输入序列中以扩展 KV-Cache，然后 Denoiser 继续产生下一个 256-token canvas，直至生成 EOS token。

由于 Encoder 的 KV-Cache 是通过 causal attention 计算的，每个token只要处理之前的部分，只需要计算每次 AR step 增加的 KV-Cache，不需要重新计算之前的。

### Scheduler

在 DiffusionGemma 中， Scheduler 的作用是控制去噪过程，但是不会直接决定拒绝或者接受哪些token（这个由 Sampler 决定）。具体如何“调度” 去噪过程，由以下三个组件确定：
- **step count**: 决定最大去噪步数，默认设定为 48 steps。步数越多以为着输出质量越高，但是生成速度较慢；
- **logits scheduler**: 将 logits 除以 温度值(temperature) 控制采样的随机性， temperature 与去噪步数呈线性关系且步数越多temperature越低。在生成早期阶段，temperature 较高，模型会更多地探索，而非最有可能的token；在生成后期阶段，temperature 较低，模型更多关注最可能出现的token，让输出内容更合理。
- **adaptive stopping**: 模型可能在最大去噪步数之前就收敛，因此在每一步都会检查 **Stability**（最近N步最高概率token预测是否相同） 和 **Confidence**（对canvas中所有token的平均置信度）。DiffusionGemma 的默认配置中，置信度阈值为0.005，稳定性阈值为1。若达到阈值无论剩余多少步都会停止去噪过程。通过 adaptive stopping，虽然默认配置下 max_denoising_steps 为 48，在一些常用bench上测试平均只需 12 steps 左右就能收敛。

### The Entropy Bounded Sampler

DiffusionGemma 默认和推荐使用的 Sampler 是 **Entropy Bounded Sampler**，其主要作用如下：
- **Initialization**：初始化为随机token(均匀分布)，和图像扩散从纯噪声开始类似；
- **Token Acceptance**：用 Entropy 衡量 canvas 中每个token的置信度。Sampler 计算 canvas 中每个位置的 entropy 并从低到高排序，根据图5的公式，逐一检查是否接受。
- **Token Renoise**: 被拒绝的token会被重新采样为随机token，进入下一轮去噪。


![DiffusionGemma architecture]({{ "/images/blog/DG-report/image5.png" | relative_url }})

*Figure 5. Entropy Bounded Sampler 的token接受标准*

DiffusionGemma 也并非严格绑定 Entropy Bounded Sampler，也可以尝试其他的Sampler。


## 两阶段训练：SFT + Sampler Distillation & RL



## 实验结果、优势与局限性分析