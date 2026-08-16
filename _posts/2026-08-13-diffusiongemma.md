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

### Stage 1: SFT

SFT的任务是让模型适应离散文本扩散过程，对于一个输入样本，模型同时接收：
- 已经确定的上下文token(来自 KV-Cache)
- canvas（长度为256，包含噪声或masked token）
- 对应的去噪目标

训练模型最小化预测与Ground Truth Canvas 之间的交叉熵损失。

在 SFT 之后，观察到两个明显的限制：
1. 在高级推理和代码任务上，SFT 模型的表现仍然低于初始化时的自回归模型
2. 当去噪步骤被限制在较少数量时（如默认的 max_denoising_steps=48），容易出现"复读机”现象

### Stage 2: Sampler Distillation & RL

二阶段同时优化两个方向：
- Reward maximization: 提升回答质量、推理能力、代码能力和指令遵循能力；
- Sampler distillation：进行步数蒸馏。

训练数据来源于 Gemma 4 RL 所使用的训练数据，涵盖了 thinking 和 no-thinking 两种模式，覆盖 helpfulness、数学、代码和指令遵循等任务。

从 SFT checkpoint 开始，模型自身作为 online teacher，使用较大的最大去噪步数和较温和的温度退火，生成高质量的去噪轨迹。然后利用这些轨迹完成两件事：用奖励信号提升回答质量，将高质量生成压缩到更少的去噪步骤中。论文中强调使用一个 **unified online learning stage**，一次梯度更新同时优化两个目标，但没有给出完整的算法细节。

二阶段的 SD-RL 训练形成了隐式的课程学习：在训练早期，模型的预测熵较高，自适应停止较晚触发，teacher需要较多去噪步骤，训练样本会包含相对完整的长去噪轨迹；随着reward提升，模型对生成内容越来越确定，token预测熵降低，模型开始接触更多 的短轨迹样本；在训练后期，当模型可以稳定产生低熵预测时，训练分布自然转向 few-step trajectory，逐渐专门化到低延迟采样。论文中将这一现象称为有奖励目标和自适应停止机制共同产生的 **curriculum learning effect**。

## 推理优化

理解 DiffusionGemma 的推理优化，首先要区分两种模型的效率来源：

> AR 模型减少单步计算量，但必须执行很多串行步骤；Diffusion 模型增加单步计算量，却能在一次前向中并行处理多个 token。

AR 解码的优势是每一步只处理一个新 token，单步计算量较小，而且可以使用高度优化的 causal attention 和 KV cache。但它的限制也很明显：
- token 必须按顺序生成；
- 每一步都要访问模型权重和 KV cache；
- 低并发时，GPU 计算单元难以被充分利用；
- 生成速度容易受显存带宽和 KV cache 搬运限制。

文本扩散模型则对一个 token canvas 进行反复修正。DiffusionGemma 使用长度为 256 的 canvas，在多个去噪步骤中同时预测这些位置。每一步都处理整个 canvas，因此单步计算量更大；但不同位置的 token 可以并行计算，而且模型可以根据预测熵提前锁定 token 或结束去噪。记 $TPF$ 为单次前向传播产生的token数，$t_{fwd}$ 为单词去噪步骤的耗时，则 解码吞吐可以近似表示为：$ TPS = \frac{TPF}{t_{fwd}} $。

如果把 Diffusion 相对 AR 的单步延迟放大倍数即为 $r$，把并行处理token数记为 $TPF$，那么相对速度大致为：$\frac{TPF}{r}$。

DiffusionGemma 在效率上的主要限制：
- DiffusionGemma 每次处理 256 个 token，不同位置可能路由到不同专家，因此一次 forward 需要访问更多 MoE expert weights
- sampling 复杂度更高，需要对 256 个位置执行采样，包括softmax、self-conditioning embedding、entropy计算、token选择等。可能是未来最值得有优化的部分之一
- 论文使用 FlashAttention-4 优化，但 attention kernel 运行时间仍为约为 Gemma 4 AR 的 4 倍。

DiffusionGemma 的效率优势主要存在于低并发场景，论文观察到当并发请求达到约32个时，AR 模型开始获得吞吐优势：低 batch size 时，AR 受到内存带宽和 KV cache 访问限制，DiffusionGemma 用更多并行计算填补了GPU空闲，当 batch 增大时，AR 也能充分利用GPU，此时 DiffusionGemma 的额外FLOPs、sampling 和 双向注意力 成本开始占主导。

目前 DiffusionGemma 更适合的场景：
- 单用户实时交互；
- 低并发在线服务；
- JSON 和结构化输出；
- 代码编辑；
- 长上下文生成；
- 对 per-user latency 敏感的场景。


## 实验结果、优势与局限性分析

DiffusionGemma 最突出的实验结论是：它牺牲了一部分绝对能力，换来了显著更低的解码延迟；在单张 H100、低并发场景下，文本扩散模式达到约 1,479 tokens/s，而 Gemma 4 自回归模型配合多 token prediction 的速度约为 303 tokens/s。

### 实验设计

论文从四个运行模式评估 DiffusionGemma：
- 文本扩散（TD）模式；
- 自回归（AR）模式；
- thinking 模式；
- non-thinking 模式。

对比模型包括：
- 初始化模型 Gemma 4 26B A4B；
- 开放权重扩散模型 LLaDA 2.1 Flash 100B；
- Nemotron Diffusion 14B；
- 闭源模型 Mercury 2。

评测覆盖数学推理、代码生成、通识知识、多模态理解、指令遵循和 agentic task completion 等能力。具体 benchmark 包括 AIME、GPQA-Diamond、LiveCodeBench-v6、Codeforces、GSM8K、MMMLU、MMMU-Pro、IFEval 和 Tau-bench 等

论文同时报告了两类指标：
- 能力指标： benchmark accuracy 或 score；
- 推理效率指标： tokens per forward（TPF）、tokens per second（TPS）、有效去噪步数、总生成 token 数、总 forward 次数和端到端延迟。

### 实验结果分析

![Figure 6]({{ "/images/blog/DG-report/image6.png" | relative_url }})

1. 主要优势为**解码速度**：在单张 H100、FP8、batch size 为 1 的条件下，DiffusionGemma 文本扩散模式的平均速度为 1,479 tokens/s，Gemma 4 AR + MTP 为 303 tokens/s。前者约为后者的 4.9 倍。如果与普通 AR 模式相比，DiffusionGemma 的速度约为其 7.2 倍
2. **能力上接近（但没有超过） Gemma 4**：在相对成熟、结构明确任务上(例如GSM8K，HumanEval，IFEVal)，接近Gemma 4 AR；在更复杂的数学推理、专家知识和竞赛编程任务上有明显差距。
3. 与其他的扩散模型相比，有**更好的速度-能力平衡**，同时实现了较强的任务能力和较高的解码速度。
4. Thinking 与 non-thinking：Thinking 模式通常能够提高复杂任务上的准确率，但代价是生成更多 token，并需要更多去噪计算。

综合看 DiffusionGemma 的主要优势：
1. **低并发**场景下延迟极低
2. **结构化输出**（json，代码编辑，OCR，模块化文本）具有天然优势，模型可以在两到三次去噪步骤内完成收敛
3. 支持 **TD 和 AR** 两种生成模式，论文认为这种混合解码方式可能成为后续系统设计方向

主要局限：
1. **复杂推理能力低于AR基线**，论文归因于：没有进行原生扩散预训练；SFT阶段相对较短；SD-RL阶段明确优化低延迟，牺牲了一部分最终能力；Gemma 4 原有架构、优化和数据配方未必适合离散文本扩散；
2. **输出过于简洁**：SD·RL 后的模型生成长度接近 SFT 模型的一半，可能是DiffusionGemma在复杂任务上表现不如AR基线的原因之一；
3. **token stuttering**：模型偶尔会出现“复读机”现象，在极少步数生成情形下，模型的生成稳定性存在问题；
4. **多模态任务存在格式控制问题**：在多模态 thinking 任务中，模型有时无法可靠生成 closing thought tag，即使推理过程本身正确，也可能影响评测和最终输出；
5. **高并发下优势减弱**：并发请求达到32后，AR模型获得吞吐优势。不过目前还没有针对 sampling 的 kernel 选择和 batch scaling 做专门优化工作。

## 附录中的补充实验与实践观察

论文附录补充了几个主实验没有完全展现的结论：DiffusionGemma 不仅可以用于高速推理，也可以通过参数高效微调适配结构化推理和领域文本生成任务。

### 下游微调：LoRA 可以改变模型的生成行为

论文提供了基于 Hackable Diffusion 的开源微调工具，并同时训练 causal encoder 和 diffusion decoder。LoRA 被应用到 attention projection、MLP gate、MoE router 和 self-conditioning block 等线性层。

在 Sudoku 任务上，LoRA rank 8 只训练约 8M 个参数，模型准确率从未微调时的 0 提升到 84.4%，有效去噪步数也从 40.65 降至 10.72。这说明领域微调不仅可以提高任务准确率，也可能降低任务上的预测熵，使模型更快收敛。

在 PubMedQA 上，LoRA 微调将 BLEU 从 10.76 提升到 20.67，但有效去噪步数从 18.09 增加到 31.57。这说明微调后的模型不一定总是更快；对于开放式文本生成，额外计算可能是换取质量提升的必要代价。

### 定性案例：双向推理与动态计算

附录中的案例说明，DiffusionGemma 的优势并不只是 TPS。由于 canvas 内的 token 可以双向交互，模型能够在后续 denoising steps 中修正早期预测。

同时，adaptive stopping 会根据任务结构动态分配计算：具有严格顺序依赖的序列需要更多步骤，而适合并行解析的任务可以更快收敛。对于强结构约束的 JSON 输出，模型甚至可以在两次去噪后完成收敛。

### 实践注意事项

DiffusionGemma 对 prompt template 和特殊控制 token 比较敏感。thinking、multimodal input 和 tool calling 都依赖严格的序列化格式。论文还提醒，多模态 thinking 中偶尔出现 closing thought tag 缺失，可能导致实际评测结果低于模型真实推理能力。

此外，论文对 Mercury 2 的速度是通过 API 黑盒估计得到的，因此跨模型 TPS 比较需要谨慎。
