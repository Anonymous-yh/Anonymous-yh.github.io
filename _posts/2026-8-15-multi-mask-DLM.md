---
layout: post
title: "Reading Notes: Multi-Mask Diffusion Language Model"
date: 2026-08-15
categories: [paper-reading]
tags: [diffusion-llm, paper-reading]
description: ""
---


> **Paper:** *Multi-Mask Diffusion Language Models for Few-Step Generation*  
> **Organization:** ByteDance Seed  
> **Paradigm:** Masked Diffusion Language Model  
> **Focus:** Few-Step Generation / Consistency Distillation  
> **Release:** 2026  
> **link:** [Multi-Mask Diffusion Language Models for Few-Step Generation](https://www.alphaxiv.org/zh/abs/2607.19686) 

## Introduction

加速离散模型的困难源于两个基本问题：
- **终端退化 (Terminal Degeneracy)**: MDM 的前向过程最终坍缩至单一的 fully-masked state，终端状态熵为零，无法为 Consistency Distillation 提供有效信息；
- **路径随机性 (Path Stochasticity)**: 离散状态空间的跳跃过程固有随机性强，难以实现像连续空间中那样确定性的 Probability-Flow 映射。

离散扩散模型中的 Uniform Diffusion Model 用随机 token 替换 [MASK] 避免终端退化的问题，但难以区分有语义token和噪声token，因此在训练上难度较大。本研究提出的 **Multi-Mask Diffusion Model** 通过引入多种掩码状态和专门的蒸馏过程，尝试解决以上问题。

## Multi-Mask Diffusion：机制与训练推理算法

### Multi-Mask Forward Process

MultiMDM 定义 **mask vocabulary**: $$\mathcal{M} = \{m_1, m_2, ..., m_k\}$$，diffusion 的完整状态空间为：$$ \mathcal{V} \cup \mathcal{M} $$，每个token $$ x_0^l $$ 分配到对应的mask状态 $$ m_{x_0^l} $$(称为 **designated mask**) 。从干净标记$$ x_0 $$ 到噪声状态 $$ x_t $$ 的转换由两个时间依赖的scheduler控制：$$ \alpha_t $$ 和 $$ \beta_t $$，$$ \alpha_t $$ 表示标记保持在其原始干净状态的概率，$$ \beta_t $$ 控制 mask state 从 designated mask 想 随机mask 混合的程度。具体的 forward process 定义为：

$$
p_t(x_t\mid x_0)=\alpha_t \mathbf{1}\{x_t=x_0\} + (1-\alpha_t)r_t^{x_0}(x_t), \qquad x_t\in\mathcal V\cup\mathcal M.
$$

$$
r_t^{x_0}(k):=\beta_t \mathbf{1}\{k = m_{x_0}\}+(1-\beta_t)u_{\mathcal M}(k),\qquad k\in\mathcal M.
$$

标记 $$ i $$ 的前向边缘概率 $p_t(x_t = k \| x_0 = i)$ 定义为：

$$
p_t(x_t = k | x_0 = i) = \begin{cases} \alpha_t, & k = i \\ (1 - \alpha_t)\beta_t + \frac{1}{M}(1 - \alpha_t)(1 - \beta_t), & k = m_i \\ \frac{1}{M}(1 - \alpha_t)(1 - \beta_t), & k \in M \setminus \{m_i\} \end{cases}
$$

推导出 forward kernel 和 backward kernel 如下：

$$
p_{t|s}(x_t\mid x_s)=
\begin{cases}
\alpha_{t|s}\mathbf{1}\{x_t=x_s\}
+
(1-\alpha_{t|s})r_t^{x_s}(x_t),
& x_s\in\mathcal V, \\[6pt]
\beta_{t|s}\mathbf{1}\{x_t=x_s\}
+
(1-\beta_{t|s})u_{\mathcal M}(x_t),
& x_s\in\mathcal M,
\end{cases}
$$

$$
p_{s|t}(x_s\mid x_t,x_0)=
\begin{cases}
\mathbf{1}\{x_s=x_0\},
& x_t=x_0, \\[6pt]
\displaystyle
\frac{\alpha_s-\alpha_t}{1-\alpha_t}\mathbf{1}\{x_s=x_0\}
+
\frac{1-\alpha_s}{1-\alpha_t}
\frac{r_s^{x_0}(x_s)\left[\beta_{t|s}\mathbf{1}\{x_t=x_s\}+(1-\beta_{t|s})u_{\mathcal M}(x_t)\right]}{r_t^{x_0}(x_t)}
\mathbf{1}\{x_s\in\mathcal M\},
& x_t\in\mathcal M.
\end{cases}
$$

对于 Multi-Mask Diffusion 的机制，有三点补充：
- **Designated-entry interpretation**: 论文证明了可以构造一个有相同单时刻边缘分布的 CTMC: clean token 首先跳到对应的 designated mask，然后再在 mask space 中随机混合，则反向过程可以理解为 先生成粗粒度的draft，再refine到最终的clean token。需要注意这只是一个等价的直观解释，实际实现还是遵循上面的forward kernel 公式。
- **masks数量**: 论文给出了一个粗略估计：$ L\log M\gtrsim \log N \quad\Longrightarrow\quad M\gtrsim N^{1/L}$，并不需要非常大的 mask vocabulary，实验中选取 $M=5\sim100$ 已经覆盖了有效的范围。论文在 OpenWebText 上测试了 $ M\in\{1,5,10,20,50,100\} $的结果，在 50 左右达到最佳，过大的mask数量会带来收益递减甚至损害性能。
- **designated mask**: 方法本身允许任意固定的 designated mask 映射；在论文中的实验中，为了简单起见，直接根据 tokenizer ID 使用 $ m_x = x\mod M $。

### 训练与推理算法

MultiMDM 从离散 diffusion 的 KL/ELBO 推导出 closed-form training objective，核心可以理解为：$\boxed{ \mathcal L_{\mathrm{MultiMDM}} = \mathcal L_{\mathrm{reconstruction}} + \mathcal L_{\mathrm{intra-mask}}} $：
- 第一项为 **clean token 重建**，鼓励模型根据带噪声上下文 $x_t$ 预测原始的 clean token $x_0$，是单mask diffusion model的标准训练目标。
- 第二项为 **掩码内识别**：鼓励模型正确识别哪个指定的掩码属于该clean token。

具体的训练目标 $L_{dist}(\theta)$ 表示为:

$$
\mathcal{L} = \mathbb{E} \left[ \sum_{\ell: x_{\ell,t} \in M} \left( \frac{-\dot{\alpha}_t}{1-\alpha_t} \log x_{\ell,t,\theta}(x_{\ell,0} | x_t) + \frac{-\dot{\beta}_t}{M\beta_t} \sum_{k \in M} \mathbb{1}_{k=m_{x_{\ell,0}}} \log x_{\ell,t,\theta}(k | x_t) \right) \right]
$$

训练目标比较复杂，但是实际训练算法是标准的：
- 从数据分布采样 clean sequence，同时为每个样本随机采样时间 $t$；
- 根据 forward kernel ，构造对应的noisy state $x_t$；
- 将 $x_t$ 输入模型，计算训练目标 $L_{dist}(\theta)$ ，正常反向传播更新参数。

在此训练框架下，可以**从预训练的 MDM 热启动**，将预训练好的单掩码 embedding 复制到所有掩码 embedding，使所有掩码在训练开始时都是相同的。

推理阶段：
- 初始化为 mask vocabulary 的均匀分布；
- 在每个时间步，将当前noisy state $x_t$ 输入模型，预测 clean-token 后验分布 $p_\theta(x_0\|x_t)$；将模型预测的 clean-token 后验分布 与 解析的backward kernel $p_{s\|t}(x_s\|x_t,x_0)$ 结合，得到 $p_{s\|t}^\theta(x_s\|x_t)$
- 采样得到下一状态 $x_s$，重复上述过程直到 $t=0$。已经恢复为 clean token 的位置在之后的反向过程中保持不变。

![Figure 1]({{ "/images/blog/multi-mask-DLM/image1.png" | relative_url }})

## 通过 Gumbel 耦合进行离散一致性蒸馏

在 MultiMDM 框架下，为解决之前提到的 路径随机性 (Path Stochasticity) 问题，通过一种称为 **共享 Gumbel 耦合** 的技术实现了类似连续扩散蒸馏的效果。

> **Gumbel-max trick**：假设有一个离散分布 $p(z)$，$z \in \{1,2,...,V}$，对每个状态 z 独立采样一个标准 Gumbel 噪声 $g_z$，然后选择得分最高的状态: $z^* = \arg\max_z (\log p(z) + g_z)$，有一个重要结论是：$ P(z^* = z) = p(z) $。

共享 Gumbel 耦合的做法是：只采样一次 Gumbel 噪声，并在所有时间点重复使用：
$$\tilde{x}_t = \arg\max_z ( \log p_t(z\mid x_0)+g_z )$$ 

在不同时间点的 $x_s$, $x_t$ 使用的是同一组 {$g_z$}，它们由同一个随机源共同决定，不是独立的随机样本。共享 Gumbel 同时保留了两个重要性质：
- 根据 Gumbel-max trick，每个时间点的边缘分布仍然正确；
- 给定共享 Gumbel 噪声后，整条采样路径是确定的，即随机性集中在初始抽取的共享噪声中，模型整体不再具有随机性。

基于共享 Gumbel 耦合的方法，具体的离散一致性蒸馏算法如下：

1. 初始化教师模型和一致性模型：首先训练一个MultiMDM 模型得到参数 $\theta$ 作为蒸馏的初始模型，随后复制一份参数作为 EMA 目标网络；
2. 采样干净序列和两个时间点（$ t\sim\mathrm{Uniform}(\delta,1), s=t-\delta\$），对序列每个位置$l$和每个状态 $z \in \mathcal{M} \cup \mathcal{V}$ 独立采样一个标准 Gumbel 噪声，并对于两个时间点分别用 Gumbel-max trick 计算 $\tilde{x}_t$ 和 $\tilde{x}_s$；
3. 将较干净的$\tilde{x}_s$输入到 EMA 目标网络，较噪的 $\tilde{x}_t$ 输入到当前模型，预测第$l$个位置的 clean token；
4. 计算一致性损失：每个序列位置上的KL散度并求和，直观含义是对于同一条共享 Gumbel 轨迹上的两个状态，模型应该对最终的 clean token 做出一致预测。
$$
\mathcal L_{\mathrm{CD}}^\delta
=
\mathbb E
\left[
\sum_{\ell=1}^{L}
D_{\mathrm{KL}}
\left(
\operatorname{sg}
\left[
f^\ell_{s,\theta_{\mathrm{EMA}}}
(\cdot\mid\tilde{x}_s)
\right]
\middle\|
f^\ell_{t,\theta_{\mathrm{online}}}
(\cdot\mid\tilde{x}_t)
\right)
\right].
$$
5. 梯度下降更新在线模型参数，然后使用指数移动平均更新 EMA 目标网络（$\theta_{\mathrm{EMA}} \leftarrow \mu\theta_{\mathrm{EMA}} + (1-\mu)\theta_{\mathrm{online}}$）。

论文将蒸馏过程分为5轮，每轮包含每轮包含 $K_R=10^4$ 次优化，并使用二倍递增的步长($\delta_k = 2^{-9+\left\lfloor k/K_R\right\rfloor}$)。先学习小时间间隔的一致性，再逐步学习更大时间间隔的跳跃。

蒸馏后的模型可以在推理阶段使用更少的时间步(更粗粒度的时间间隔)，达到相当的生成质量。

## 实验结果：性能与Scalability

MultiMDM 的实验主要为了回答三个问题：
- 多掩码设计是否改善了教师模型本身？
- 共享 Gumbel 一致性蒸馏能否支持少步生成？
- 多掩码设计能否从受控的 $170\text{M}$ 模型扩展到更大的预训练扩散语言模型？

**评估指标**：论文采用两种比较方式，结合采样温度和GenPPL
1. Entropy-aligned: 调节每个方法温度，使生成样本具有相同的熵，比较GenPPL；
2. Temperature-aligned: 固定温度（设为1），比较GenPPL.

![Figure 2]({{ "/images/blog/multi-mask-DLM/image2.png" | relative_url }})

在 OpenWebText 和 LM1B 上比较了几种模型的预训练性能：
- MDLM：单掩码 masked diffusion；
- DUO，即 The Diffusion Duality 中的 uniform-state diffusion；
- CANDI：混合离散—连续扩散模型；
- MultiMDM-rand：从随机初始化开始训练；
- MultiMDM-cont：从预训练 MDLM 继续适配

结论：**MultiMDM-cont 不仅适合后续蒸馏，它本身也比单掩码 MDLM 具有更好的生成质量**。并且从预训练 MDLM 继续训练 MultiMDM 表现更好，不必完全从头训练。

![Figure 3]({{ "/images/blog/multi-mask-DLM/image3.png" | relative_url }})

论文比较了：
- MultiMDM + shared-Gumbel consistency distillation；
- MDLM + shared-Gumbel；
- DUO-DCD 的 shared-Gaussian coupling；
- 使用 SDTT 的另一种蒸馏方案。

总体趋势是**蒸馏后的 MultiMDM 曲线更靠近低困惑度、高熵区域，尤其在 4 步等低采样预算下更明显**。以及 MultiMDM 在 SDTT 蒸馏下也能得到更强的蒸馏模型，说明性能的提升一部分来自于 MultiMDM 本身，而不完全依赖于特定蒸馏算法。

![Figure 4]({{ "/images/blog/multi-mask-DLM/image4.png" | relative_url }})

将 Multi-Mask Scaling 到 LLaDA-8B-Base 模型得到 MultiLLaDA，具体做法是：
- 将 LLaDA 的单掩码状态空间扩展为 $M=50$ 个 mask，加入新的 mask-token embedding
- 保留原有 clean-token 输出头；
- 使用 LoRA 进行低秩适配。

在 MATH500、GSM8K、HumanEval 和 MBPP 上比较原始 LLaDA 与 MultiLLaDA，并使用 TPS=1,2,4 三种反向采样预算。初步说明 MultiMDM 不局限于小规模模型。不过这里使用的是 LoRA 适配，并且没有给出 8B 规模的 shared-Gumbel 蒸馏的结果。

## Problem

1. 关于 Desginate mask 的选择，论文用了简单的 tokendizer ID 取模的方式，是否有更好的选择策略？
2. 核心的实验规模比较小，主实验只有 170M DiT，在 LM1B 和 OpenWebText 上做 unconditional generation，还不能说明是否高质量语言生成的能力？
3. 8B scaling 实验：MultiLLaDA 相比 baseline，用 DCLM + OpenCode 数据多训练了 5000 steps，和原始LLaDA 比较是否公平？

