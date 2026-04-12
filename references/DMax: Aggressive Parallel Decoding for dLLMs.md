## Contents
- 1 Introduction
- 2 Preliminaries
- 3 Methodology
  - 3.1 On-Policy Uniform Training
  - 3.2 Soft Parallel Decoding
- 4 Experiments
  - 4.1 Experimental Setup
  - 4.2 Experimental Results
- 5 Ablation Study
- 6 Related Work
- 7 Conclusion
- References

## Abstract

Abstract We present DMax, a new paradigm for efficient diffusion language models (dLLMs). It mitigates error accumulation in parallel decoding, enabling aggressive decoding parallelism while preserving generation quality. Unlike conventional masked dLLMs that decode through a binary mask-to-token transition, DMax reformulates decoding as a progressive self-refinement from mask embeddings to token embeddings. At the core of our approach is On-Policy Uniform Training, a novel training strategy that efficiently unifies masked and uniform dLLMs, equipping the model to recover clean tokens from both masked inputs and its own erroneous predictions. Building on this foundation, we further propose Soft Parallel Decoding. We represent each intermediate decoding state as an interpolation between the predicted token embedding and the mask embedding, enabling iterative self-revising in embedding space. Extensive experiments across a variety of benchmarks demonstrate the effectiveness of DMax. Compared with the original LLaDA-2.0-mini, our method improves TPF on GSM8K from 2.04 to 5.47 while preserving accuracy. On MBPP, it increases TPF from 2.71 to 5.86 while maintaining comparable performance. On two H200 GPUs, our model achieves an average of 1,338 TPS at batch size 1. Code is available at: https://github.com/czg1225/DMax

## 1 Introduction

Recently, Diffusion Language Models (dLLMs) [93, 95, 98, 42, 56, 108] have emerged as a compelling alternative to the long-standing dominance of Autoregressive Language Models (AR-LLM) [1, 6, 25] in text generation. The primary allure of dLLMs lies in their capacity for parallel decoding, which holds great promise for improving inference efficiency

Despite this promise, the practical decoding parallelism of existing dLLMs [58, 109, 92, 17, 10, 90] remains limited, as their performance drops sharply under aggressive parallel decoding. Some prior work has attempted to improve this trade-off through improved decoding [37, 81, 41, 27, 8, 32, 34, 89] or distillation strategies [16, 62, 101, 38]. Nevertheless, these methods do not address the fundamental bottleneck underlying parallel decoding in current dLLM paradigm: error accumulation.

In current mask-based dLLMs, decoding is a binary, one-way mask-to-token process. Once a masked position is decoded into a token, that token is fixed and propagated as context to subsequent decoding steps, with no opportunity for revision. Under highly parallel decoding, erroneous predictions are inevitable. Once such errors are committed, they contaminate future predictions and trigger cascading error accumulation, ultimately leading to semantic collapse. Unlike speculative decoding [39, 11, 43], dLLMs lack a mechanism to recover from incorrect predictions, which fundamentally restricts their performance under highly parallel decoding. Addressing this challenge requires a new dLLM paradigm with an intrinsic capability to revise its own predictions during decoding.

Building on this insight, we propose DMax, a novel paradigm that reformulates the binary mask-to-token decoding process into a self-revising transformation in the embedding space. Central to our approach is On-Policy Uniform Training (OPUT), a training recipe that efficiently extends a pretrained masked diffusion language model into a self-corrective uniform diffusion language model while preserving its original mask denoising capability. Unlike conventional uniform diffusion training that constructs noisy sequences by randomly sampling tokens from the vocabulary, OPUT samples noisy inputs on-policy from the model’s own predictive distribution. This substantially bridges the train-inference gap and enables the model to effectively learn to correct its own potential prediction errors. Building upon OPUT, we further present Soft Parallel Decoding (SPD) for inference. Instead of treating decoded tokens as discrete and irrevocable commitments, SPD represents each intermediate decoding state as a hybrid soft embedding, formed by interpolating between the predicted token embedding and the mask embedding according to the model’s prediction confidence. This simple design provides the model with confidence priors from previous steps, enabling more robust self-correction.

Using LLaDA-2.0-mini [10], a state-of-the-art open-source dLLM, as the base model, we validate the effectiveness of our method across multiple widely used benchmarks. On the mathematical reasoning benchmark GSM8K [20], our method increases tokens per forward (TPF) from 2.04 to 5.48 with only minimal accuracy degradation relative to the original model. On the code generation benchmark MBPP [5], it improves TPF from 2.71 to 5.86 while maintaining comparable performance.

In summary, we propose DMax, a novel paradigm that enables highly parallel decoding for dLLMs while preserving strong performance. Our central idea is to mitigate the error accumulation issue caused by the conventional one-way mask-to-token decoding. To realize this, we introduce two key designs: on-policy uniform training and soft parallel decoding. Extensive experiments demonstrate the effectiveness and superiority of our approach. This work establishes a new strong baseline for future research on parallel decoding in dLLMs.

## 2 Preliminaries

We begin by briefly reviewing the diffusion language modeling paradigms, and then highlight the central challenge for highly parallel decoding and introduce our key motivation.

Masked Diffusion Language Models (MDLMs).
MDLMs [70, 4, 66, 105, 51] formulate text generation as a discrete denoising process over token sequences, where clean tokens are progressively replaced by a special [MASK] symbol during corruption. Let $x_{0}=(x_{0}^{1},\dots,x_{0}^{L})\in\mathcal{V}^{L}$ denote a clean sequence of length $L$, where $\mathcal{V}$ is the vocabulary. Given a corrupted sequence $x_{t}$ at noise level $t\in[0,1]$, the denoising model is trained to recover the original tokens only at masked positions. The standard MDLM objective is

$$ $\mathcal{L}_{\mathrm{MDLM}}(\theta)=-\mathbb{E}_{x_{0},\,t,\,x_{t}}\left[\frac{1}{t}\sum_{i=1}^{L}\mathbf{1}(x_{t}^{i}=\texttt{[MASK]})\log p_{\theta}(x_{0}^{i}\mid x_{t})\right].$ (1) $$

At inference time, MDLMs start from a fully masked sequence and iteratively decode masked positions in parallel, with an optional remasking step to enable further refinement.

Uniform Diffusion Language Models (UDLMs).
UDLMs [68, 67, 69] generalize the corruption process by replacing tokens with uniformly sampled vocabulary tokens rather than a dedicated [MASK] symbol. As a result, the model is trained to recover clean tokens from arbitrary noisy token inputs, instead of only from masked positions. A standard UDLM training objective is

$$ $\mathcal{L}_{\mathrm{UDLM}}(\theta)=-\mathbb{E}_{x_{0},\,t,\,x_{t}}\left[\sum_{i=1}^{L}\log p_{\theta}(x_{0}^{i}\mid x_{t})\right].$ (2) $$

During inference, UDLMs typically start from a fully noisy sequence sampled uniformly from the vocabulary and iteratively update all positions.

Error accumulation in MDLMs.
Existing dLLMs based on the MDLM paradigm degrade sharply under highly parallel decoding, which limits the practical speedup. The main reason behind it is error accumulation. MDLM decoding follows a binary mask-to-token process: each position is either a mask token or a committed token. Once a masked position is decoded, its prediction is treated as fixed context for subsequent steps. Early mistakes cannot be revised, and instead propagate through later denoising steps as erroneous context.

UDLMs as a Promising Solution.
In contrast, UDLMs are trained to denoise from arbitrary vocabulary tokens rather than only from [MASK], so all positions can be re-evaluated at every decoding step. This token-to-token denoising mechanism naturally enables self-correction and improves robustness to prediction errors. However, UDLM decoding typically starts from a fully random sequence, which makes denoising harder and leads to very unstable generation.

Unify the Strengths of MDLMs and UDLMs.
Motivated by this trade-off, we propose to unify the strengths of both paradigms. Specifically, we retain a fully masked sequence as the initialization of UDLM decoding to preserve stability, while continuing to re-predict all tokens that have been decoded from [MASK] at every subsequent step. This design combines the stable initialization with the self-revising capability, enabling a more robust parallel decoding process.

## 3 Methodology

Figure: Figure 2: Overview of the proposed On-Policy Uniform Training.
Refer to caption: https://arxiv.org/html/2604.08302/2604.08302v1/x2.png

### 3.1 On-Policy Uniform Training

A practical way to achieve this goal is to extend a pretrained MDLM into a UDLM. Accordingly, our first objective is to endow a pretrained MDLM with the self-revision capability of UDLMs while preserving its original mask denoising ability.

Extending MDLM toward UDLM is Nontrivial. This is nontrivial because the training objective of UDLMs differs substantially from that of MDLMs and is considerably harder to optimize. In the standard UDLM training paradigm, a clean sequence is first corrupted by randomly selecting a subset of positions and replacing the selected tokens with tokens sampled uniformly from the vocabulary. The resulting noisy sequence is then used as model input, and the model is trained to recover the original clean sequence.

However, this training strategy is often unstable in practice and tends to yield suboptimal performance. A key reason is that uniformly sampled tokens lie far outside the natural language manifold, producing highly unnatural corrupted inputs. As a result, the model must spend substantial capacity merely learning to map these corrupted sequences back toward plausible language, rather than directly acquiring effective language modeling and self-correction behaviors. More importantly, this corruption process introduces a severe train–inference mismatch. Unlike conventional UDLMs, our paradigm first predicts tokens from masked positions in parallel and then iteratively refines its own predictions. Consequently, the noisy sequences encountered at inference time are sampled from the model’s own output distribution rather than from a uniform vocabulary distribution. This mismatch hinders self-correction and leads to ineffective training.

On-Policy Uniform Training. To address these issues, we propose On-Policy Uniform Training (OPUT), a simple yet effective method for equipping MDLMs with self-corrective denoising capability. The core idea is to construct training inputs using noisy sequences sampled on-policy from the model’s own predictive distribution, rather than from a uniform vocabulary distribution, thereby bridging the train–inference gap. The overview of the training procedure is shown in Figure [2](#S3.F2).

Training Procedure.
Let $M_{\theta}$ denote a pretrained diffusion language model built on the MDLM paradigm, parameterized by $\theta$.
We further adapt $M_{\theta}$ on a training dataset $\mathcal{D}$ of clean sequences $x_{0}=(x_{0}^{1},\dots,x_{0}^{L})$. At each training iteration, we first sample a corruption level $t\sim\mathrm{Uniform}(t_{l},t_{h})$, where $t_{l}$ and $t_{h}$ denote the lower and upper bounds of the noise level, respectively. Given a clean sequence $x_{0}\sim\mathcal{D}$, we construct a masked noisy sequence $x_{t}^{(m)}$ by independently replacing each token with [MASK] with probability $t$.

We feed $x_{t}^{(m)}$ into $M_{\theta}$ and predict all masked positions in parallel.
By sampling from the model’s predictive distribution at masked positions, we obtain a predicted noisy sequence $x_{t}^{(p)}$, defined as

$$ $x_{t}^{(p),i}=\begin{cases}x_{t}^{(m),i},&\text{if }x_{t}^{(m),i}\neq\texttt{[MASK]},\[4.0pt] \hat{x}^{i},\quad\hat{x}^{i}\sim p_{\theta}(\cdot\mid x_{t}^{(m)}),&\text{if }x_{t}^{(m),i}=\texttt{[MASK]}.\end{cases}$ (3) $$

Importantly, $x_{t}^{(p)}$ is sampled using the current model parameters at each iteration, making this a strictly on-policy rollout process.

Next, we perform two forward passes, using the masked noisy sequence $x_{t}^{(m)}$ and the predicted noisy sequence $x_{t}^{(p)}$ as inputs, respectively:

$$ $p_{\theta}^{(m)}(\cdot\mid x_{t}^{(m)})=M_{\theta}(x_{t}^{(m)}),\qquad p_{\theta}^{(p)}(\cdot\mid x_{t}^{(p)})=M_{\theta}(x_{t}^{(p)}).$ (4) $$

We then supervise both outputs against the original clean sequence $x_{0}$ using cross-entropy loss over *all* token positions, regardless of whether a position is masked:

$$ $\mathcal{L}_{\mathrm{mask}}=-\sum_{i=1}^{L}\log p_{\theta}^{(m)}(x_{0}^{i}\mid x_{t}^{(m)}),\qquad\mathcal{L}_{\mathrm{pred}}=-\sum_{i=1}^{L}\log p_{\theta}^{(p)}(x_{0}^{i}\mid x_{t}^{(p)}).$ (5) $$

The final training objective is

$$ $\mathcal{L}_{\mathrm{on\mbox{-}policy}}=\mathcal{L}_{\mathrm{mask}}+\mathcal{L}_{\mathrm{pred}},$ (6) $$

By reducing the train–inference mismatch, the proposed OPUT strategy enables a pretrained MDLM to efficiently learn self-correction through limited post-training, while retaining its original mask denoising ability. As a result, the model can correct self-generated errors and effectively mitigate error accumulation under highly parallel decoding. On LLaDA-2.0-mini, our method improves GSM8K accuracy from $78\%$ to $90\%$ under confidence-threshold decoding with a threshold of $0.5$, while also delivering faster decoding.

Figure: Figure 3: Overview of the Soft Parallel Decoding procedure in DMax.
Refer to caption: https://arxiv.org/html/2604.08302/2604.08302v1/x3.png

### 3.2 Soft Parallel Decoding

Although OPUT substantially mitigates error accumulation, it still struggles when many erroneous predictions arise simultaneously within a block. When many positions are decoded in parallel, correlated errors can appear at once, making them difficult to fully correct through iterative refinement. For example, for OPUT-trained LLaDA-2.0-mini, if we decode all masked positions in a block at once using a confidence threshold of 0, and then iteratively refine them, the accuracy on GSM8K drops to only $68\%$.

Soft Parallel Decoding.
To further enhance self-revising in iterative refinement, we propose soft parallel decoding. The central idea is to preserve predictive uncertainty from earlier iterations and explicitly propagate it to later refinement steps. Concretely, instead of treating intermediate decoding states as discrete tokens, we represent each decoded token as a soft embedding interpolated between the predicted token embedding and the mask embedding. Because the mask embedding naturally encodes maximal uncertainty, this interpolation serves as an explicit carrier of uncertainty across iterations. This enables the model to better distinguish confident predictions from unreliable ones, allowing it to focus on refining low-confidence tokens while avoiding interference from noisy signals.

Decoding Procedure.
An overview of the decoding process is shown in Figure [3](#S3.F3). It follows a block-wise semi-autoregressive process. For each block, we partition its positions into two sets: *mask positions* and *token positions*. At initialization, all positions in the block are mask positions. At each decoding step, we use an aggressive confidence threshold $\tau_{\mathrm{dec}}$ to promote some mask positions into token positions. Specifically, we scan the masked region from left to right and promote only its longest contiguous prefix whose confidence exceeds $\tau_{\mathrm{dec}}$. Once the first mask position with confidence below $\tau_{\mathrm{dec}}$ is encountered, all mask positions to its right remain masked. If no mask position satisfies this criterion, we still promote the leftmost mask position to ensure decoding progress. This design keeps the masked region contiguous and prevents unreliable future tokens on the right from interfering with mask predictions on the left.

At decoding step $t$, every mask position uses the mask embedding as model input:

$$ $\mathbf{h}^{(t)}_{j}=\mathbf{e}_{\mathrm{mask}},\qquad j\in\mathcal{M}^{(t)}.$ (7) $$

where $\mathcal{M}^{(t)}$ denotes the set of mask positions at step $t$.

For each token position $j\in\mathcal{T}^{(t)}$, where $\mathcal{T}^{(t)}$ is the set of token positions, we construct a hybrid embedding from the top-1 prediction at the previous step $t-1$ as the model input. Let
$y^{(t-1)}_{j}$
denote the top-1 predicted token at position $j$, and let
$\pi^{(t-1)}_{j}$
be its predicted probability. We assign the remaining probability mass to the mask embedding:

$$ $\pi^{(t-1)}_{j,\mathrm{mask}}=1-\pi^{(t-1)}_{j}.$ (8) $$

The unnormalized hybrid embedding is then

$$ $\tilde{\mathbf{h}}^{(t)}_{j}=\pi^{(t-1)}_{j}\,\mathbf{e}\!\left(y^{(t-1)}_{j}\right)+\pi^{(t-1)}_{j,\mathrm{mask}}\,\mathbf{e}_{\mathrm{mask}},\qquad j\in\mathcal{T}^{(t)}.$ (9) $$

Directly adding high-dimensional embeddings may distort their magnitude and lead to norm collapse. To avoid this issue, we renormalize the hybrid embedding so that its norm matches the probability-weighted sum of the component norms:

$$ $\mathbf{h}^{(t)}_{j}=\frac{\tilde{\mathbf{h}}^{(t)}_{j}}{\left\|\tilde{\mathbf{h}}^{(t)}_{j}\right\|_{2}}\left(\pi^{(t-1)}_{j}\left\|\mathbf{e}\!\left(y^{(t-1)}_{j}\right)\right\|_{2}+\pi^{(t-1)}_{j,\mathrm{mask}}\left\|\mathbf{e}_{\mathrm{mask}}\right\|_{2}\right).$ (10) $$

This hybrid embedding serves as a soft intermediate state between decoding steps, explicitly carrying forward the uncertainty of previous predictions.

We regard a block as having converged to a stable state if either of the following conditions holds: (1) the top-1 predictions at all positions remain unchanged for two consecutive decoding steps, or (2) the confidence of every position in the block exceeds a high acceptance threshold $\tau_{\mathrm{acc}}$. Once a block converges, we commit all token positions in the block according to the final predictions and move on to the next block.

Figure: Algorithm 1 Soft Parallel Decoding (Block-Wise)

By interpolating prediction-mask embeddings as intermediate states, the model receives an explicit uncertainty prior before every forward pass, leading to substantially more robust parallel decoding. On OPUT-trained LLaDA-2.0-mini, under the highly aggressive setting of $\tau_{\mathrm{dec}}=0$, soft parallel decoding improves GSM8K accuracy from $68\%$ to $90\%$ while achieving a higher speedup.

OPUT as a Prerequisite. Notably, soft parallel decoding must be used together with OPUT-trained models. OPUT trains the model to recover the correct target not only from masked inputs, but also from its own sampled predictions. As a result, the model learns a consistent mapping from both mask embeddings and self-predicted token embeddings toward the correct output, which makes interpolation between them meaningful. In contrast, applying soft parallel decoding to a standard diffusion language model without OPUT leads to catastrophic performance collapse.

**Table 1: Comparison with the original model and different baselines. For our DMax-Math model, we set the decoding threshold to 0.5; for the DMax-Coder model, we set it to 0.65. In addition to TPF, TPS, and accuracy, we also report the AUP score to provide a more comprehensive evaluation of parallel decoding performance. All evaluations are under zero-shot and a batch size of 1.**
| Benchmark | Method | TPF $\uparrow$ | TPS $\uparrow$ | Acc. $\uparrow$ | AUP Score $\uparrow$ |
| --- | --- | --- | --- | --- | --- |
| Math & Reasoning Benchmarks |  |  |  |  |  |
| GSM8K | LLaDA-2.0-mini | 2.04 | 512 | 92.6% | 340 |
|  | Hierarchical Decoding | 2.44 | 577 | 91.6% | 357 |
|  | dParallel SFT | 2.79 | 721 | 92.3% | 395 |
|  | Uniform Diffusion Training | 2.26 | 493 | 68.7% | 0 |
|  | DMax-Math | 5.48 | 1258 | 92.1% | 557 |
| MATH500 | LLaDA-2.0-mini | 2.58 | 626 | 75.8% | 257 |
|  | Hierarchical Decoding | 3.01 | 669 | 73.0% | 268 |
|  | dParallel SFT | 3.42 | 823 | 75.8% | 310 |
|  | Uniform Diffusion Training | 2.43 | 530 | 33.6% | 0 |
|  | DMax-Math | 5.94 | 1286 | 75.4% | 507 |
| Minerva-Algebra | LLaDA-2.0-mini | 3.01 | 755 | 91.4% | 363 |
|  | Hierarchical Decoding | 3.40 | 787 | 90.6% | 382 |
|  | dParallel SFT | 3.91 | 943 | 91.4% | 430 |
|  | Uniform Diffusion Training | 2.55 | 551 | 42.7% | 0 |
|  | DMax-Math | 7.03 | 1492 | 91.5% | 658 |
| ASDIV | LLaDA-2.0-mini | 2.03 | 512 | 92.8% | 354 |
|  | Hierarchical Decoding | 2.43 | 528 | 92.5% | 366 |
|  | dParallel SFT | 2.72 | 663 | 93.0% | 459 |
|  | Uniform Diffusion Training | 2.51 | 515 | 80.8% | 0 |
|  | DMax-Math | 5.62 | 1172 | 92.5% | 556 |
| Code Generation Benchmarks |  |  |  |  |  |
| HumanEval-Instruct | LLaDA-2.0-mini | 4.38 | 1044 | 84.2% | 369 |
|  | Hierarchical Decoding | 4.67 | 1014 | 81.1% | 379 |
|  | dParallel SFT | 5.12 | 1229 | 76.8% | 394 |
|  | Uniform Diffusion Training | 2.93 | 628 | 15.2% | 0 |
|  | DMax-Coder | 7.36 | 1557 | 83.5% | 637 |
| MBPP-Instruct | LLaDA-2.0-mini | 2.71 | 662 | 80.6% | 276 |
|  | Hierarchical Decoding | 2.88 | 685 | 76.6% | 241 |
|  | dParallel SFT | 3.66 | 880 | 74.7% | 273 |
|  | Uniform Diffusion Training | 2.84 | 608 | 23.4% | 0 |
|  | DMax-Coder | 5.86 | 1264 | 79.2% | 482 |

## 4 Experiments

### 4.1 Experimental Setup

Implementation Details.
We build our method on LLaDA-2.0-mini [10], a state-of-the-art open-source diffusion language model. During training, we use OPUT with a fixed mask ratio of 0.75. We perform full-parameter fine-tuning for 2 epochs with a batch size of 8, an initial learning rate of $2\times 10^{-6}$, and a cosine learning rate schedule. Training follows the block-diffusion setting with a block size of 32. To avoid extra memory overhead, the masked noisy sequence and the predicted noisy sequence are optimized in separate iterations within the same epoch, rather than jointly in a single iteration. Under this setup, we train two models: DMax-Math for mathematical reasoning and DMax-Coder for code generation tasks. All training runs are conducted on 8 H200 GPUs. At inference time, we adopt the proposed SPD decoding strategy under semi-autoregressive block diffusion with a block size of 32. The acceptance threshold for determining whether a block has converged to a stable state is set to $\tau_{\mathrm{acc}}=0.9$.

Training Data.
We construct all training data through self-distillation. Specifically, we take prompts from public datasets and use LLaDA-2.0-mini to generate responses as training targets. For math, prompts are collected from GSM8K trainset [20], PRM12K [44], a subset of Numina-Math [40], and a subset of OpenThoughts [26]. For code, prompts are drawn from a subset of OpenCodeInstruct [2]. Responses are generated with a confidence threshold of 0.95, a block size of 32, and a maximum generation length of 2048 tokens. We discard incomplete generations that do not finish within the length budget. This yields 0.7M math samples and 1.0M code samples. Notably, we do not use any external high-quality responses; all supervision is obtained from the model’s own generations.

Evaluation Details.
We evaluate our method on multiple benchmarks. For mathematical reasoning, we use GSM8K [20], MATH500 [44], Minerva-Algebra [29], and ASDIV [55], and prompt the model to produce chain-of-thought [80] reasoning. For code generation, we use the instruction versions of HumanEval [13] and MBPP [5]. All evaluations are conducted with the dInFer [54] framework on 2 H200 GPUs using tensor parallelism. Besides TPF, TPS, and accuracy, we also report AUP Score [62] to measure parallel decoding performance. The generation length for all benchmarks is 2048.

Baselines.
We compare our method against four baselines in terms of both decoding efficiency and generation accuracy:
(1) LLaDA-2.0-mini, the base model, evaluated with its default confidence-threshold-based parallel decoding strategy using a threshold of 0.95;
(2) Hierarchical Decoding, an advanced inference strategy that improves parallel decoding via a divide-and-conquer procedure [61]. The low threshold is set as 0.2;
(3) dParallel-SFT, for which we use the LLaDA-2.0-mini-CAP model [10], where the certainty-forcing loss proposed in dParallel [16] is incorporated into large-scale supervised fine-tuning to improve decoding parallelism; and
(4) Uniform Diffusion Training, which continues training the base model using the conventional UDLM objective. In addition to masked noisy sequences, this baseline also replaces tokens with random vocabulary samples to construct uniformly corrupted noisy sequences, while keeping all other training settings identical to those of DMax. During inference, it updates all tokens within a block at every step until convergence.

Figure: Figure 4: Comparison of accuracy-TPF trade-off curves between original LLaDA-2.0-mini model and our method. We present curves on GSM8K, MATH500, HumanEval and MBPP benchmarks.
Refer to caption: https://arxiv.org/html/2604.08302/2604.08302v1/x4.png

**Table 2: Our proposed new paradigm also improves the model’s accuracy at low parallelism.**
| Benchmarks | LLaDA-2.0-mini | DMax |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  | TPF $\uparrow$ | Acc. $\uparrow$ | TPF $\uparrow$ | Acc. $\uparrow$ |
| GSM8K | 2.04 | 92.6% | 3.54 (+1.50) | 93.4% (+0.8%) |  |
| MATH500 | 2.58 | 75.8% | 3.45 (+0.87) | 78.0% (+2.2%) |  |
| Minerva-Algebra | 3.01 | 91.4% | 4.96 (+1.95) | 93.6% (+2.2%) |  |
| ASDIV | 2.03 | 92.8% | 3.10 (+1.07) | 93.5% (+0.7%) |  |
| HumanEval-Instruct | 4.38 | 84.2% | 4.58 (+0.20) | 87.2% (+3.0%) |  |
| MBPP-Instruct | 2.71 | 80.6% | 3.58 (+0.87) | 83.4% (+2.8%) |  |

### 4.2 Experimental Results

Aggressive Parallelism While Preserving Accuracy.
As shown in Table [1](#S3.T1), compared with the original LLaDA-2.0-mini, our method substantially increases decoding parallelism, improving the average TPF from 2.8 to 6.2 while preserving the original accuracy. In contrast, the other baselines provide only limited gains in parallel decoding. This advantage is further reflected in the AUP Score, where DMax consistently outperforms both the original model and all baselines by a large margin. These results demonstrate that our paradigm enables a much stronger parallel decoding capability than conventional MDLMs. Moreover, on two H200 GPUs, our model achieves a practical inference throughput of over 1000 tokens per second.

On-Policy Training as the Cornerstone.
Table [1](#S3.T1) also compares our method with conventional uniform diffusion training. The latter neither improves decoding speed nor preserves model quality, instead causing a noticeable performance drop. We find that this failure stems from the large mismatch between the randomly sampled noisy sequences used in training and the model’s actual decoding trajectories at inference time. Consequently, the model struggles to revise erroneous predictions while unnecessarily perturbing correct ones, resulting in unstable oscillations within each block. By contrast, our on-policy training samples noisy sequences from the model’s own outputs, effectively bridging this train–inference gap and substantially improving self-revision under parallel decoding.

Superior Efficiency–Performance Trade-off.
Figure [4](#S4.F4) compares the accuracy–TPF trade-off curves of our method and the original model on GSM8K, MATH500, HumanEval, and MBPP. As TPF increases, the original model suffers a sharp accuracy drop, whereas our method maintains stable performance. For instance, on MATH500, at around 6.5 TPF, our method still retains over 71.6% accuracy, while the original model falls to 15.2%. The gap is even larger on code benchmarks: on MBPP, at a similar TPF, our method achieves 79.2%, whereas the original model drops to only 2.3%. This superior trade-off stems from the self-revision capability of our paradigm, which effectively mitigates error accumulation under aggressive parallel decoding.

Improved Performance at Low Parallelism.
By enabling dLLMs to revise their own predictions, our method not only mitigates error accumulation under aggressive parallel decoding, but also improves performance in the low-parallelism regime. Through iterative re-evaluation of earlier predictions, the model can recover from reasoning errors that would otherwise remain on the original decoding path. As shown in Table [2](#S4.T2), our method consistently improves accuracy by 0.8%–3.0% across multiple benchmarks at low parallelism. Importantly, these gains are obtained using only the model’s own generated responses as training data, without introducing any external supervision.

**Table 3: Ablation on different training and inference strategies with different decoding parallelism.**
| Train | Inference | $\tau_{\mathrm{dec}}=0.95$ | $\tau_{\mathrm{dec}}=0.50$ | $\tau_{\mathrm{dec}}=0.0$ |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| On-Policy<br>Rollout | Contiguous<br>Prefix | Hybrid<br>Embedding | TPF $\uparrow$ | Acc. $\uparrow$ | TPF $\uparrow$ | Acc. $\uparrow$ | TPF $\uparrow$ | Acc. $\uparrow$ |
|  |  |  | 2.04 | 92.6% | 4.47 | 78.0% | 7.86 | 0.9% |
|  | ✓ | ✓ | 1.04 | 0.0% | 1.73 | 0.0% | 5.39 | 0.0% |
| ✓ |  |  | 2.95 | 92.6% | 5.14 | 90.1% | 5.89 | 68.2% |
| ✓ | ✓ |  | 2.85 | 93.0% | 5.28 | 91.3% | 5.98 | 69.6% |
| ✓ |  | ✓ | 3.25 | 92.8% | 5.64 | 91.4% | 6.01 | 90.4% |
| ✓ | ✓ | ✓ | 3.00 | 93.3% | 5.48 | 92.1% | 6.01 | 90.4% |

## 5 Ablation Study

Ablation Study on Training and Inference Strategies.
Table [3](#S4.T3) presents a comprehensive ablation study of both our training and inference designs. We compare different combinations of training and decoding strategies on GSM8K under three decoding thresholds, $\tau_{\mathrm{dec}}\in\{0.95,0.5,0.0\}$. On-policy rollout is the core of our training method. Even with OPUT alone, the model acquires the ability to revise its own errors, yielding substantial accuracy gains over the original model at $\tau_{\mathrm{dec}}=0.5$ and $0.0$. Our proposed SPD further improves robustness when many erroneous predictions emerge simultaneously, allowing the model to remain stable under highly parallel decoding and to preserve strong performance even in the extreme case of $\tau_{\mathrm{dec}}=0.0$. The key ingredient of SPD is to use soft embeddings, rather than discrete tokens, as intermediate decoding states. Maintaining the non-masked region as a contiguous prefix further improves performance. Another important result is that OPUT is a prerequisite for SPD. As shown in Table [3](#S4.T3), directly applying SPD to the original model causes generation to collapse. This is because OPUT trains the model to recover clean tokens from both mask tokens and predicted tokens, making interpolation between their embeddings a meaningful and effective input for denoising.

**Table 4: Ablation study on block-level convergence criteria. The decoding threshold is set to 0.5.**
| Convergence | GSM8K | MBPP |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Consistency | Confidence | TPF $\uparrow$ | Acc. $\uparrow$ | TPF $\uparrow$ | Acc. $\uparrow$ |
| ✓ |  | 5.13 | 92.1% | 5.16 | 79.9% |
|  | ✓ | 2.28 | 92.2% | 3.36 | 80.1% |
| ✓ | ✓ | 5.48 | 92.1% | 5.86 | 79.2% |

Ablation Study on Convergence Criteria.
We further study in Table [4](#S5.T4) how different block-level convergence criteria affect the efficiency–performance trade-off. We consider two criteria: (1) *consistency*, where decoding is considered converged if the model produces the same top-1 prediction for the block in two consecutive steps; and (2) *confidence*, where decoding is considered converged if the confidence of every token in the block exceeds 0.9. As shown in Table [4](#S5.T4), consistency serves as the primary convergence signal, with most blocks terminating once this condition is met. Adding the confidence criterion can further improve TPF by allowing decoding to stop before two consecutive identical predictions are observed, thereby saving the final forward pass. Importantly, neither criterion affects the accuracy.

## 6 Related Work

Diffusion Language Models.
Diffusion models [31, 71] have become dominant in visual generation [64, 60, 65, 99], and recent work has explored their application to text generation. Among existing paradigms, masked diffusion language models (MDLMs) [70, 4, 66, 105, 51] have emerged as a promising alternative to AR-LLMs by modeling language in discrete space through masked token prediction. Building on this formulation, LLaDA [58] and Dream [92] scale MDLMs to the billion-parameter regime with large-scale pretraining, demonstrating their practical potential. LLaDA-2.0 [10] and LLaDA-MoE [110] further show that MDLMs can be effectively scaled with mixture-of-experts architectures. Beyond these developments, dLLMs are also attracting increasing attention in reasoning [109, 59, 86, 63, 74, 57, 103], multimodal tasks [94, 96, 90, 91, 48, 82, 97, 18], code generation [87, 24, 21], long-context modeling [47, 28, 106], and agent [104, 102].

Accelerating Diffusion Language Models.
dLLMs are viewed as promising due to their potential for low-cost inference, yet their efficiency remains largely underexplored. Existing efforts improve efficiency from several perspectives. Some methods reduce the cost of each decoding step through techniques including KV caching [53, 49, 84, 35, 45], token dropping [15, 36, 72, 85], and sparse attention [79, 19]. Others design more effective decoding strategies [37, 81, 41, 27, 8, 32, 34, 89, 50, 12, 77, 61, 22] to improve generation efficiency. A separate line of work [73, 62, 101, 7, 14, 33] learns better decoding trajectories so that fewer decoding steps are required. dParallel [16] employs certainty-forcing distillation to accelerate confidence convergence and enable higher parallel decoding. Other methods [83, 17, 78, 3, 52, 75, 46, 23] interpolate between diffusion and autoregressive language models to better balance speed and accuracy. [9, 76, 100] implement uniform training, which trains the model to recover clean tokens from random noisy tokens, thereby enabling token correction during generation. SM [30] and EvoToken [107] introduce soft embeddings into the decoding process, but neither method translates this design into improved decoding efficiency. Further efforts [88] leverage compression techniques to construct lightweight dLLMs.

## 7 Conclusion

In this paper, we present DMax, a novel paradigm for efficient diffusion language models that mitigates error accumulation for parallel decoding. DMax enables aggressive decoding parallelism while preserving the accuracy of the original model. We introduce two key components of our approach, namely On-Policy Uniform Training and Soft Parallel Decoding, and demonstrate their effectiveness through extensive experiments on diverse benchmarks. Our results establish a strong new baseline for parallel decoding in dLLMs and suggest a promising new direction for dLLMs.

## References

- [1]
J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. (2023)
Gpt-4 technical report.
arXiv preprint arXiv:2303.08774.
Cited by: [§1](#S1.p1.1).
- [2]
W. U. Ahmad, A. Ficek, M. Samadi, J. Huang, V. Noroozi, S. Majumdar, and B. Ginsburg (2025)
Opencodeinstruct: a large-scale instruction tuning dataset for code llms.
arXiv preprint arXiv:2504.04030.
Cited by: [§4.1](#S4.SS1.p2.1).
- [3]
M. Arriola, A. Gokaslan, J. T. Chiu, Z. Yang, Z. Qi, J. Han, S. S. Sahoo, and V. Kuleshov (2025)
Block diffusion: interpolating between autoregressive and diffusion language models.
arXiv preprint arXiv:2503.09573.
Cited by: [§6](#S6.p2.1).
- [4]
J. Austin, D. D. Johnson, J. Ho, D. Tarlow, and R. Van Den Berg (2021)
Structured denoising diffusion models in discrete state-spaces.
Advances in neural information processing systems 34, pp. 17981–17993.
Cited by: [§2](#S2.p2.5),
[§6](#S6.p1.1).
- [5]
J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, et al. (2021)
Program synthesis with large language models.
arXiv preprint arXiv:2108.07732.
Cited by: [§1](#S1.p5.1),
[§4.1](#S4.SS1.p3.1).
- [6]
J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, et al. (2023)
Qwen technical report.
arXiv preprint arXiv:2309.16609.
Cited by: [§1](#S1.p1.1).
- [7]
W. Bao, Z. Chen, D. Xu, and Y. Shang (2025)
Learning to parallel: accelerating diffusion large language models via adaptive parallel decoding.
In The Fourteenth International Conference on Learning Representations,
Cited by: [§6](#S6.p2.1).
- [8]
H. Ben-Hamu, I. Gat, D. Severo, N. Nolte, and B. Karrer (2025)
Accelerated sampling from masked diffusion models via entropy bounded unmasking.
arXiv preprint arXiv:2505.24857.
Cited by: [§1](#S1.p2.1),
[§6](#S6.p2.1).
- [9]
T. Bie, M. Cao, X. Cao, B. Chen, F. Chen, K. Chen, L. Du, D. Feng, H. Feng, M. Gong, et al. (2026)
Llada2. 1: speeding up text diffusion via token editing.
arXiv preprint arXiv:2602.08676.
Cited by: [§6](#S6.p2.1).
- [10]
T. Bie, M. Cao, K. Chen, L. Du, M. Gong, Z. Gong, Y. Gu, J. Hu, Z. Huang, Z. Lan, et al. (2025)
Llada2. 0: scaling up diffusion language models to 100b.
arXiv preprint arXiv:2512.15745.
Cited by: [§1](#S1.p2.1),
[§1](#S1.p5.1),
[§4.1](#S4.SS1.p1.2),
[§4.1](#S4.SS1.p4.1),
[§6](#S6.p1.1).
- [11]
T. Cai, Y. Li, Z. Geng, H. Peng, J. D. Lee, D. Chen, and T. Dao (2024)
Medusa: simple llm inference acceleration framework with multiple decoding heads.
arXiv preprint arXiv:2401.10774.
Cited by: [§1](#S1.p3.1).
- [12]
J. Chen, Y. Liang, and Z. Liu (2026)
DFlash: block diffusion for flash speculative decoding.
arXiv preprint arXiv:2602.06036.
Cited by: [§6](#S6.p2.1).
- [13]
M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba (2021)
Evaluating large language models trained on code.
External Links: 2107.03374
Cited by: [§4.1](#S4.SS1.p3.1).
- [14]
S. Chen, J. Jiao, L. J. Ratliff, and B. Zhu (2025)
DUltra: ultra-fast diffusion language models via reinforcement learning.
arXiv preprint arXiv:2512.21446.
Cited by: [§6](#S6.p2.1).
- [15]
X. Chen, S. Huang, C. Guo, C. Wei, Y. He, J. Zhang, H. Li, Y. Chen, et al. (2025)
DPad: efficient diffusion language models with suffix dropout.
arXiv preprint arXiv:2508.14148.
Cited by: [§6](#S6.p2.1).
- [16]
Z. Chen, G. Fang, X. Ma, R. Yu, and X. Wang (2025)
Dparallel: learnable parallel decoding for dllms.
arXiv preprint arXiv:2509.26488.
Cited by: [§1](#S1.p2.1),
[§4.1](#S4.SS1.p4.1),
[§6](#S6.p2.1).
- [17]
S. Cheng, Y. Bian, D. Liu, L. Zhang, Q. Yao, Z. Tian, W. Wang, Q. Guo, K. Chen, B. Qi, et al. (2025)
Sdar: a synergistic diffusion-autoregression paradigm for scalable sequence generation.
arXiv preprint arXiv:2510.06303.
Cited by: [§1](#S1.p2.1),
[§6](#S6.p2.1).
- [18]
S. Cheng, Y. Jiang, Z. Zhou, D. Liu, W. Tao, L. Zhang, B. Qi, and B. Zhou (2025)
Sdar-vl: stable and efficient block-wise diffusion for vision-language understanding.
arXiv preprint arXiv:2512.14068.
Cited by: [§6](#S6.p1.1).
- [19]
A. Christoforos and C. Davis (2025)
MoE-diffuseq: enhancing long-document diffusion models with sparse attention and mixture of experts.
arXiv preprint arXiv:2512.20604.
Cited by: [§6](#S6.p2.1).
- [20]
K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. (2021)
Training verifiers to solve math word problems.
arXiv preprint arXiv:2110.14168.
Cited by: [§1](#S1.p5.1),
[§4.1](#S4.SS1.p2.1),
[§4.1](#S4.SS1.p3.1).
- [21]
C. Fan, W. Heng, B. Li, S. Liu, Y. Song, J. Su, X. Qu, K. Shen, and W. Wei (2026)
Stable-diffcoder: pushing the frontier of code diffusion large language model.
arXiv preprint arXiv:2601.15892.
Cited by: [§6](#S6.p1.1).
- [22]
S. Feng, Z. Chen, X. Ma, G. Fang, and X. Wang (2026)
DVoting: fast voting for dllms.
arXiv preprint arXiv:2602.12153.
Cited by: [§6](#S6.p2.1).
- [23]
Y. Fu, L. Whalen, Z. Ye, X. Dong, S. Diao, J. Liu, C. Wu, H. Zhang, E. Xie, S. Han, et al. (2025)
Efficient-dlm: from autoregressive to diffusion language models, and beyond in speed.
arXiv preprint arXiv:2512.14067.
Cited by: [§6](#S6.p2.1).
- [24]
S. Gong, R. Zhang, H. Zheng, J. Gu, N. Jaitly, L. Kong, and Y. Zhang (2025)
Diffucoder: understanding and improving masked diffusion models for code generation.
arXiv preprint arXiv:2506.20639.
Cited by: [§6](#S6.p1.1).
- [25]
A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al. (2024)
The llama 3 herd of models.
arXiv preprint arXiv:2407.21783.
Cited by: [§1](#S1.p1.1).
- [26]
E. Guha, R. Marten, S. Keh, N. Raoof, G. Smyrnis, H. Bansal, M. Nezhurina, J. Mercat, T. Vu, Z. Sprague, et al. (2025)
Openthoughts: data recipes for reasoning models.
arXiv preprint arXiv:2506.04178.
Cited by: [§4.1](#S4.SS1.p2.1).
- [27]
D. Gwak, M. Jung, J. Park, M. Park, C. Park, J. Hyung, and J. Choo (2025)
Reward-weighted sampling: enhancing non-autoregressive characteristics in masked diffusion llms.
arXiv preprint arXiv:2509.00707.
Cited by: [§1](#S1.p2.1),
[§6](#S6.p2.1).
- [28]
G. He, S. Nie, F. Zhu, Y. Zhao, T. Bai, R. Yan, J. Fu, C. Li, and B. Yuan (2025)
Ultrallada: scaling the context length to 128k for diffusion large language models.
arXiv preprint arXiv:2510.10481.
Cited by: [§6](#S6.p1.1).
- [29]
D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt (2021)
Measuring mathematical problem solving with the math dataset.
arXiv preprint arXiv:2103.03874.
Cited by: [§4.1](#S4.SS1.p3.1).
- [30]
M. Hersche, S. Moor-Smith, T. Hofmann, and A. Rahimi (2025)
Soft-masked diffusion language models.
arXiv preprint arXiv:2510.17206.
Cited by: [§6](#S6.p2.1).
- [31]
J. Ho, A. Jain, and P. Abbeel (2020)
Denoising diffusion probabilistic models.
Advances in neural information processing systems 33, pp. 6840–6851.
Cited by: [§6](#S6.p1.1).
- [32]
F. Hong, G. Yu, Y. Ye, H. Huang, H. Zheng, Y. Zhang, Y. Wang, and J. Yao (2025)
Wide-in, narrow-out: revokable decoding for efficient and effective dllms.
arXiv preprint arXiv:2507.18578.
Cited by: [§1](#S1.p2.1),
[§6](#S6.p2.1).
- [33]
Y. Hu, Y. Jin, P. Liu, K. Yu, and Z. Deng (2026)
LightningRL: breaking the accuracy-parallelism trade-off of block-wise dllms via reinforcement learning.
arXiv preprint arXiv:2603.13319.
Cited by: [§6](#S6.p2.1).
- [34]
Y. Hu, H. Singh, M. Maheswaran, H. Xi, C. Hooper, J. Zhang, A. Tomar, M. W. Mahoney, S. Min, M. Farajtabar, et al. (2026)
Residual context diffusion language models.
arXiv preprint arXiv:2601.22954.
Cited by: [§1](#S1.p2.1),
[§6](#S6.p2.1).
- [35]
Z. Hu, J. Meng, Y. Akhauri, M. S. Abdelfattah, J. Seo, Z. Zhang, and U. Gupta (2025)
Accelerating diffusion language model inference via efficient kv caching and guided diffusion.
arXiv preprint arXiv:2505.21467.
Cited by: [§6](#S6.p2.1).
- [36]
J. Huang, Y. Zhang, Y. Yang, B. Huang, B. Qi, D. Liu, and L. Zhang (2025)
Mask tokens as prophet: fine-grained cache eviction for efficient dllm inference.
arXiv preprint arXiv:2510.09309.
Cited by: [§6](#S6.p2.1).
- [37]
D. Israel, G. V. d. Broeck, and A. Grover (2025)
Accelerating diffusion llms via adaptive parallel decoding.
arXiv preprint arXiv:2506.00413.
Cited by: [§1](#S1.p2.1),
[§6](#S6.p2.1).
- [38]
M. Kim, C. Xu, C. Hooper, H. Singh, B. Athiwaratkun, C. Zhang, K. Keutzer, and A. Gholami (2025)
CDLM: consistency diffusion language models for faster sampling.
arXiv preprint arXiv:2511.19269.
Cited by: [§1](#S1.p2.1).
- [39]
Y. Leviathan, M. Kalman, and Y. Matias (2023)
Fast inference from transformers via speculative decoding.
In International Conference on Machine Learning,
pp. 19274–19286.
Cited by: [§1](#S1.p3.1).
- [40]
J. Li, E. Beeching, L. Tunstall, B. Lipkin, R. Soletskyi, S. Huang, K. Rasul, L. Yu, A. Q. Jiang, Z. Shen, et al. (2024)
Numinamath: the largest public dataset in ai4maths with 860k pairs of competition math problems and solutions.
Hugging Face repository 13 (9), pp. 9.
Cited by: [§4.1](#S4.SS1.p2.1).
- [41]
J. Li, X. Dong, Y. Zang, Y. Cao, J. Wang, and D. Lin (2025)
Beyond fixed: variable-length denoising for diffusion large language models.
arXiv e-prints, pp. arXiv–2508.
Cited by: [§1](#S1.p2.1),
[§6](#S6.p2.1).
- [42]
T. Li, M. Chen, B. Guo, and Z. Shen (2025)
A survey on diffusion language models.
arXiv preprint arXiv:2508.10875.
Cited by: [§1](#S1.p1.1).
- [43]
Y. Li, F. Wei, C. Zhang, and H. Zhang (2024)
Eagle: speculative sampling requires rethinking feature uncertainty.
arXiv preprint arXiv:2401.15077.
Cited by: [§1](#S1.p3.1).
- [44]
H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe (2023)
Let’s verify step by step.
In The Twelfth International Conference on Learning Representations,
Cited by: [§4.1](#S4.SS1.p2.1),
[§4.1](#S4.SS1.p3.1).
- [45]
A. Liu, M. He, S. Zeng, S. Zhang, L. Zhang, C. Wu, W. Jia, Y. Liu, X. Zhou, and J. Zhou (2025)
Wedlm: reconciling diffusion language models with standard causal attention for fast inference.
arXiv preprint arXiv:2512.22737.
Cited by: [§6](#S6.p2.1).
- [46]
J. Liu, X. Dong, Z. Ye, R. Mehta, Y. Fu, V. Singh, J. Kautz, C. Zhang, and P. Molchanov (2025)
Tidar: think in diffusion, talk in autoregression.
arXiv preprint arXiv:2511.08923.
Cited by: [§6](#S6.p2.1).
- [47]
X. Liu, Y. Song, Z. Liu, Z. Huang, Q. Guo, Z. He, and X. Qiu (2026)
Longllada: unlocking long context capabilities in diffusion llms.
In Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 40, pp. 32186–32194.
Cited by: [§6](#S6.p1.1).
- [48]
Y. Liu, P. Ding, T. Jiang, X. Wang, W. Song, M. Lin, H. Zhao, H. Zhang, Z. Zhuang, W. Zhao, et al. (2026)
MMaDA-vla: large diffusion vision-language-action model with unified multi-modal instruction and generation.
arXiv preprint arXiv:2603.25406.
Cited by: [§6](#S6.p1.1).
- [49]
Z. Liu, Y. Yang, Y. Zhang, J. Chen, C. Zou, Q. Wei, S. Wang, and L. Zhang (2025)
Dllm-cache: accelerating diffusion large language models with adaptive caching.
arXiv preprint arXiv:2506.06295.
Cited by: [§6](#S6.p2.1).
- [50]
L. Long, Y. Huang, S. Bai, R. Gong, J. Zhang, A. Zhou, and J. Yang (2026)
Focus-dllm: accelerating long-context diffusion llm inference via confidence-guided context focusing.
arXiv preprint arXiv:2602.02159.
Cited by: [§6](#S6.p2.1).
- [51]
A. Lou, C. Meng, and S. Ermon (2023)
Discrete diffusion language modeling by estimating the ratios of the data distribution.
Cited by: [§2](#S2.p2.5),
[§6](#S6.p1.1).
- [52]
L. Ma, Y. Cui, K. Han, and Y. Wang (2026)
Diffusion in diffusion: breaking the autoregressive bottleneck in block diffusion models.
arXiv preprint arXiv:2601.13599.
Cited by: [§6](#S6.p2.1).
- [53]
X. Ma, R. Yu, G. Fang, and X. Wang (2025)
Dkv-cache: the cache for diffusion language models.
arXiv preprint arXiv:2505.15781.
Cited by: [§6](#S6.p2.1).
- [54]
Y. Ma, L. Du, L. Wei, K. Chen, Q. Xu, K. Wang, G. Feng, G. Lu, L. Liu, X. Qi, et al. (2025)
Dinfer: an efficient inference framework for diffusion language models.
arXiv preprint arXiv:2510.08666.
Cited by: [§4.1](#S4.SS1.p3.1).
- [55]
S. Miao, C. Liang, and K. Su (2021)
A diverse corpus for evaluating and developing english math word problem solvers.
External Links: 2106.15772
Cited by: [§4.1](#S4.SS1.p3.1).
- [56]
J. Ni, Q. Liu, L. Dou, C. Du, Z. Wang, H. Yan, T. Pang, and M. Q. Shieh (2025)
Diffusion language models are super data learners.
arXiv preprint arXiv:2511.03276.
Cited by: [§1](#S1.p1.1).
- [57]
Z. Ni, S. Wang, Y. Yue, T. Yu, W. Zhao, Y. Hua, T. Chen, J. Song, C. Yu, B. Zheng, et al. (2026)
The flexibility trap: why arbitrary order limits reasoning potential in diffusion language models.
arXiv preprint arXiv:2601.15165.
Cited by: [§6](#S6.p1.1).
- [58]
S. Nie, F. Zhu, Z. You, X. Zhang, J. Ou, J. Hu, J. Zhou, Y. Lin, J. Wen, and C. Li (2025)
Large language diffusion models.
arXiv preprint arXiv:2502.09992.
Cited by: [§1](#S1.p2.1),
[§6](#S6.p1.1).
- [59]
L. Pan, S. Tao, Y. Zhai, Z. Fu, L. Fang, M. He, L. Zhang, Z. Liu, B. Ding, A. Liu, et al. (2025)
D-treerpo: towards more reliable policy optimization for diffusion language models.
arXiv preprint arXiv:2512.09675.
Cited by: [§6](#S6.p1.1).
- [60]
D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach (2023)
Sdxl: improving latent diffusion models for high-resolution image synthesis.
arXiv preprint arXiv:2307.01952.
Cited by: [§6](#S6.p1.1).
- [61]
X. Qi, L. Du, X. Zhang, L. Wei, T. Jin, and D. Zheng
Hierarchy decoding: a training-free parallel decoding strategy for diffusion large language models.
In The Fourteenth International Conference on Learning Representations,
Cited by: [§4.1](#S4.SS1.p4.1),
[§6](#S6.p2.1).
- [62]
Y. Qian, J. Su, L. Hu, P. Zhang, Z. Deng, P. Zhao, and H. Zhang (2026)
D3LLM: ultra-fast diffusion llm using pseudo-trajectory distillation.
arXiv preprint arXiv:2601.07568.
Cited by: [§1](#S1.p2.1),
[§4.1](#S4.SS1.p3.1),
[§6](#S6.p2.1).
- [63]
K. Rojas, J. Lin, K. Rasul, A. Schneider, Y. Nevmyvaka, M. Tao, and W. Deng (2025)
Improving reasoning for diffusion language models via group diffusion policy optimization.
arXiv preprint arXiv:2510.08554.
Cited by: [§6](#S6.p1.1).
- [64]
R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer (2022)
High-resolution image synthesis with latent diffusion models.
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
pp. 10684–10695.
Cited by: [§6](#S6.p1.1).
- [65]
N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman (2023)
Dreambooth: fine tuning text-to-image diffusion models for subject-driven generation.
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
pp. 22500–22510.
Cited by: [§6](#S6.p1.1).
- [66]
S. Sahoo, M. Arriola, Y. Schiff, A. Gokaslan, E. Marroquin, J. Chiu, A. Rush, and V. Kuleshov (2024)
Simple and effective masked diffusion language models.
Advances in Neural Information Processing Systems 37, pp. 130136–130184.
Cited by: [§2](#S2.p2.5),
[§6](#S6.p1.1).
- [67]
S. S. Sahoo, J. Deschenaux, A. Gokaslan, G. Wang, J. Chiu, and V. Kuleshov (2025)
The diffusion duality.
arXiv preprint arXiv:2506.10892.
Cited by: [§2](#S2.p3.1).
- [68]
Y. Schiff, S. S. Sahoo, H. Phung, G. Wang, A. Rush, V. Kuleshov, H. Dalla-Torre, S. Boshar, B. P. de Almeida, and T. Pierrot (2025)
Simple guidance mechanisms for discrete diffusion models.
In … International Conference on Learning Representations,
Vol. 2025, pp. 44153.
Cited by: [§2](#S2.p3.1).
- [69]
S. Sekhar Sahoo, J. Lemercier, Z. Yang, J. Deschenaux, J. Liu, J. Thickstun, and A. Jukic (2026)
Scaling beyond masked diffusion language models.
arXiv e-prints, pp. arXiv–2602.
Cited by: [§2](#S2.p3.1).
- [70]
J. Shi, K. Han, Z. Wang, A. Doucet, and M. Titsias (2024)
Simplified and generalized masked diffusion for discrete data.
Advances in neural information processing systems 37, pp. 103131–103167.
Cited by: [§2](#S2.p2.5),
[§6](#S6.p1.1).
- [71]
J. Song, C. Meng, and S. Ermon (2020)
Denoising diffusion implicit models.
arXiv preprint arXiv:2010.02502.
Cited by: [§6](#S6.p1.1).
- [72]
Y. Song, X. Liu, R. Li, Z. Liu, Z. Huang, Q. Guo, Z. He, and X. Qiu (2026)
Sparse-dllm: accelerating diffusion llms with dynamic cache eviction.
In Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 40, pp. 33038–33046.
Cited by: [§6](#S6.p2.1).
- [73]
Y. Song, Z. Zhang, C. Luo, P. Gao, F. Xia, H. Luo, Z. Li, Y. Yang, H. Yu, X. Qu, et al. (2025)
Seed diffusion: a large-scale diffusion language model with high-speed inference.
arXiv preprint arXiv:2508.02193.
Cited by: [§6](#S6.p2.1).
- [74]
X. Tang, R. Dolga, S. Yoon, and I. Bogunovic (2025)
Wd1: weighted policy optimization for reasoning in diffusion language models.
arXiv preprint arXiv:2507.08838.
Cited by: [§6](#S6.p1.1).
- [75]
Y. Tian, Y. Liang, S. Zhang, Y. Shu, G. Yang, W. He, S. Fang, T. Guo, K. Han, C. Xu, et al. (2025)
From next-token to next-block: a principled adaptation path for diffusion llms.
arXiv preprint arXiv:2512.06776.
Cited by: [§6](#S6.p2.1).
- [76]
D. Von Rütte, J. Fluri, Y. Ding, A. Orvieto, B. Schölkopf, and T. Hofmann (2025)
Generalized interpolating discrete diffusion.
arXiv preprint arXiv:2503.04482.
Cited by: [§6](#S6.p2.1).
- [77]
K. Wang, Z. Jiang, H. Feng, W. Zhao, L. Liu, J. Li, Z. Lan, and W. Lin (2025)
Creditdecoding: accelerating parallel decoding in diffusion large language models with trace credits.
arXiv preprint arXiv:2510.06133.
Cited by: [§6](#S6.p2.1).
- [78]
X. Wang, C. Xu, Y. Jin, J. Jin, H. Zhang, and Z. Deng (2025)
Diffusion llms can do faster-than-ar inference via discrete diffusion forcing.
arXiv preprint arXiv:2508.09192.
Cited by: [§6](#S6.p2.1).
- [79]
Z. Wang, G. Fang, X. Ma, X. Yang, and X. Wang (2026)
SparseD: sparse attention for diffusion language models.
In The Fourteenth International Conference on Learning Representations,
External Links: [Link](https://openreview.net/forum?id=dwbrZtYP04)
Cited by: [§6](#S6.p2.1).
- [80]
J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. (2022)
Chain-of-thought prompting elicits reasoning in large language models.
Advances in neural information processing systems 35, pp. 24824–24837.
Cited by: [§4.1](#S4.SS1.p3.1).
- [81]
Q. Wei, Y. Zhang, Z. Liu, D. Liu, and L. Zhang (2025)
Accelerating diffusion large language models with slowfast: the three golden principles.
arXiv preprint arXiv:2506.10848.
Cited by: [§1](#S1.p2.1),
[§6](#S6.p2.1).
- [82]
Y. Wen, H. Li, K. Gu, Y. Zhao, T. Wang, and X. Sun (2025)
Llada-vla: vision language diffusion action models.
arXiv preprint arXiv:2509.06932.
Cited by: [§6](#S6.p1.1).
- [83]
C. Wu, H. Zhang, S. Xue, S. Diao, Y. Fu, Z. Liu, P. Molchanov, P. Luo, S. Han, and E. Xie (2025)
Fast-dllm v2: efficient block-diffusion llm.
arXiv preprint arXiv:2509.26328.
Cited by: [§6](#S6.p2.1).
- [84]
C. Wu, H. Zhang, S. Xue, Z. Liu, S. Diao, L. Zhu, P. Luo, S. Han, and E. Xie (2025)
Fast-dllm: training-free acceleration of diffusion llm by enabling kv cache and parallel decoding.
arXiv preprint arXiv:2505.22618.
Cited by: [§6](#S6.p2.1).
- [85]
Z. Xiao, Z. Hao, J. Guo, Y. Luo, J. Liu, J. Xu, and H. Hu (2026)
Streaming-dllm: accelerating diffusion llms via suffix pruning and dynamic decoding.
arXiv e-prints, pp. arXiv–2601.
Cited by: [§6](#S6.p2.1).
- [86]
S. Xie, L. Kong, X. Song, X. Dong, G. Chen, E. P. Xing, and K. Zhang (2025)
Step-aware policy optimization for reasoning in diffusion large language models.
arXiv preprint arXiv:2510.01544.
Cited by: [§6](#S6.p1.1).
- [87]
Z. Xie, J. Ye, L. Zheng, J. Gao, J. Dong, Z. Wu, X. Zhao, S. Gong, X. Jiang, Z. Li, et al. (2025)
Dream-coder 7b: an open diffusion language model for code.
arXiv preprint arXiv:2509.01142.
Cited by: [§6](#S6.p1.1).
- [88]
C. Xu and D. Yang (2025)
Dllmquant: quantizing diffusion-based large language models.
arXiv preprint arXiv:2508.14090.
Cited by: [§6](#S6.p2.1).
- [89]
C. Xu, Y. Jin, J. Li, Y. Tu, G. Long, D. Tu, M. Song, H. Si, T. Hou, J. Yan, et al. (2025)
Lopa: scaling dllm inference via lookahead parallel decoding.
arXiv preprint arXiv:2512.16229.
Cited by: [§1](#S1.p2.1),
[§6](#S6.p2.1).
- [90]
L. Yang, Y. Tian, B. Li, X. Zhang, K. Shen, Y. Tong, and M. Wang (2025)
Mmada: multimodal large diffusion language models.
arXiv preprint arXiv:2505.15809.
Cited by: [§1](#S1.p2.1),
[§6](#S6.p1.1).
- [91]
J. Ye, S. Gong, J. Gao, J. Fan, S. Wu, W. Bi, H. Bai, L. Shang, and L. Kong (2025)
Dream-vl & dream-vla: open vision-language and vision-language-action models with diffusion language model backbone.
arXiv preprint arXiv:2512.22615.
Cited by: [§6](#S6.p1.1).
- [92]
J. Ye, Z. Xie, L. Zheng, J. Gao, Z. Wu, X. Jiang, Z. Li, and L. Kong (2025)
Dream 7b: diffusion large language models.
arXiv preprint arXiv:2508.15487.
Cited by: [§1](#S1.p2.1),
[§6](#S6.p1.1).
- [93]
Q. Yi, X. Chen, C. Zhang, Z. Zhou, L. Zhu, and X. Kong (2024)
Diffusion models in text generation: a survey.
PeerJ Computer Science 10, pp. e1905.
Cited by: [§1](#S1.p1.1).
- [94]
Z. You, S. Nie, X. Zhang, J. Hu, J. Zhou, Z. Lu, J. Wen, and C. Li (2025)
Llada-v: large language diffusion models with visual instruction tuning.
arXiv preprint arXiv:2505.16933.
Cited by: [§6](#S6.p1.1).
- [95]
R. Yu, Q. Li, and X. Wang (2025)
Discrete diffusion in large language and multimodal models: a survey.
arXiv preprint arXiv:2506.13759.
Cited by: [§1](#S1.p1.1).
- [96]
R. Yu, X. Ma, and X. Wang (2025)
Dimple: discrete diffusion multimodal large language model with parallel decoding.
arXiv preprint arXiv:2505.16990.
Cited by: [§6](#S6.p1.1).
- [97]
L. Zeng, J. Yao, B. Liao, H. Tao, W. Liu, and X. Wang (2025)
DiffusionVL: translating any autoregressive models into diffusion vision language models.
arXiv preprint arXiv:2512.15713.
Cited by: [§6](#S6.p1.1).
- [98]
L. Zhang, L. Fang, C. Duan, M. He, L. Pan, P. Xiao, S. Huang, Y. Zhai, X. Hu, P. S. Yu, et al. (2025)
A survey on parallel text generation: from parallel decoding to diffusion language models.
arXiv preprint arXiv:2508.08712.
Cited by: [§1](#S1.p1.1).
- [99]
L. Zhang, A. Rao, and M. Agrawala (2023)
Adding conditional control to text-to-image diffusion models.
In Proceedings of the IEEE/CVF international conference on computer vision,
pp. 3836–3847.
Cited by: [§6](#S6.p1.1).
- [100]
S. Zhang, F. Z. Peng, Y. Zhang, J. Pan, and G. G. Chrysos (2025)
Corrective diffusion language models.
arXiv preprint arXiv:2512.15596.
Cited by: [§6](#S6.p2.1).
- [101]
T. Zhang, X. Zhang, L. Han, H. Shi, X. He, Z. Li, H. Wang, K. Xu, A. Srivastava, V. Pavlovic, et al. (2026)
T3D: few-step diffusion language models via trajectory self-distillation with direct discriminative optimization.
arXiv preprint arXiv:2602.12262.
Cited by: [§1](#S1.p2.1),
[§6](#S6.p2.1).
- [102]
J. Zhao, S. Xu, Z. Sun, F. Zhu, J. Ou, Y. Shi, C. Li, X. Zhang, and J. Xu (2026)
DLLM-searcher: adapting diffusion large language model for search agents.
arXiv preprint arXiv:2602.07035.
Cited by: [§6](#S6.p1.1).
- [103]
S. Zhao, D. Gupta, Q. Zheng, and A. Grover (2025)
D1: scaling reasoning in diffusion large language models via reinforcement learning.
arXiv preprint arXiv:2504.12216.
Cited by: [§6](#S6.p1.1).
- [104]
H. Zhen, W. Lin, R. Liu, K. Han, Y. Li, Y. Tian, H. Chen, X. Li, X. Li, C. Chen, et al. (2026)
DLLM agent: see farther, run faster.
arXiv preprint arXiv:2602.07451.
Cited by: [§6](#S6.p1.1).
- [105]
K. Zheng, Y. Chen, H. Mao, M. Liu, J. Zhu, and Q. Zhang (2024)
Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling.
arXiv preprint arXiv:2409.02908.
Cited by: [§2](#S2.p2.5),
[§6](#S6.p1.1).
- [106]
L. Zheng, B. Shi, Y. Hu, J. Zhang, R. Li, S. Chen, W. Li, and K. Li (2026)
Mosaic: unlocking long-context inference for diffusion llms via global memory planning and dynamic peak taming.
arXiv preprint arXiv:2601.06562.
Cited by: [§6](#S6.p1.1).
- [107]
L. Zhong, L. Wu, B. Fang, T. Feng, C. Jing, W. Wang, J. Zhang, H. Chen, and C. Shen (2026)
Beyond hard masks: progressive token evolution for diffusion language models.
arXiv preprint arXiv:2601.07351.
Cited by: [§6](#S6.p2.1).
- [108]
Z. Zhou, L. Chen, H. Tong, and D. Song (2026)
Dllm: simple diffusion language modeling.
arXiv preprint arXiv:2602.22661.
Cited by: [§1](#S1.p1.1).
- [109]
F. Zhu, R. Wang, S. Nie, X. Zhang, C. Wu, J. Hu, J. Zhou, J. Chen, Y. Lin, J. Wen, et al. (2025)
Llada 1.5: variance-reduced preference optimization for large language diffusion models.
arXiv preprint arXiv:2505.19223.
Cited by: [§1](#S1.p2.1),
[§6](#S6.p1.1).
- [110]
F. Zhu, Z. You, Y. Xing, Z. Huang, L. Liu, Y. Zhuang, G. Lu, K. Wang, X. Wang, L. Wei, et al. (2025)
Llada-moe: a sparse moe diffusion language model.
arXiv preprint arXiv:2509.24389.
Cited by: [§6](#S6.p1.1).
