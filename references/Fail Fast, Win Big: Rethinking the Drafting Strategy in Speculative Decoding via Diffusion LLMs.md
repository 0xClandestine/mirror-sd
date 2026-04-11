Sections:
Abstract
1 Introduction
2 Background
3 Motivation
    3.1 dLLMs’ Concavity of Accuracy Improvements
    3.2 The Varying Decoding Difficulty within Sequences
4 Method
    4.1 “Fail Fast”: Embracing Error-Proneness for Faster Speculation
    4.2 “Win Big”: Dynamic Expansion of Speculation Length for Faster Verification
        Confidence as a difficulty signal.
        Dynamic speculation length expansion logic.
    4.3 Policy Design vs. Training-Based Approaches
5 Evaluation
    5.1 Setup
    5.2 End-to-End Results
6 Conclusion
References
Appendix A List of abbreviations and notations
Appendix B Detailed Performance Comparison
Appendix C Hyperparameter Sensitivity Analysis of FailFast
Appendix D Best-Effort Baselines
    D.1 Fast-dLLM Drafter
    D.2 SuffixDecoding
    D.3 EAGLE-3
Appendix E (Failed) Explorations of Alternative Design Choices and Limitations
Appendix F Visualization of Example Trajectory

Files Content:

## Contents
- 1 Introduction
- 2 Background
- 3 Motivation
  - 3.1 dLLMs’ Concavity of Accuracy Improvements
  - 3.2 The Varying Decoding Difficulty within Sequences
- 4 Method
  - 4.1 “Fail Fast”: Embracing Error-Proneness for Faster Speculation
  - 4.2 “Win Big”: Dynamic Expansion of Speculation Length for Faster Verification
    - Confidence as a difficulty signal.
    - Dynamic speculation length expansion logic.
  - 4.3 Policy Design vs. Training-Based Approaches
- 5 Evaluation
  - 5.1 Setup
  - 5.2 End-to-End Results
- 6 Conclusion
- References
- Appendix A List of abbreviations and notations
- Appendix B Detailed Performance Comparison
- Appendix C Hyperparameter Sensitivity Analysis of FailFast
- Appendix D Best-Effort Baselines
  - D.1 Fast-dLLM Drafter
  - D.2 SuffixDecoding
  - D.3 EAGLE-3
- Appendix E (Failed) Explorations of Alternative Design Choices and Limitations
- Appendix F Visualization of Example Trajectory

## Abstract

Abstract Diffusion Large Language Models (dLLMs) offer fast, parallel token generation, but their standalone use is plagued by an inherent efficiency-quality tradeoff.
We show that, if carefully applied, the attributes of dLLMs can actually be a strength for drafters in speculative decoding with autoregressive (AR) verifiers.
Our core insight is that dLLM’s speed from parallel decoding drastically lowers the risk of costly rejections, providing a practical mechanism to effectively realize the (elusive) lengthy drafts that lead to large speedups with speculative decoding.
We present FailFast , a dLLM-based speculative decoding framework that realizes this approach by dynamically adapting its speculation length.
It “fails fast” by spending minimal compute in hard-to-speculate regions to shrink speculation latency and “wins big” by aggressively extending draft lengths in easier regions to reduce verification latency (in many cases, speculating and accepting 70 tokens at a time!).
Without any fine-tuning, FailFast delivers lossless acceleration of AR LLMs and achieves up to 4.9 × \times speedup over vanilla decoding, 1.7 × \times over the best naive dLLM drafter, and 1.7 × \times over EAGLE-3 across diverse models and workloads.
We open-source FailFast at https://github.com/ruipeterpan/failfast .

## 1 Introduction

Figure: Figure 1: Intuition behind the behavior of FailFast and other baseline drafters in each round of speculation. The AR drafter takes 10 forward passes to speculate 10 tokens, whereas the naive dLLM drafter employs confidence-aware parallel decoding, taking $<10$ forward passes to reach a similar quality. In contrast, FailFast spends minimal compute and dynamically determines how many tokens to propose based on the confidence of speculated tokens. In harder-to-speculate regions (top right), it adopts a shorter speculation length and minimizes the amount of compute to “fail fast”, further minimizing the speculation latency. In easier-to-speculate regions (bottom right), it aggressively extends the speculation length to “win big” and reduce the verification overhead. This example illustrates two extremes of FailFast; in practice, it dynamically navigates through a spectrum of speculation lengths based on decoding difficulty.
Refer to caption: https://arxiv.org/html/2512.20573/x1.png

A new wave of Diffusion Large Language Models (dLLMs) (Khanna et al., 2025; Song et al., 2025; gem, 2025; Bie et al., 2025; Wu et al., 2025b) has emerged as a compelling alternative to the standard autoregressive paradigm in large language models. Unlike autoregressive (AR) LLMs, which are constrained to generating tokens one by one from left to right, dLLMs possess the unique capability to predict and unmask multiple tokens at arbitrary positions simultaneously. Crucially, this decoding process is highly customizable: the model’s unmasking strategy determines exactly which and how many tokens are unmasked during each denoising step (a model forward pass). As such, dLLMs are highly attractive for low-latency inference.

Yet despite their speed, parallel generation imposes a fundamental limit on modeling accuracy. This limitation stems from the conditional independence assumption required for simultaneous sampling of multiple tokens; by treating tokens generated within the same step as independent of one another, the decoding process inevitably ignores crucial mutual dependencies (Wu et al., 2025b; Kang et al., 2025). Consequently, a direct tension emerges between efficiency and quality. Improving the generation speed (i.e., using fewer forward passes) necessitates unmasking a larger number of tokens per step, which exacerbates the risk of quality degradation. Conversely, maximizing quality forces the sampling procedure to adopt a strict left-to-right, one-token-per-step order that essentially falls back to the speed of autoregressive generation.

While existing work strives to alleviate the stark compute-accuracy tradeoff of dLLMs as standalone generators (Kang et al., 2025; Bie et al., 2025; Qian et al., 2026), this work instead focuses on motivating and realizing a scenario in which we argue that dLLMs are intrinsically beneficial: as draft models in speculative decoding (Leviathan et al., 2023) with autoregressive target models. Our proposal extends beyond a simple drop-in replacement of dLLMs as drafters in existing speculative decoding strategies to reap their latency benefits – indeed, we later show how this can forgo substantial benefits they bring. Instead, our approach is rooted in two key observations that challenge the status quo for both speculative decoding design and considerations around the limitations of dLLMs.

First, dLLMs can generate long drafts quickly. In AR LLM inference, the decoding latency scales with the number of output tokens (i.e., the number of model forward passes) (Yan et al., 2024; Agrawal et al., 2024), whereas dLLMs can unmask multiple tokens in each forward pass, so the latency is instead linear to the number of model forward passes. dLLMs’ ability to generate more tokens quickly motivates a rethink of the central challenge involved in designing effective speculative decoding strategies, i.e., balancing the fact that getting longer drafts accepted yields bigger wins (Liu et al., 2024b; Huang et al., 2024), but longer drafts come with higher risk due to each token’s probability of acceptance dropping exponentially.

Second, dLLMs exhibit distinct accuracy-compute concavity patterns at the sequence- and draft-level.
At the sequence level, the accuracy improvement of each additional forward pass yields diminishing returns. However, within each sequence, there are easier regions of tokens – e.g., simpler tasks like summarizing prior context (Pan et al., 2025a) – where minimal compute (i.e., a single forward pass) suffices for accurate generation.
In contrast, for harder regions where existing speculative decoding strategies typically struggle (e.g., difficult tasks like complex arithmetic), dLLMs require more compute to slowly refine their quality.
This observation drastically relaxes the issues with the latency-accuracy tension inherent to dLLMs: while additional forward passes are necessary for standalone generation, where each token matters for end-to-end quality, they are often unneeded for drafting in speculative decoding. Indeed, beyond the fact that all drafts are ultimately verified (making some errors tolerable), most wins typically come from the easier regions in a sequence (Yan et al., 2024) where initial dLLM draft accuracy routinely suffices.

Capitalizing on these observations, we present FailFast, our dLLM-based speculative decoding framework that revamps the design philosophy of speculative decoding frameworks. Its core operation is governed by two principles:

- 1.
Fail Fast: While most related work attempts to improve the quality of speculated tokens – e.g., through fine-tuning drafters or adopting an ensemble of small drafters – the quality of speculation is still fundamentally limited by the capacity of the small drafter model(s). We *fail fast* by deliberately spending minimal compute to speculate on tokens, further reducing drafting latency spent on tokens likely to be rejected anyway, while still (empirically) generating tokens in easy regions with high accuracy.
- 2.
Win Big: In easier regions where the speculated tokens are often accepted (even with minimal drafter compute), we *win big* by aggressively increasing the speculation length – in many cases, speculating 70 tokens in one round and getting all of them accepted – to avoid frequently going back and forth between the drafter and verifier, reducing the verification latency. Our signal for speculation easiness is simple and intuitive: the drafter’s confidence in its speculated tokens, which we find to be highly correlated with region hardness.

Across diverse models and workloads, FailFast achieves lossless acceleration of autoregressive target model generation, delivering speedups of up to 4.9$\times$ over vanilla decoding, 1.7$\times$ over a strong dLLM drafter baseline, and 1.7$\times$ over EAGLE-3.
We evaluate on diverse models and datasets and show that the intrinsic properties of dLLMs motivate fundamental changes to drafting strategies, which FailFast exploits to leverage their strengths as draft models.
By enabling off-the-shelf, already-aligned (Wu et al., 2025a; Bie et al., 2025; Gong et al., 2024; Zhou et al., 2025) dLLMs to act as effective drafters without any additional training or fine-tuning, FailFast fundamentally changes the drafting paradigm and naturally benefits from continued improvements in dLLM alignment (Sandler et al., 2025).

## 2 Background

Diffusion language models. Recent research has established Diffusion Language Models (dLLMs) as a promising alternative to standard autoregressive generation, with examples spanning both open-source (Nie et al., 2025; Bie et al., 2025; Ye et al., 2025) and closed-source models (Khanna et al., 2025; gem, 2025; Song et al., 2025). Architecturally, dLLMs retain the standard Transformer backbone, preserving access to familiar metrics such as log-probabilities and token confidence scores. However, they diverge fundamentally in their decoding mechanism: instead of generating tokens strictly left-to-right, dLLMs generate text through an iterative unmasking process. This decoding is typically semi-autoregressive (Arriola et al., 2025; gem, 2025; Wu et al., 2025a, b): the output sequence is divided into blocks, with inter-block attention remaining causal (later blocks attend to earlier blocks), but intra-block attention is bidirectional. (This semi-autoregressive decoding also enables optimizations like KV caching (Wu et al., 2025b).) Within each block, tokens are unmasked in a non-deterministic order determined by token confidence rather than position, allowing the model to unmask “easy” tokens before “hard” ones. To further improve efficiency, recent works have proposed acceleration techniques such as approximate KV caching (Wu et al., 2025b; Ma et al., 2025; Liu et al., 2025b), confidence-aware parallel decoding (Wu et al., 2025a, b), and self-speculative decoding (Agrawal et al., 2025).

Speculative decoding. Borrowing from classical computer architecture principles (Burton, 1985), speculative decoding has emerged as a standard technique to alleviate the memory-bound nature (Agrawal et al., 2024) of LLM inference (Stern et al., 2018; Leviathan et al., 2023; Yan et al., 2024; Liu et al., 2025a). The process operates on a propose-and-verify cycle: in each round, a lightweight drafter first generates a draft, which is subsequently validated by the target model. The speculation phase prioritizes efficiency, relying on methods such as a standalone draft model (Leviathan et al., 2023; Miao et al., 2024), a trainable module on top of the target model (Cai et al., 2024; Li et al., 2025), a tree-based token cache (Oliaro et al., 2024; Luo et al., 2024; Zhao et al., 2024), an n-gram lookup table (Fu et al., 2024), or a retrieval-based datastore (He et al., 2023). In the verification phase, the target model performs a parallel chunked-prefill over these candidates, which usually consists of either a single sequence of tokens as in (Leviathan et al., 2023) or tree-like structures to further boost the quality of speculation (Miao et al., 2024; Cai et al., 2024; Li et al., 2025; Chen et al., 2024), and accepts the longest valid prefix. Consequently, the speculation length $n$ is typically conservative (e.g., $n=$ 3-10) (Liu et al., 2024b; Fu et al., 2024) to maintain an optimal balance between the speculation overhead and the rate of token acceptance.

## 3 Motivation

### 3.1 dLLMs’ Concavity of Accuracy Improvements

Figure: Figure 2: [Qwen2.5-32B, MATH] dLLMs’ concavity of accuracy improvements when speculating 8 tokens per round. Doing 2.6$\times$ more drafter forward passes only increased the acceptance rate from 53.2% to 60.5%.
Refer to caption: https://arxiv.org/html/2512.20573/x2.png

The quality of a dLLM’s output is a function of compute. As non-autoregressive models, dLLMs generate multiple tokens in parallel during an iterative denoising process. In each forward pass, a select few masked tokens with higher confidence within the full sequence of regular and mask tokens are unmasked into regular tokens. As such, the computational cost of a dLLM is predominantly determined by the number of denoising steps (forward passes), not the number of tokens generated. By increasing the number of denoising steps, the model iteratively refines the output, trading latency for higher quality/fidelity.

dLLMs can outperform AR LLMs in generation speed while maintaining accuracy by spending fewer forward passes to generate the same number of output tokens (Wu et al., 2025b). However, naively deploying them as if they are equally-accurate but faster AR drafters in speculative decoding does not capitalize on their full latency benefit. Crucially, we find that dLLMs show concavity in accuracy wins as more compute is spent – improvements with each new forward pass have diminishing returns.
In Fig. [2](https://arxiv.org/html/2512.20573v3#S3.F2), we naively adopt a dLLM as the drafter and show that a higher compute budget leads to diminishing returns in overall acceptance rate.
A dLLM can, in theory, unmask a sequence of unlimited length using a single forward pass. Although the quality of such one-step generation is relatively low and more denoising steps improve the quality – and those denoising steps are crucial for quality if the dLLM is doing standalone generation – the role of draft models in speculative decoding is, by definition, to correctly decode the easier tokens, which is a nice match for the fast yet inaccurate one-step generation.

Figure: Figure 3: [Qwen2.5-32B, MATH] A semi-autoregressive dLLM that performs one-step generation achieves a higher theoretical speedup than that of an autoregressive drafter.(^2^22The sawtooth pattern arises because drafting 9–16 tokens costs two forward passes, while 17 tokens requires three (a 50% latency increase due to semi-autoregressive generation of 8-token blocks).)
Refer to caption: https://arxiv.org/html/2512.20573/x3.png

In Fig. [2](https://arxiv.org/html/2512.20573v3#footnote2), we analyze the theoretical speedup (Theorem 3.8 in (Leviathan et al., 2023)) of speculative decoding over vanilla decoding using different drafters of the same size.
The autoregressive drafter generates one token in each forward pass, incurring drafting latency linear in the draft length, whereas the semi-autoregressive dLLM drafter performs one-step generation and generates 8 tokens in each forward pass (Wu et al., 2025a), resulting in sublinear drafting latency.
Even though longer speculation lengths exponentially reduce the acceptance probabilities of the tokens later in the draft, the dLLM’s one-step generation compensates by dramatically lowering the average latency to generate each draft token, increasing the maximum speedup from 3.0$\times$ (at speculation length 8) to 5.2$\times$ (at length 16).

### 3.2 The Varying Decoding Difficulty within Sequences

Importantly, even though the concavity property manifests across entire sequences, within each sequence, there are easier regions and tokens where minimal compute suffices for high-quality generation.
Similar to prior work (Liu et al., 2024a; Huang et al., 2024; Pan et al., 2025a), we observe that in natural language generation, the difficulty of generation and speculation differs between regions of the output sequence. Easier regions often consist of syntactic copying, summarization of prior context and input prompt, formulaic enumerations, simple arithmetic, etc. In these regions of lower difficulty, a capable draft model can achieve a near-perfect acceptance rate.
Conversely, harder regions involve complex planning, multi-step reasoning, or knowledge retrieval where the draft model’s capacity is insufficient, leading to frequent rejections. In Fig. [4](https://arxiv.org/html/2512.20573v3#S3.F4), we pick five random queries from one of our evaluation datasets, run an AR draft model, and classify tokens in the output sequence into “easier” tokens (including both correct speculations and the “bonus tokens” when all drafted tokens are accepted) and “harder” tokens (rejected by the target model).
Further, in Fig. [5](https://arxiv.org/html/2512.20573v3#S3.F5), we report the fraction of tokens that belong to a consecutive block of more than $X$ easy tokens across different tasks. If an easy region is defined to be a consecutive block of $>$10 “easy” tokens, 50-70% of tokens belong to an easy region. Moreover, 15-26% of tokens belong to an easy region of $>$50 tokens.

Figure: Figure 4: [Qwen2.5-32B, MATH] The varying difficulty of decoding within a sequence. The raster plot visualizes whether each output token is accepted (green, “easier”) or rejected and regenerated by the target model (white, “harder”). Note that an all-white chunk at the right end indicates a shorter sequence.
Refer to caption: https://arxiv.org/html/2512.20573/x4.png

Figure: Figure 5: [Qwen2.5-32B] A large fraction of tokens belong to a consecutive block of more than $X$ “easy” tokens.
Refer to caption: https://arxiv.org/html/2512.20573/x5.png

Standard speculative decoding approaches typically utilize a static speculation length (e.g., $n=10$ tokens). While this aims to balance speculation overhead and potential speedup, it is suboptimal for the dynamic nature of generation:

- $\bullet$
In easier regions (e.g., tokens 150-380 for query 4 in Fig. [4](https://arxiv.org/html/2512.20573v3#S3.F4)): A short, fixed speculation length results in undue verification overhead. Even though the drafter could correctly speculate the next 10 tokens, its token generation is forced to pause every 10 tokens to invoke the target model for verification. Since target model inference is memory-bound, the frequent loading of model weights incurs significant latency overhead, preventing the system from reaching peak overall efficiency.
- $\bullet$
In harder regions (e.g., tokens 200-250 for query 18 in Fig. [4](https://arxiv.org/html/2512.20573v3#S3.F4)): A fixed speculation length results in undue speculation overhead. The drafter wastes compute generating 10 tokens, most of which are likely to be rejected (after the first or second position), resulting in wasted computation and hurting the end-to-end speedup.

Altogether, these properties provide a unique opportunity to change the way drafters are used in speculative decoding.
To cope with the varying difficulty of decoding, an ideal speculative decoding paradigm would dynamically increase the speculation length in easy regions to reduce the frequency of verification and amortize its cost, while minimizing the drafter’s compute in hard regions to minimize wasted effort.
Standard autoregressive (AR) draft models are ill-suited for this dynamic strategy – each AR drafter’s forward pass only generates a single token, which makes large speculation lengths (e.g., speculating $>$30 tokens at a time) prohibitively expensive and risky (Fig. [2](https://arxiv.org/html/2512.20573v3#footnote2)), since the probability of each token’s acceptance drops exponentially, and the drafting latency eventually overshadows the savings from parallel verification.
In contrast, we can reduce the compute budget of dLLM drafters to relax this harsh tradeoff, providing an avenue to get lengthy drafts rapidly.

## 4 Method

Capitalizing on the aforementioned observations, we present FailFast, a speculative decoding framework that employs Diffusion LLMs (dLLMs) as draft models to autoregressive verifiers.
FailFast departs from traditional optimization strategies by deliberately minimizing the computational effort of the drafter to fail fast and minimize the speculation latency of the draft model; in regions of lower speculation difficulty, it wins big through aggressively extending the speculation length and reducing the verification latency of the target model.

### 4.1 “Fail Fast”: Embracing Error-Proneness for Faster Speculation

There has been a myriad of related work on improving the efficacy of speculative decoding by improving the draft acceptance rate. Typically, this is achieved by enhancing the drafted tokens’ quality through using an ensemble of small drafters to construct token trees (Miao et al., 2024; Cai et al., 2024; Li et al., 2025; Chen et al., 2024), extensive fine-tuning of the draft model (Liu et al., 2023; atl, 2025; Sandler et al., 2025), performing online drafter selection (Liu et al., 2025a), etc.

In our work, we propose a counter-intuitive approach: we explicitly embrace the error-proneness of drafters in speculative decoding.
Instead of refining the draft output to match the target distribution better, we restrict the dLLM to the absolute minimum computational budget – typically only using one denoising step (model forward pass). Our rationale is twofold:

- $\bullet$
Easy regions: In regions of low difficulty (e.g., summarization), even a coarse, one-step generation from a small dLLM is often sufficient to speculate the correct tokens (Fig. [2](https://arxiv.org/html/2512.20573v3#S3.F2)). Additional refinement steps in these regions are wasted compute.
- $\bullet$
Hard regions: In regions of high difficulty (e.g., complex reasoning), a small draft model is statistically likely to diverge from the target model regardless of how many refinement steps are applied. Although spending additional compute to refine the draft tokens does indeed improve the acceptance rate, it yields diminishing returns (Fig. [2](https://arxiv.org/html/2512.20573v3#S3.F2)): the quality of speculated tokens is still inherently bounded by the capability of the small drafter, and a significant portion of tokens is eventually rejected anyway.

By adopting dLLMs as draft models, we relax the requirement for high-quality language modeling compared to using them as standalone models, allowing us to operate in a low-quality but blazing-fast mode that minimizes speculation latency per token.
The verification stage then catches errors, enabling a “fail fast” approach that quickly returns the task of decoding to the target model with minimal speculation overhead when the task becomes difficult.

### 4.2 “Win Big”: Dynamic Expansion of Speculation Length for Faster Verification

While reducing speculation latency addresses the cost of generating drafts, it does not lower the verification latency – the overhead incurred by executing the large target model, which is a factor of both the number of verifications (rounds of speculation) and the verification latency in each round. If we strictly use short speculation lengths, the target model is invoked frequently, creating a latency bottleneck.

To mitigate this, FailFast capitalizes on “easy” segments by opportunistically increasing the speculation length.
When the draft model encounters a subsequence that is easier to speculate, we defer verification and aggressively extend the length of the speculated draft.
Because speculation has, by design, such a low cost in FailFast, the risk of “failing” is low, and we can afford these aggressive speculations.

#### Confidence as a difficulty signal.

To detect these easy regions without ground-truth knowledge of difficulty, we utilize the dLLM’s internal confidence (Fu et al., 2025; Dai et al., 2024) as a proxy for speculation difficulty. Although dLLMs are non-autoregressive, they use the same Transformers backbone as AR LLMs, so we still have access to the probability distribution over the vocabulary for each token. For a dLLM predicting a token $x_{i}$ at position $i$, we define its confidence $C_{i}$ as the maximum probability in the output distribution $P_{i}$:
$C_{i}=\max_{v\in V}P_{i}(x_{i}=v)$, where $V$ is the vocabulary.

Figure: Algorithm 1 FailFast Main Logic in Each Round

#### Dynamic speculation length expansion logic.

Our adaptive strategy is detailed in Algorithm [1](https://arxiv.org/html/2512.20573v3#alg1). We begin with a default speculation length $N$. After generating these initial tokens (using minimal compute), we inspect their confidence scores.
If all tokens’ confidence in the current speculated sequence exceeds a confidence threshold $\tau$ (indicating an “easy” region), we assume the draft will likely be accepted. Instead of stopping to verify, we immediately extend the speculation length by another $N$ tokens and invoke the drafter again.
We repeat this extension process until a low-confidence token is detected or a maximum sequence length $N_{\max}$ is reached.
This mechanism allows FailFast to generate massive chunks of tokens (e.g., 70 tokens) in easy regions in a single round of speculation, and to get a majority of them accepted in many cases, which drastically reduces the number of times the target model needs to be invoked (verification rounds) – the verification overhead in each round stays the same because short prefills are memory-bound (Agrawal et al., 2024) – while reverting to short bursts in harder regions.

While the notion of dynamically adjusting the draft length is an established concept in speculative decoding (Liu et al., 2024b), the resulting speedups are often limited, hindered by the risk of invoking undue speculation latency.
For example, SpecDec++ (Huang et al., 2024) achieves an average speedup of $\sim$10% over its baselines.
In comparison, FailFast overcomes this by using minimal compute to minimize the risk of incorrect speculation, allowing it to drastically adapt the speculation length on-the-fly.

We include a discussion of alternative design choices and optimizations in Appendix [E](https://arxiv.org/html/2512.20573v3#A5).

### 4.3 Policy Design vs. Training-Based Approaches

Our work focuses specifically on how to adapt drafting policies when using dLLMs as drafters, rather than improving performance through additional training.
Indeed, most state-of-the-art dLLMs are adapted from pretrained autoregressive (AR) weights (Wu et al., 2025a; Bie et al., 2025; Gong et al., 2024; Zhou et al., 2025) and are therefore already implicitly aligned with the target AR models when used as drafters.
Leveraging this property, we study how to better exploit dLLMs at inference time via policy design.
That said, our approach is complementary to prior optimizations that aim to improve the overall acceptance rate of drafted tokens (Miao et al., 2024; Chen et al., 2024; Liu et al., 2023; atl, 2025; Liu et al., 2025a), including recent work that fine-tunes the dLLM drafter for improved alignment and higher acceptance rates (Christopher et al., 2025; Sandler et al., 2025). These techniques are complementary to FailFast, as the dLLMs we employ can be further fine-tuned to fit the base model’s distribution better.

## 5 Evaluation

**Table 1: Comparison of speedups of different drafting methods over vanilla decoding. FailFast achieves the highest average speedup.**
| Drafting Method | Dataset |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| MATH | AIME | GSM8K | GPQA | HumanEval | Average |  |
| Target Model: Qwen2.5-32B-Instruct |  |  |  |  |  |  |
| AR Draft Model | ${2.93\times}$ | ${2.84\times}$ | ${2.82\times}$ | ${2.26\times}$ | ${2.72\times}$ | ${2.71\times}$ |
| Fast-dLLM | ${3.57\times}$ | ${3.29\times}$ | ${3.10\times}$ | ${2.42\times}$ | ${3.16\times}$ | ${3.11\times}$ |
| EAGLE-3 (w/ draft tree) | ${2.87\times}$ | ${2.85\times}$ | ${2.59\times}$ | ${2.42\times}$ | ${3.31\times}$ | ${2.81\times}$ |
| FailFast (ours) | ${\mathbf{4.90\times}}$ | ${\mathbf{4.40\times}}$ | ${\mathbf{3.71\times}}$ | ${\mathbf{3.11\times}}$ | ${\mathbf{4.06\times}}$ | ${\mathbf{4.04\times}}$ |
| Target Model: Qwen2.5-14B-Instruct |  |  |  |  |  |  |
| AR Draft Model | ${2.05\times}$ | ${1.94\times}$ | ${1.95\times}$ | ${1.71\times}$ | ${1.91\times}$ | ${1.91\times}$ |
| Fast-dLLM | ${2.57\times}$ | ${2.22\times}$ | ${2.22\times}$ | ${1.86\times}$ | ${2.23\times}$ | ${2.22\times}$ |
| EAGLE-3 (w/ draft tree) | ${2.48\times}$ | ${2.38\times}$ | ${2.23\times}$ | ${2.04\times}$ | ${2.68\times}$ | ${2.36\times}$ |
| FailFast (ours) | ${\mathbf{3.92\times}}$ | ${\mathbf{3.37\times}}$ | ${\mathbf{3.04\times}}$ | ${\mathbf{2.54\times}}$ | ${\mathbf{3.41\times}}$ | ${\mathbf{3.26\times}}$ |
| Target Model: Qwen2.5-7B-Instruct |  |  |  |  |  |  |
| AR Draft Model | ${1.43\times}$ | ${1.42\times}$ | ${1.41\times}$ | ${1.25\times}$ | ${1.40\times}$ | ${1.38\times}$ |
| Fast-dLLM | ${1.95\times}$ | ${1.57\times}$ | ${1.54\times}$ | ${1.34\times}$ | ${1.61\times}$ | ${1.60\times}$ |
| EAGLE-3 (w/ draft tree) | ${2.31\times}$ | ${2.36\times}$ | ${2.02\times}$ | ${\mathbf{1.98\times}}$ | ${2.39\times}$ | ${2.21\times}$ |
| FailFast (ours) | ${\mathbf{3.06\times}}$ | ${\mathbf{2.63\times}}$ | ${\mathbf{2.34\times}}$ | ${1.89\times}$ | ${\mathbf{2.71\times}}$ | ${\mathbf{2.52\times}}$ |

### 5.1 Setup

Models and Baselines. In our main experiments, we evaluate three target models of different sizes: Qwen2.5-{32B,14B,7B}-Instruct (Yang et al., 2024). The different speculative decoding schemes we evaluate are listed below. For all schemes, we set the generation temperature of all models to 0 and set the maximum number of output tokens to 1024.
We discuss more details of our best-effort reproduction of all baselines and additional baselines in Appendix [D](https://arxiv.org/html/2512.20573v3#A4).

- $\bullet$
AR drafter (Leviathan et al., 2023): We adopt Qwen2.5-1.5B-Instruct as our autoregressive speculative sampling baseline, which uses the same training data as Qwen2.5-32B. For each dataset, we conduct an extensive parameter sweep ($n=$ 3-20) and pick the speculation length that achieves the best speedup.
- $\bullet$
Fast-dLLM (Wu et al., 2025a): For the dLLM drafter baseline, we employ Fast_dLLM_v2_1.5B (Wu et al., 2025a), the state-of-the-art semi-autoregressive diffusion LLM that embeds efficiency optimizations such as approximate caching and confidence-aware parallel decoding. We adopt its default hyperparameters that achieve the best balance between accuracy and generation speed. This baseline represents naively plugging in a dLLM as the draft model without the additional optimizations of FailFast.
- $\bullet$
EAGLE-3 (Li et al., 2025):
EAGLE-3 is one of the most widely deployed state-of-the-art speculative decoding drafters. As a lightweight single-layer autoregressive Transformer, EAGLE-3 introduces a multi-layer feature fusion and a training-time test mechanism for improving drafting quality. Different from other strategies where a single chain of draft tokens is proposed for verification, EAGLE-3 leverages a draft-tree-based mechanism that improves the overall token acceptance rate.
- $\bullet$
FailFast: FailFast employs Fast_dLLM_v2_1.5B as the underlying model and performs one-step generation and dynamic speculation length expansion. For the hyperparameters in Alg. [1](https://arxiv.org/html/2512.20573v3#alg1), we set $\tau=\{0.4,0.45,0.5\}$ for the {32,14,7}B target model and $N=10$. Although picking specific hyperparameters for each dataset/model combination yields higher speedups, we use the same set of hyperparameters across datasets to demonstrate generalizability. FailFast’s performance degrades gracefully as hyperparameters are tweaked (Appendix [C](https://arxiv.org/html/2512.20573v3#A3)).

Datasets. Similar to prior work (Wu et al., 2025a; Sandler et al., 2025; Li et al., 2025), we evaluate FailFast on a wide range of diverse benchmarks: MATH, AIME, and GSM8K (Hendrycks et al., 2021; aim, 2025; Cobbe et al., 2021) for mathematical reasoning, GPQA (Rein et al., 2024) for knowledge-intensive question answering, and HumanEval (Chen, 2021) for code generation.

Hardware. We run our evaluations on two NVIDIA A6000-48GB GPUs connected via PCIe 4.0 x16. We profile latency using SGLang (Zheng et al., 2023), enable prefix caching (Pan et al., 2025b) for both the draft model and target model, and use TP=2 for all target models.

### 5.2 End-to-End Results

Table [1](https://arxiv.org/html/2512.20573v3#S5.T1) presents the end-to-end speedups over vanilla decoding across different schemes and datasets.

Workload sensitivity. Across models and datasets, FailFast achieves an overall speedup of 1.9-4.9$\times$. Compared to baselines, FailFast achieves a speedup of 1.3-2.1$\times$ over the AR drafter baseline and a speedup of 1.2-1.7$\times$ over the Fast-dLLM baseline. FailFast has a higher win over baselines on datasets that are easier to speculate: math and coding workloads exhibit centralized regions of predictable tokens (e.g., intermediate calculations and code generation), allowing FailFast to capitalize on long speculation windows. Conversely, datasets where “easy” tokens are scattered rather than clustered offer fewer opportunities for aggressive speculation length expansion, though FailFast still outperforms fixed-length baselines with moderate length expansion.

Efficiency of FailFast. Naively plugging in a dLLM as the drafter (Fast-dLLM) retains the acceptance rate (Tab. [5](https://arxiv.org/html/2512.20573v3#footnote5)) and achieves a speedup over the AR drafter. However, as shown in Fig. [6](https://arxiv.org/html/2512.20573v3#S5.F6), this gain is primarily derived from a reduction in speculation latency (22.9% on average)(^3^33This improvement falls short of the theoretical end-to-end speedups of dLLMs (Wu et al., 2025a) because our termination criterion requires the next continuous $n$ tokens to be unmasked; due to the random decoding order within token blocks, the leftmost tokens are not necessarily decoded first.) while the verification latency remains near-constant.

Figure: Figure 6: Breakdown of end-to-end latency into speculation (lighter, upper) and verification (darker, lower). While applying a dLLM drafter (Fast-dLLM) reduces the speculation latency, FailFast further reduces the speculation latency (in each round) while also reducing the verification latency (the number of rounds).
Refer to caption: https://arxiv.org/html/2512.20573/x6.png

In contrast, FailFast further reduces speculation latency significantly (a further 41.1% over Fast-dLLM) by employing an ultra-small compute budget for the drafter. Although our approach inevitably degrades the acceptance rate (Tab. [5](https://arxiv.org/html/2512.20573v3#footnote5)) – a combined result of a low drafter compute budget and proposing more tokens on average in each round – we achieve superior end-to-end speedups (e.g., 3.1-4.9$\times$ for Qwen2.5-32B). In addition, FailFast reduces verification latency (Fig. [6](https://arxiv.org/html/2512.20573v3#S5.F6)) by an average of 17.1% via adopting adaptive speculation lengths: through dynamically expanding the speculation window in “easy” regions, we can speculate and verify up to 70 tokens (Fig. [8](https://arxiv.org/html/2512.20573v3#A3.F8)) in a single round. This drastic reduction in the total number of speculation-verification rounds and the speculation overhead in each round outweighs the cost of a lower per-token acceptance rate. We include an example trajectory in Appendix [F](https://arxiv.org/html/2512.20573v3#A6) to visualize FailFast in action.

Detailed analyses. Due to space constraints, we only further dissect FailFast’s performance improvements on Qwen2.5-32B on the MATH dataset, and we report the data on all other datasets and model combinations in Appendix [B](https://arxiv.org/html/2512.20573v3#A2).
Even though the AR drafter has a higher overall acceptance rate (Tab. [5](https://arxiv.org/html/2512.20573v3#footnote5)), for each of its forward passes, it produces only one token and accepts 0.6 tokens on average. In contrast, FailFast can produce an average of 4.1 tokens per forward pass and accept 1.7 of them.
Another source of FailFast’s win comes from its aggressive speculation length expansion. Fig. [8](https://arxiv.org/html/2512.20573v3#A3.F8) shows a CDF of the accepted/speculated length across rounds. In $\sim$20% of the rounds, we extended the speculation length beyond the default, reducing the average number of speculation-verification rounds by 16.0%.
Interestingly, FailFast’s dynamic policy converges to an average speculation length of 14.6 tokens, closely matching the theoretically optimal of 16 tokens from our prior analysis (Fig. [2](https://arxiv.org/html/2512.20573v3#footnote2)).

Figure: Figure 7: [Qwen2.5-32B, MATH] CDF of the number of accepted/speculated tokens in each round. Full results are in Tab. [5](https://arxiv.org/html/2512.20573v3#footnote5).
Refer to caption: https://arxiv.org/html/2512.20573/x7.png

Comparisons with EAGLE-3.
Across models, FailFast achieves an average speedup of 1.0–1.4$\times$ over EAGLE-3, with gains of up to 1.3–1.7$\times$ on specific datasets.
Even though EAGLE-3 incurs a smaller drafting latency in each of its forward passes – it only has one layer compared to the 28 layers in other drafters, reducing overheads associated with loading weights and launching kernels – FailFast still outperforms EAGLE-3.
This is because EAGLE-3 remains an autoregressive drafter that generates only one token per forward pass, whereas FailFast generates multiple tokens in each pass.
Moreover, while EAGLE-3 speculates a tree of 50–60 tokens per round, its draft tree is relatively shallow (default depth 8 (Li et al., 2025)) compared to FailFast, which drafts a chain of up to 70 tokens in a single round.
As a result, EAGLE-3’s upper bound on the number of accepted tokens is constrained by tree depth, limiting its ability to exploit extended regions of easier tokens.

Due to space constraints, we refer readers to the Appendix sections for a more in-depth discussion regarding EAGLE-3, the implementation details of baselines, a hyperparameter sensitivity analysis of FailFast, and a discussion of explorations of alternative design choices.

## 6 Conclusion

In this paper, we present FailFast, a novel speculative decoding framework that leverages dLLMs to achieve significant lossless acceleration. By spending minimal compute in hard regions to “fail fast” and aggressively extending speculation lengths in easier segments to “win big,” FailFast minimizes both speculation and verification latencies. Without requiring fine-tuning, the framework achieves up to 4.9$\times$ speedup over vanilla decoding and significantly outperforms state-of-the-art baselines like Fast-dLLM and EAGLE-3, demonstrating that dLLMs are uniquely suited for adaptive, high-efficiency drafting.

Impact Statement This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

Acknowledgments We thank Princeton’s Systems for Artificial Intelligence Lab (SAIL) and Princeton Language and Intelligence (PLI) for providing the hardware resources for running experiments.
Rui would like to thank the many full-time researchers and interns at Google SRG for their help and support, especially Nikos Pagonas, Howie Chen, Manos Chatzakis, Vinay Banakar, Liana Patel, Jiani Zhang, and Lily Tsai.
In addition, Rui would like to thank Minghao Yan for a random conversation on speculative decoding back in 2024 that planted the seed of this work and for providing feedback on an earlier version of this paper.

## References

- aim (2025)
Aime 2024 dataset card.
[https://huggingface.co/datasets/HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024), 2025.
- amd (2025)
Accelerating llm inference: Up to 3x speedup on mi300x with speculative decoding.
[https://rocm.blogs.amd.com/artificial-intelligence/spec_decode_mi300x/README.html](https://rocm.blogs.amd.com/artificial-intelligence/spec_decode_mi300x/README.html), 2025.
- atl (2025)
Adaptive-learning speculator system (atlas): A new paradigm in llm inference via runtime-learning accelerators — together ai.
[https://www.together.ai/blog/adaptive-learning-speculator-system-atlas](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas), 2025.
- eag (2025)
Fly eagle(3) fly: Faster inference with vllm & speculative decoding — red hat developer.
[https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding), 2025.
- gem (2025)
Gemini diffusion - google deepmind.
[https://deepmind.google/models/gemini-diffusion/](https://deepmind.google/models/gemini-diffusion/), 2025.
- spe (2025)
Specforge.
[https://github.com/sgl-project/SpecForge](https://github.com/sgl-project/SpecForge), 2025.
- Agrawal et al. (2024)
Agrawal, A., Kedia, N., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B., Tumanov, A., and Ramjee, R.
Taming throughput-latency tradeoff in llm inference with sarathi-serve.
In *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)*, pp. 117–134, 2024.
- Agrawal et al. (2025)
Agrawal, S., Garrepalli, R., Goel, R., Lee, M., Lott, C., and Porikli, F.
Spiffy: Multiplying diffusion llm acceleration via lossless speculative decoding.
*arXiv preprint arXiv:2509.18085*, 2025.
- Arriola et al. (2025)
Arriola, M., Gokaslan, A., Chiu, J. T., Yang, Z., Qi, Z., Han, J., Sahoo, S. S., and Kuleshov, V.
Block diffusion: Interpolating between autoregressive and diffusion language models.
*arXiv preprint arXiv:2503.09573*, 2025.
- Bie et al. (2025)
Bie, T., Cao, M., Chen, K., Du, L., Gong, M., Gong, Z., Gu, Y., Hu, J., Huang, Z., Lan, Z., et al.
Llada2.0: Scaling up diffusion language models to 100b.
*arXiv preprint arXiv:2512.15745*, 2025.
- Burton (1985)
Burton, F. W.
Speculative computation, parallelism, and functional programming.
*IEEE Transactions on Computers*, 100(12):1190–1193, 1985.
- Cai et al. (2024)
Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., and Dao, T.
Medusa: Simple llm inference acceleration framework with multiple decoding heads.
*arXiv preprint arXiv:2401.10774*, 2024.
- Chen (2021)
Chen, M.
Evaluating large language models trained on code.
*arXiv preprint arXiv:2107.03374*, 2021.
- Chen et al. (2024)
Chen, Z., May, A., Svirschevski, R., Huang, Y.-H., Ryabinin, M., Jia, Z., and Chen, B.
Sequoia: Scalable and robust speculative decoding.
*Advances in Neural Information Processing Systems*, 37:129531–129563, 2024.
- Christopher et al. (2025)
Christopher, J. K., Bartoldson, B. R., Ben-Nun, T., Cardei, M., Kailkhura, B., and Fioretto, F.
Speculative diffusion decoding: Accelerating language generation through diffusion.
In *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pp. 12042–12059, 2025.
- Cobbe et al. (2021)
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al.
Training verifiers to solve math word problems.
*arXiv preprint arXiv:2110.14168*, 2021.
- Dai et al. (2024)
Dai, Y., Pan, R., Iyer, A., Li, K., and Netravali, R.
Apparate: Rethinking early exits to tame latency-throughput tensions in ml serving.
In *Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles*, pp. 607–623, 2024.
- Dao & Gu (2024)
Dao, T. and Gu, A.
Transformers are ssms: Generalized models and efficient algorithms through structured state space duality.
*arXiv preprint arXiv:2405.21060*, 2024.
- Ding et al. (2023)
Ding, N., Chen, Y., Xu, B., Qin, Y., Hu, S., Liu, Z., Sun, M., and Zhou, B.
Enhancing chat language models by scaling high-quality instructional conversations.
In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pp. 3029–3051, 2023.
- Fu et al. (2024)
Fu, Y., Bailis, P., Stoica, I., and Zhang, H.
Break the sequential dependency of llm inference using lookahead decoding.
*arXiv preprint arXiv:2402.02057*, 2024.
- Fu et al. (2025)
Fu, Y., Wang, X., Tian, Y., and Zhao, J.
Deep think with confidence.
*arXiv preprint arXiv:2508.15260*, 2025.
- Gong et al. (2024)
Gong, S., Agarwal, S., Zhang, Y., Ye, J., Zheng, L., Li, M., An, C., Zhao, P., Bi, W., Han, J., et al.
Scaling diffusion language models via adaptation from autoregressive models.
*arXiv preprint arXiv:2410.17891*, 2024.
- He et al. (2023)
He, Z., Zhong, Z., Cai, T., Lee, J. D., and He, D.
Rest: Retrieval-based speculative decoding.
*arXiv preprint arXiv:2311.08252*, 2023.
- Hendrycks et al. (2021)
Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J.
Measuring mathematical problem solving with the math dataset.
*arXiv preprint arXiv:2103.03874*, 2021.
- Huang et al. (2024)
Huang, K., Guo, X., and Wang, M.
Specdec++: Boosting speculative decoding via adaptive candidate lengths.
*arXiv preprint arXiv:2405.19715*, 2024.
- Jacobson (1988)
Jacobson, V.
Congestion avoidance and control.
*ACM SIGCOMM computer communication review*, 18(4):314–329, 1988.
- Kang et al. (2025)
Kang, W., Galim, K., Oh, S., Lee, M., Zeng, Y., Zhang, S., Hooper, C., Hu, Y., Koo, H. I., Cho, N. I., et al.
Parallelbench: Understanding the trade-offs of parallel decoding in diffusion llms.
*arXiv preprint arXiv:2510.04767*, 2025.
- Khanna et al. (2025)
Khanna, S., Kharbanda, S., Li, S., Varma, H., Wang, E., Birnbaum, S., Luo, Z., Miraoui, Y., Palrecha, A., Ermon, S., et al.
Mercury: Ultra-fast language models based on diffusion.
*arXiv preprint arXiv:2506.17298*, 2025.
- Kwon et al. (2023)
Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., and Stoica, I.
Efficient memory management for large language model serving with pagedattention.
In *Proceedings of the 29th Symposium on Operating Systems Principles*, pp. 611–626, 2023.
- Leviathan et al. (2023)
Leviathan, Y., Kalman, M., and Matias, Y.
Fast inference from transformers via speculative decoding.
In *International Conference on Machine Learning*, pp. 19274–19286. PMLR, 2023.
- Li et al. (2025)
Li, Y., Wei, F., Zhang, C., and Zhang, H.
Eagle-3: Scaling up inference acceleration of large language models via training-time test.
*arXiv preprint arXiv:2503.01840*, 2025.
- Liu et al. (2025a)
Liu, H., Huang, J., Jia, Z., Park, Y., and Wang, Y.-X.
Not-a-bandit: Provably no-regret drafter selection in speculative decoding for llms.
*arXiv preprint arXiv:2510.20064*, 2025a.
- Liu et al. (2024a)
Liu, J., Wang, Q., Wang, J., and Cai, X.
Speculative decoding via early-exiting for faster llm inference with thompson sampling control mechanism.
*arXiv preprint arXiv:2406.03853*, 2024a.
- Liu et al. (2023)
Liu, X., Hu, L., Bailis, P., Cheung, A., Deng, Z., Stoica, I., and Zhang, H.
Online speculative decoding.
*arXiv preprint arXiv:2310.07177*, 2023.
- Liu et al. (2024b)
Liu, X., Daniel, C., Hu, L., Kwon, W., Li, Z., Mo, X., Cheung, A., Deng, Z., Stoica, I., and Zhang, H.
Optimizing speculative decoding for serving large language models using goodput.
*arXiv preprint arXiv:2406.14066*, 2024b.
- Liu et al. (2025b)
Liu, Z., Yang, Y., Zhang, Y., Chen, J., Zou, C., Wei, Q., Wang, S., and Zhang, L.
dllm-cache: Accelerating diffusion large language models with adaptive caching.
*arXiv preprint arXiv:2506.06295*, 2025b.
- Luo et al. (2024)
Luo, X., Wang, Y., Zhu, Q., Zhang, Z., Zhang, X., Yang, Q., Xu, D., and Che, W.
Turning trash into treasure: Accelerating inference of large language models with token recycling, 2024.
URL [https://arxiv.org/abs/2408.08696](https://arxiv.org/abs/2408.08696).
- Ma et al. (2025)
Ma, X., Yu, R., Fang, G., and Wang, X.
dkv-cache: The cache for diffusion language models.
*arXiv preprint arXiv:2505.15781*, 2025.
- Miao et al. (2024)
Miao, X., Oliaro, G., Zhang, Z., Cheng, X., Wang, Z., Zhang, Z., Wong, R. Y. Y., Zhu, A., Yang, L., Shi, X., et al.
Specinfer: Accelerating large language model serving with tree-based speculative inference and verification.
In *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3*, pp. 932–949, 2024.
- Nie et al. (2025)
Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., and Li, C.
Large language diffusion models.
*arXiv preprint arXiv:2502.09992*, 2025.
- Oliaro et al. (2024)
Oliaro, G., Jia, Z., Campos, D., and Qiao, A.
Suffixdecoding: A model-free approach to speeding up large language model inference, 2024.
URL [https://arxiv.org/abs/2411.04975](https://arxiv.org/abs/2411.04975).
- Pan et al. (2025a)
Pan, R., Dai, Y., Zhang, Z., Oliaro, G., Jia, Z., and Netravali, R.
Specreason: Fast and accurate inference-time compute via speculative reasoning.
*arXiv preprint arXiv:2504.07891*, 2025a.
- Pan et al. (2025b)
Pan, R., Wang, Z., Jia, Z., Karakus, C., Zancato, L., Dao, T., Wang, Y., and Netravali, R.
Marconi: Prefix caching for the era of hybrid llms.
In *Eighth Conference on Machine Learning and Systems*, 2025b.
- Qian et al. (2026)
Qian, Y.-Y., Su, J., Hu, L., Zhang, P., Deng, Z., Zhao, P., and Zhang, H.
d3llm: Ultra-fast diffusion llm using pseudo-trajectory distillation.
*arXiv preprint arXiv:2601.07568*, 2026.
- Rein et al. (2024)
Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R.
Gpqa: A graduate-level google-proof q&a benchmark.
In *First Conference on Language Modeling*, 2024.
- Sadhukhan et al. (2024)
Sadhukhan, R., Chen, J., Chen, Z., Tiwari, V., Lai, R., Shi, J., Yen, I. E.-H., May, A., Chen, T., and Chen, B.
Magicdec: Breaking the latency-throughput tradeoff for long context generation with speculative decoding.
*arXiv preprint arXiv:2408.11049*, 2024.
- Sandler et al. (2025)
Sandler, J., Christopher, J. K., Hartvigsen, T., and Fioretto, N.
Specdiff-2: Scaling diffusion drafter alignment for faster speculative decoding.
*arXiv preprint arXiv:2511.00606*, 2025.
- Song et al. (2025)
Song, Y., Zhang, Z., Luo, C., Gao, P., Xia, F., Luo, H., Li, Z., Yang, Y., Yu, H., Qu, X., et al.
Seed diffusion: A large-scale diffusion language model with high-speed inference.
*arXiv preprint arXiv:2508.02193*, 2025.
- Stern et al. (2018)
Stern, M., Shazeer, N., and Uszkoreit, J.
Blockwise parallel decoding for deep autoregressive models.
*Advances in Neural Information Processing Systems*, 31, 2018.
- Waleffe et al. (2024)
Waleffe, R., Byeon, W., Riach, D., Norick, B., Korthikanti, V., Dao, T., Gu, A., Hatamizadeh, A., Singh, S., Narayanan, D., et al.
An empirical study of mamba-based language models.
*arXiv preprint arXiv:2406.07887*, 2024.
- Wu et al. (2025a)
Wu, C., Zhang, H., Xue, S., Diao, S., Fu, Y., Liu, Z., Molchanov, P., Luo, P., Han, S., and Xie, E.
Fast-dllm v2: Efficient block-diffusion llm.
*arXiv preprint arXiv:2509.26328*, 2025a.
- Wu et al. (2025b)
Wu, C., Zhang, H., Xue, S., Liu, Z., Diao, S., Zhu, L., Luo, P., Han, S., and Xie, E.
Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding.
*arXiv preprint arXiv:2505.22618*, 2025b.
- Xia et al. (2024)
Xia, H., Yang, Z., Dong, Q., Wang, P., Li, Y., Ge, T., Liu, T., Li, W., and Sui, Z.
Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding.
*arXiv preprint arXiv:2401.07851*, 2024.
- Yan et al. (2024)
Yan, M., Agarwal, S., and Venkataraman, S.
Decoding speculative decoding.
*arXiv preprint arXiv:2402.01528*, 2024.
- Yang et al. (2024)
Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., et al.
Qwen2. 5 technical report.
*arXiv preprint arXiv:2412.15115*, 2024.
- Ye et al. (2025)
Ye, J., Xie, Z., Zheng, L., Gao, J., Wu, Z., Jiang, X., Li, Z., and Kong, L.
Dream 7b: Diffusion large language models.
*arXiv preprint arXiv:2508.15487*, 2025.
- Zhao et al. (2024)
Zhao, Y., Xie, Z., Liang, C., Zhuang, C., and Gu, J.
Lookahead: An inference acceleration framework for large language model with lossless generation accuracy.
In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, KDD ’24, pp. 6344–6355. Association for Computing Machinery, 2024.
ISBN 9798400704901.
doi: 10.1145/3637528.3671614.
- Zheng et al. (2023)
Zheng, L., Yin, L., Xie, Z., Huang, J., Sun, C., Yu, C., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., et al.
Efficiently programming large language models using sglang.
2023.
- Zhou et al. (2025)
Zhou, Z., Chen, L., Tong, H., and Song, D.
dllm: Simple diffusion language modeling.
[https://github.com/ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm), 2025.

## Appendix A List of abbreviations and notations

**Table 2: Glossary of abbreviations, notations, and terminology.**
| Notation | Definition |
| --- | --- |
| AR | Autoregressive |
| dLLM | Diffusion language models |
| Round | A speculation-verification round |
| $n$ / $N$ | Speculation length (number of tokens) in each round |

## Appendix B Detailed Performance Comparison

**Table 3: Detailed performance comparison. We report the average acceptance rate, average accepted/speculated lengths in each round(^5^55The acceptance rate does not exactly equal the average accepted length divided by the average speculation length, because the average of averages does not necessarily equal the overall average. Note that, unlike prior work (Sandler et al., 2025), we do not add one to the average accepted length to represent the token (either correcting the rejected token, or the bonus token if none is rejected) from the verifier.), maximum accepted/speculated lengths across rounds, average number of speculation-verification rounds, and the average number of drafter forward passes in each round.**
| Drafting Method | Metric | Dataset |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MATH | AIME | GSM8K | GPQA | HumanEval | Average |  |  |
| Target: Qwen2.5-32B-Instruct |  |  |  |  |  |  |  |
| AR drafter | Acceptance Rate | 58.2% | 56.0% | 55.3% | 51.2% | 56.1% | 55.4% |
| Avg (Acc./Spec.) Len | 4.7 / 8 | 4.5 / 8 | 4.4 / 8 | 2.6 / 5 | 3.9 / 7 | 4.0 / 7.2 |  |
| Max (Acc./Spec.) Len | 8 / 8 | 8 / 8 | 8 / 8 | 5 / 5 | 7 / 7 | - |  |
| Num Speculation Rounds | 93.6 | 138.8 | 56.1 | 138.3 | 76.6 | 100.7 |  |
| Avg Drafter Passes / Round | 8 | 8 | 8 | 5 | 7 | 7.2 |  |
| Fast-dLLM | Acceptance Rate | 60.5% | 56.9% | 53.2% | 52.7% | 57.7% | 56.2% |
| Avg (Acc./Spec.) Len | 4.8 / 8 | 4.5 / 8 | 4.3 / 8 | 2.6 / 5 | 4.0 / 7 | 4.0 / 7.2 |  |
| Max (Acc./Spec.) Len | 8 / 8 | 8 / 8 | 8 / 8 | 5 / 5 | 7 / 7 | - |  |
| Num Speculation Rounds | 90.4 | 137.4 | 57.3 | 135.3 | 74.3 | 98.9 |  |
| Avg Drafter Passes / Round | 5.7 | 6.1 | 6.1 | 4.4 | 5.2 | 5.5 |  |
| FailFast | Acceptance Rate | 40.6% | 38.0% | 33.6% | 26.4% | 31.7% | 34.1% |
| Avg (Acc./Spec.) Len | 6.0 / 14.6 | 5.1 / 13.3 | 4.1 / 12.1 | 3.3 / 12.0 | 4.5 / 11.3 | 4.6 / 12.7 |  |
| Max (Acc./Spec.) Len | 70 / 70 | 70 / 70 | 50 / 70 | 70 / 70 | 60 / 70 | - |  |
| Num Speculation Rounds | 78.6 | 129.9 | 59.1 | 119.0 | 66.9 | 90.7 |  |
| Avg Drafter Passes / Round | 3.6 | 3.4 | 3.2 | 3.1 | 3.1 | 3.3 |  |
| Target: Qwen2.5-14B-Instruct |  |  |  |  |  |  |  |
| AR drafter | Acceptance Rate | 68.3% | 66.8% | 64.1% | 60.2% | 69.9% | 65.9% |
| Avg (Acc./Spec.) Len | 4.1 / 6 | 3.3 / 5 | 3.8 / 6 | 2.4 / 4 | 2.8 / 4 | 3.3 / 5.0 |  |
| Max (Acc./Spec.) Len | 6 / 6 | 5 / 5 | 6 / 6 | 4 / 4 | 4 / 4 | - |  |
| Num Speculation Rounds | 93.0 | 175.5 | 75.5 | 176.3 | 89.3 | 121.9 |  |
| Avg Drafter Passes / Round | 6 | 5 | 6 | 4 | 4 | 5 |  |
| Fast-dLLM | Acceptance Rate | 68.9% | 66.8% | 61.9% | 61.3% | 73.9% | 66.6% |
| Avg (Acc./Spec.) Len | 4.1 / 6 | 3.3 / 5 | 3.7 / 6 | 2.5 / 4 | 3.0 / 4 | 3.3 / 5.0 |  |
| Max (Acc./Spec.) Len | 6 / 6 | 5 / 5 | 6 / 6 | 4 / 4 | 4 / 4 | - |  |
| Num Speculation Rounds | 93.0 | 175.3 | 78.1 | 173.8 | 86.6 | 121.4 |  |
| Avg Drafter Passes / Round | 4.2 | 4.0 | 4.6 | 3.6 | 3.2 | 3.9 |  |
| FailFast | Acceptance Rate | 43.5% | 39.2% | 36.4% | 29.2% | 39.3% | 37.5% |
| Avg (Acc./Spec.) Len | 6.5 / 14.5 | 5.2 / 12.7 | 4.4 / 12.2 | 3.5 / 11.7 | 5.2 / 13.2 | 5.0 / 12.9 |  |
| Max (Acc./Spec.) Len | 60 / 70 | 70 / 70 | 60 / 60 | 70 / 70 | 60 / 70 | - |  |
| Num Speculation Rounds | 71.4 | 127.8 | 67.1 | 136.1 | 55.4 | 91.6 |  |
| Avg Drafter Passes / Round | 3.6 | 3.2 | 3.2 | 3.0 | 3.3 | 3.3 |  |

## Appendix C Hyperparameter Sensitivity Analysis of FailFast

Figure: (a) Impact of the confidence threshold $\tau$ in Alg. [1](https://arxiv.org/html/2512.20573v3#alg1).
Refer to caption: https://arxiv.org/html/2512.20573/x8.png

We analyze the sensitivity of FailFast to its key hyperparameters to understand the trade-offs between speculation overhead and verification efficiency. We focus our analysis on Qwen2.5-32B on MATH, and vary its default hyperparameters $\tau=0.4$ and $N=10$.
One advantage of FailFast is that, because drafting is low-latency by design, it does not require extensive tuning of hyperparameters to reach optimal performance, and the speculation length is self-adaptive based on token difficulty,
whereas traditional speculative decoding schemes typically require workload-aware tuning of the speculation length $N$ to reach optimal performance.

Confidence Threshold ($\tau$). The confidence threshold controls the aggressiveness of the speculation length expansion. We find that $\tau\in[0.3,0.55]$ yields optimal performance. A threshold that is too high (e.g., 0.7) makes FailFast overly conservative, preventing the proposal of long sequences in easy regions. Conversely, a threshold that is too low (e.g., 0.1) increases speculation latency overhead by generating tokens that are statistically likely to be rejected by the verifier.

Maximum Speculation Length ($N_{\max}$). We observe that $N_{\max}$ values between 30 and 70 perform reliably well. We adopt a default value of $N_{max}=60$ because LLM prefill for short sequences is typically memory-bandwidth bound, and the typical tipping point between memory-bound and compute-bound for medium-sized LLMs typically ranges between 64 and 128 tokens on modern GPUs (Agrawal et al., 2024). In high-throughput inference settings where batch sizes are large, bigger values of $N_{\max}$ still work, although they might inflate the verification latency slightly – the prefill pass would eventually become compute-bound for large sequence lengths, even with prefix caching (Pan et al., 2025b). In that case, synchronous verification might force a reduction in $N_{\max}$ to avoid inflating the per-round verification latency, and FailFast’s performance improvement from “winning big” might drop slightly. However, FailFast’s performance improvement from “failing fast” persists.
We note that speculative decoding is inherently more suitable for inference acceleration at smaller batch sizes to begin with and provides diminishing gains as batch size increases, where target model inference becomes less memory-bound and more compute-bound (Sadhukhan et al., 2024; amd, 2025).

## Appendix D Best-Effort Baselines

We evaluate FailFast against a diverse set of strong baselines, covering both single-layer and multi-layer drafters and single-token predictors and multi-token predictors (in each drafter forward pass). We acknowledge that there exist many other speculative decoding methods, e.g., Medusa (Cai et al., 2024), Lookahead Decoding (Fu et al., 2024), etc. Since none of them have been incorporated into high-throughput inference engines (Kwon et al., 2023; Zheng et al., 2023) and EAGLE-3 outperforms them (Li et al., 2025; Xia et al., 2024), we omit those schemes from our evaluations.

### D.1 Fast-dLLM Drafter

Our Fast-dLLM baseline adopts the same speculation length as the AR drafter. To show the full potential of a naive baseline, we also conduct an extensive parameter sweep ($n=$ 3-20) for the optimal speculation length of the dLLM drafter and pick a (different) length that achieves the best speedup for each dataset. The end-to-end speedup is reported in Tab. [4](https://arxiv.org/html/2512.20573v3#A4.T4). This baseline is idealistic because it requires extensive offline profiling to find the optimal tradeoff between the acceptance rate and speculation latency, which is indeterministic because of Fast-dLLM’s confidence-aware parallel decoding. More importantly, we find that on average, picking the best speculation length only brings a 2.3-3.3% performance improvement over using the same speculation length as the AR drafter, so we omit Fast-dLLM+ from the main paper for conciseness.

**Table 4: Comparison of Fast-dLLM (same speculation length as the AR drafter) with Fast-dLLM+ (best speculation length on each dataset from a parameter sweep).**
| Drafting Method | Dataset |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| MATH | AIME | GSM8K | GPQA | HumanEval | Average |  |
| Target Model: Qwen2.5-32B-Instruct |  |  |  |  |  |  |
| Fast-dLLM | ${3.57\times}$ | ${3.29\times}$ | ${3.10\times}$ | ${2.42\times}$ | ${3.16\times}$ | ${3.11\times}$ |
| Fast-dLLM+ | ${3.75\times}$ | ${3.34\times}$ | ${3.10\times}$ | ${2.48\times}$ | ${3.23\times}$ | ${3.18\times}$ |
| Target Model: Qwen2.5-14B-Instruct |  |  |  |  |  |  |
| Fast-dLLM | ${2.57\times}$ | ${2.22\times}$ | ${2.22\times}$ | ${1.86\times}$ | ${2.23\times}$ | ${2.22\times}$ |
| Fast-dLLM+ | ${2.69\times}$ | ${2.28\times}$ | ${2.23\times}$ | ${1.87\times}$ | ${2.40\times}$ | ${2.30\times}$ |

### D.2 SuffixDecoding

SuffixDecoding (Oliaro et al., 2024) represents the state of the art non-parametric drafting method in speculative decoding. It maintains a suffix tree of tokens from previously generated text and the current prompt. If it finds an exact match of the most recent tokens generated, it “speculates” that the LLM will continue the sequence the same way it did in the past. SuffixDecoding only captures wins on the easy regions that exactly repeat the previous context. In contrast, FailFast adaptively determines which subsequences are easy and captures wins on easier segments that aren’t just simply repeating prior context, gaining more wins.

In our evaluations, we find that SuffixDecoding provides negligible speedups over vanilla decoding on the workloads that we evaluate. Fundamentally, SuffixDecoding operates as a pattern-matcher rather than a predictive model; it yields the greatest efficiency gains when the LLM generates sequences identical to those in prior context, such as SQL schemas, boilerplate code, or repetitive reasoning loops. In agentic coding tasks (e.g., SWE-Bench), where an agent iteratively interacts with file systems and error logs, the method achieves high speedups by matching recurring token sequences. Conversely, in “one-shot” benchmarks such as HumanEval, and in mathematical reasoning chains where sequences are unique and lack frequent reflection, the suffix tree finds few matches.

### D.3 EAGLE-3

Figure: (a) 32B training.
Refer to caption: https://arxiv.org/html/2512.20573/x10.png

EAGLE-3 training. There are no public EAGLE-3 weights available for Qwen2.5. Thus, we conducted pretraining of EAGLE-3 weights using the ultrachat-200K (Ding et al., 2023) text corpus in SpecForge (spe, 2025). We adopted the default training hyperparameters in SpecForge and trained EAGLE-3 to match the target model’s output until convergence.
We plot the moving average of the training accuracy in Fig. [9](https://arxiv.org/html/2512.20573v3#A4.F9).
During evaluation, we use the model checkpoints after five epochs of training ($\sim$260k training steps). We find that even though further training improves training-time accuracy, they have a negligible impact on the end-to-end speedup of EAGLE-3.
We open-source our EAGLE-3 model weights and training scripts.

Draft token trees. To ensure a fair and accurate representation of EAGLE-3’s peak performance (Li et al., 2025; eag, 2025), we evaluate EAGLE-3 in SGLang v0.5.6.post2 (Zheng et al., 2023) with a draft token tree, with 50-60 tokens in each tree (following the best default hyperparameter configurations reported in the EAGLE-3 paper (Li et al., 2025), roughly matching $N_{max}$ in FailFast).
For completeness, we additionally evaluate EAGLE-3 configured to propose a single chain of 3-20 draft tokens (rather than a tree). This variant consistently underperforms the full draft-tree version of EAGLE-3, and we therefore omit its results from the evaluation.
Similar to the AR drafter (Fig. [2](https://arxiv.org/html/2512.20573v3#footnote2)), adopting a fixed speculation length of 50-60 tokens in a single chain of draft tokens resulted in a significant slowdown compared to vanilla decoding due to the speculation overhead in autoregressive drafting.
We note that FailFast’s one-step generation is compatible with token trees, which can be realized via a custom attention mask (Agrawal et al., 2025). We leave this extension as future work.

## Appendix E (Failed) Explorations of Alternative Design Choices and Limitations

Dynamic speculation length schemes. Initially, borrowing on ideas in network congestion control (Jacobson, 1988), we tried to use the acceptance rate of previous rounds as an indication of token “easiness” in the current round. However, we found that the notion of easiness appears to be highly local and not well correlated across rounds. This finding prompted us to look forward (ahead into the future, i.e., the current tokens being speculated), not backward.

Reusing previous drafts. In speculative decoding, all tokens following the first rejection are discarded. However, we observe that in many workloads – particularly reasoning chains of thoughts – rejections are often minor corrections (e.g., changing “thus” to “therefore”, as can be observed in the example trajectory in Fig. [12](https://arxiv.org/html/2512.20573v3#A6.F12)), while the subsequent tokens remain of high utility. In traditional speculative decoding, where speculation lengths are relatively short (3–16 tokens), reusing previous drafts offers marginal utility. In contrast, the aggressive speculation length expansion in FailFast makes reuse more beneficial; for example, if out of 60 proposed tokens, we accepted the first 19 tokens and corrected the 20th token, the remaining 40 tokens could be directly plugged back in if the 20th token was the only token that had to be corrected.
Nevertheless, because the drafting process in FailFast is inherently low-latency, the marginal speedups gained from reuse are often secondary.
When these cases do occur, they represent an opportunistic performance gain. For a specific example in the MATH dataset (Fig. [12](https://arxiv.org/html/2512.20573v3#A6.F12)), we implemented a preliminary mechanism for reusing drafts: after each speculation round, we determine if the suffix of the current proposal exists within the tokens rejected in the previous round. If a match is found, the new proposal is appended with the tokens following that suffix.
For instance, if the rejected tokens from the last round are ‘xefghij’ and the current draft is ‘abcdefg’, our new proposal will be ‘abcdefghij’.
For this specific query, reuse provided a $\sim$15% speedup over FailFast, reaching an impressive end-to-end speedup of 8.8$\times$ over vanilla decoding.
However, since most rejections in other queries involve more than simple one-token corrections, the average speedup across the dataset is only $\sim$2%, as the gains are diluted by rounds with fewer reuse opportunities. Consequently, we omit this technique from our primary results.

Data-matched comparisons of parametric drafters. We acknowledge that the training data used for our dLLM drafter and our EAGLE baseline do not exactly match. Our dLLM drafter, Fast_dLLM_v2_1.5B, was adapted from Qwen2.5-1.5B-Instruct via a block-wise diffusion training process (Wu et al., 2025a). As a result, the difference in their performance might be partially attributed to this factor (although the comparisons with AR-drafters are indeed data-matched). Due to the significant amount of resources needed for a rigorous controlled experiment of different drafting methods, where training needs to be data-matched, parameter-matched, and FLOP-matched (Waleffe et al., 2024; Dao & Gu, 2024), we leave it as future work and use this paper as a first step in demonstrating dLLM drafter’s potential.

## Appendix F Visualization of Example Trajectory

Figure: Figure 10: Ground truth trajectory.
Refer to caption: https://arxiv.org/html/2512.20573/x13.png

In this section, we visualize the speculative decoding trajectory of an example query (question 2 from the MATH dataset). Individual tokens are distinguished by alternating background colors. Accepted tokens are shown as standard text. Rejected draft tokens are marked with a strikethrough and reduced opacity. Target-generated tokens (either corrections applied after a rejection or bonus tokens appended after a fully accepted draft) are highlighted with a solid border. This visualization illustrates the precise behavior of different speculative decoding baselines. The question is as follows:

If $f(x)=\frac{3x-2}{x-2}$, what is the value of
$f(-2)+f(-1)+f(0)$? Express your answer as a common fraction.

Figure: Figure 11: Autoregressive drafter’s trajectory.

Figure: Figure 12: FailFast’s trajectory.

In Fig. [10](https://arxiv.org/html/2512.20573v3#A6.F10), we first show the ground-truth trajectory as a reference. Fig. [11](https://arxiv.org/html/2512.20573v3#A6.F11) shows the trajectory of an autoregressive drafter with a static speculation length of 8 tokens. In many segments of the trajectory (e.g., line 11, 17, and 31-39), there are consecutive rounds where the acceptance rate is near-perfect; however, the target model is frequently invoked for verification in these segments, resulting in excessive latency overhead.

Finally, in Fig. [12](https://arxiv.org/html/2512.20573v3#A6.F12), we demonstrate the dynamic speculation length of FailFast in action.
The main win from FailFast’s aggressive speculation length expansion comes from these rounds:

- $\bullet$
Round 4 (line 3), 18/30 accepted (7 drafter forward passes in this round)
- $\bullet$
Round 5 (line 3), 20/20 accepted (6 drafter forward passes in this round)
- $\bullet$
Round 10 (line 14), 18/20 accepted (6 drafter forward passes in this round)
- $\bullet$
Round 11 (line 14), 28/60 accepted (11 drafter forward passes in this round)
- $\bullet$
Round 14 (line 38), 48/60 accepted (12 drafter forward passes in this round)
- $\bullet$
Round 15 (lines 38-40), 23/60 accepted (11 drafter forward passes in this round)
- $\bullet$
Round 16 (lines 42-48), 60/60 accepted (11 drafter forward passes in this round)
- $\bullet$
Round 18 (lines 48-52), 50/50 accepted (9 drafter forward passes in this round)
- $\bullet$
Round 22 (line 55), 60/60 accepted (12 drafter forward passes in this round)
- $\bullet$
Round 23 (line 60), 30/30 accepted (7 drafter forward passes in this round)

We note that the number of actual forward passes deviates from the intuition in Fig. [1](https://arxiv.org/html/2512.20573v3#S1.F1) due to a number of reasons: We do one-pass generation following Fast-dLLM’s default small block size of 8, so generating 10 tokens might span across three small blocks, which requires three forward passes even though we are doing one-pass generation for each small block; occasionally, we need extra forward passes on a full block (e.g., 32 tokens) to populate the KV cache of prior drafted tokens (Wu et al., 2025a).
