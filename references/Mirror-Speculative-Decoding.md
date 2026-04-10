Sections:
Abstract
1 Introduction
2 Speculative Decoding: Formalization and Limits
3 Mirror Speculative Decoding
    3.1 Early-Exit Proxies and Branch-Complete Concurrent Speculation.
    3.2 Draft execution with Speculative Streaming
        Multi-stream attention (MSA) factorization.
        Work-conserving draft generation within Mirror-SD.
    3.3 Heterogeneous Sharding of Mirror-SD
        Target sharding
        Draft sharding.
        Cross-accelerator rendezvous.
    3.4 Latency Analysis
4 Experiments
    4.1 Evaluation protocol
    4.2 Tri-objective analysis with an MT-Bench diagnostic
        Findings.
    4.3 Effectiveness
    4.4 Generalizability across model families
5 Conclusion
Appendix
    Appendix Contents
    Appendix A Related Works
        Speculative decoding with draft models.
        Single-model approaches.
        Dynamic and adaptive decoding.
        Positioning.
    Appendix B Correctness: Acceptance and Distribution
        Acceptance operator (rule-level equivalence).
        Distributional equivalence (when the verified draft path is identically distributed).
        Sufficient condition for equation 12 .
    Appendix C Latency and Communication Analysis
        Comparison to vanilla SD (per step).
    Appendix D Extended Ablations & Empirical Analysis
        D.1 Batching Effects
        D.2 Draft-side speedups with speculative streaming
    Appendix E Fallback Dynamics: Influence of Top- κ \kappa and Early-Exit Depth
        E.1 Setup and definitions
        E.2 Monotonicity in k k
        E.3 Monotonicity in early-exit depth
        E.4 Empirical confirmation
        E.5 Practical recommendation
    Appendix F Additional Experimental Details
        F.1 Target and Draft Sharding
        F.2 Draft Model Configuration
    Appendix G LLM Usage Statement
## Contents
- 1 Introduction
- 2 Speculative Decoding: Formalization and Limits
- 3 Mirror Speculative Decoding
  - 3.1 Early-Exit Proxies and Branch-Complete Concurrent Speculation.
  - 3.2 Draft execution with Speculative Streaming
    - Multi-stream attention (MSA) factorization.
    - Work-conserving draft generation within Mirror-SD.
  - 3.3 Heterogeneous Sharding of Mirror-SD
    - Target sharding
    - Draft sharding.
    - Cross-accelerator rendezvous.
  - 3.4 Latency Analysis
- 4 Experiments
  - 4.1 Evaluation protocol
  - 4.2 Tri-objective analysis with an MT-Bench diagnostic
    - Findings.
  - 4.3 Effectiveness
  - 4.4 Generalizability across model families
- 5 Conclusion
- Appendix
  - Appendix Contents
  - Appendix A Related Works
    - Speculative decoding with draft models.
    - Single-model approaches.
    - Dynamic and adaptive decoding.
    - Positioning.
  - Appendix B Correctness: Acceptance and Distribution
    - Acceptance operator (rule-level equivalence).
    - Distributional equivalence (when the verified draft path is identically distributed).
    - Sufficient condition for equation 12 .
  - Appendix C Latency and Communication Analysis
    - Comparison to vanilla SD (per step).
  - Appendix D Extended Ablations & Empirical Analysis
    - D.1 Batching Effects
    - D.2 Draft-side speedups with speculative streaming
  - Appendix E Fallback Dynamics: Influence of Top- κ \kappa and Early-Exit Depth
    - E.1 Setup and definitions
    - E.2 Monotonicity in k k
    - E.3 Monotonicity in early-exit depth
    - E.4 Empirical confirmation
    - E.5 Practical recommendation
  - Appendix F Additional Experimental Details
    - F.1 Target and Draft Sharding
    - F.2 Draft Model Configuration
  - Appendix G LLM Usage Statement

## Abstract

Abstract Speculative decoding accelerates LLM inference with draft lookahead, but its effectiveness is bottlenecked by autoregressive draft generation: larger drafts improve acceptance yet also increase speculation latency overhead, capping speedup. Existing approaches such as Medusa, Hydra, EAGLE partially address draft inefficiency, but ultimately trade acceptance rates for reduced draft latency, or preserve acceptance at the cost of added overheads that limit scaling. Modern SoCs increasingly integrate heterogeneous accelerators, most commonly GPUs and NPUs with complementary throughput and efficiency characteristics, yet existing approaches are accelerator-agnostic and usually place both draft and target on the same type of device, which leaves cross-accelerator parallelism unused. We introduce Mirror Speculative Decoding (Mirror-SD), which breaks the latency–acceptance tradeoff by launching branch-complete rollouts from early-exit signals in parallel with the target’s suffix and by explicitly mapping computation across heterogeneous accelerators. In this design, the draft speculates forward token continuations for target to verify, while the target speculates correction paths for the draft, creating a bidirectional speculative process. To further reduce draft speculation latency overhead while preserving acceptance semantics, we pair Mirror-SD with speculative streaming (SS) so the draft emits multiple tokens per step. This dual strategy of combining parallel heterogeneous execution and SS pushes speculative decoding closer to its ideal regime of high acceptance while reducing speculation overhead. On SpecBench with server-scale models from 14B to 66B parameters, Mirror-SD consistently delivers realistic end-to-end gains, achieving 2.8 × \times –5.8 × \times wall-time speedups across diverse tasks representing 30% average relative improvement over the strongest baseline, EAGLE3.

### 1 Introduction

Autoregressive (AR) large language models (LLMs) have achieved state-of-the-art performance across a wide spectrum of natural language processing (NLP) tasks, yet their decoding latency remains a fundamental bottleneck, particularly for real-time applications such as interactive dialogue, code generation, and on-device assistants  (brown2020language; pope2023efficiently). Speculative decoding (SD) has emerged as a promising paradigm to mitigate this limitation by coupling a lightweight *draft model* with a larger, high-fidelity *target model* (leviathan2023fast; chen2023accelerating). In the canonical two-model SD framework, the draft model generates candidate tokens which are then verified by the target model in a serial pipeline. While this approach reduces the number of target model invocations, the sequential dependency between draft and target stages limits achievable speedups. Recent works attempt to relax the serial constraints by equipping the target itself with speculative capacity. Medusa  (medusa) equips the target with parallel decoding heads, while EAGLE  (eagle) introduces a dedicated speculation layer.
However, the same trade-off remains: larger speculative modules improve acceptance at the cost of higher draft construction latency, while smaller ones reduce overhead but lower acceptance and limit speedup. A detailed discussion of related approaches is provided in Appendix [A](https://arxiv.org/html/2510.13161v2#A1).

The central challenge of speculative decoding lies in reconciling these competing factors: (i) enabling *parallel execution* of draft and target models to eliminate serial dependencies, (ii) *scaling the draft capacity* to achieve higher acceptance rates without incurring proportional latency overhead, and (iii) designing *bandwidth-efficient communication protocols* that allow draft and target to exchange token-level feedback with minimal synchronization overhead. Achieving this balance reframes speculative decoding from primarily a model-level optimization toward a system-level co-design challenge, opening the path to real-time and efficient LLM inference.

Modern System on Chip (SoC) architectures increasingly feature heterogeneous compute units that combine general-purpose CPUs with specialized accelerators such as GPUs and dedicated neural processing units (NPUs) (jouppi2021ten; intel_ultra_2023; amd_ryzen_ai_2023). This design trend enables efficient partitioning of workloads across compute substrates optimized for different performance and power trade-offs. For instance, Apple’s M-series chips integrate a high-throughput GPU and a dedicated Apple Neural Engine (ANE)  (apple_m2_ultra; apple_m3_2023). Similarly, Qualcomm’s Snapdragon 8 Gen 3 features an Adreno GPU alongside a Hexagon NPU optimized for mixed-precision inference (qualcomm_snapdragon_2023). This architectural heterogeneity motivates a division-of-labor strategy for speculative decoding, wherein the draft model operates on the NPU exploiting its efficiency for approximate inference, while the target model executes on the GPU, which is better suited for high-fidelity, throughput-critical computation. Such partitioning leverages available NPU capacity and reduces contention on the GPU, thereby improving end-to-end latency in multi-accelerator deployments.

In this work, we propose a novel architecture that operationalizes this vision by partitioning speculative decoding across heterogeneous compute units, mapping draft inference onto compute-dense NPUs and target verification onto high-throughput GPUs. This design leverages underutilized accelerator capacity, overlaps execution between models, and employs token-level feedback mechanisms to maximize acceptance while minimizing draft construction latency overhead.

### 2 Speculative Decoding: Formalization and Limits

To ground our discussion, we first formalize standard autoregressive (AR) decoding and speculative decoding (SD), establishing the baseline needed to analyze the limits of SD precisely.

Autoregressive (AR) decoding.
Let $\mathcal{V}$ denote a finite vocabulary. We write $x_{1:m}\in\mathcal{V}^{m}$ for the context of length $m$ and $y_{1:T}\in\mathcal{V}^{T}$ for the response of length $T$ to be generated. A decoder-only AR model with parameters $\theta$ defines the conditional distribution

$$ $p_{\theta}(y_{1:T}\mid x_{1:m})=\prod_{t=1}^{T}p_{\theta}\left(y_{t}\mid x_{1:m},y_{<t}\right),\qquad p_{\theta}(\cdot\mid x_{1:m},y_{<t})=\mathrm{Softmax}\big(W,h_{t}\big),$ (1) $$

where $h_{t}\in\mathbb{R}^{H}$ is the *next-token* representation at position $m+t$, and $W\in\mathbb{R}^{|\mathcal{V}|\times H}$ is the output head mapping hidden states to vocabulary logits (Radford2018ImprovingLU; vaswani2017attention).
Scaling inference of such models often requires distributing computation across multiple devices via tensor parallelism, which partitions per-layer parameters across devices and aggregates partial results with collectives such as AllReduce (HansenPalmus2024CommunicationCF; Li2024TPILLMS7). The per-token latency is then set by the critical path combining local compute and synchronizations.

Speculative decoding (SD).
Speculative decoding augments a *target model* $f_{target}(\cdot\mid\cdot)$ with a computationally cheaper *draft model* $f_{draft}(\cdot\mid\cdot)$  (leviathan2023fast; chen2023accelerating).
At step $t$, conditioned on the verified prefix $(x,y_{<t})$, the draft proposes a $\gamma$-token window

$$ $\hat{y}_{t+1:t+\gamma}\sim f_{draft}\!\left(\cdot\,\middle|\,y_{<t},x\right),$ (2) $$

which the target then verifies left-to-right, producing the largest prefix on which both models agree:

$$ $A_{t}\triangleq\max\Big\{r\in\{0,\dots,\gamma\}:\ \forall j\leq r,\ \hat{y}_{t+j}=\arg\max f_{target}\!\left(\cdot\,\middle|\,y_{<t+j-1},x\right)\Big\}.$ (3) $$

The agreed-upon tokens are committed as $y_{t+1:t+A_{t}}=\hat{y}_{t+1:t+A_{t}}$.
If the draft and target disagree before the end of the window ($A_{t}<\gamma$), the target emits a correction $y_{t+A_{t}+1}$ and decoding resumes from $(x,y_{\leq t+A_{t}})$.
The (window-normalized) *acceptance rate* is

$$ $\rho(\gamma;\phi,\theta)\;=\;\frac{\mathbb{E}[A_{t}]}{\gamma}\;\in[0,1],$ (4) $$

which quantifies the expected fraction of the draft’s proposals that are accepted by the target for window length $\gamma$. Let $T_{\text{draft}}(\gamma;\phi)$ and $T_{\text{target}}(\gamma;\theta)$ denote the wall-times to produce and to verify the window in  [Equations 2](https://arxiv.org/html/2510.13161v2#S2.E2) and [3](https://arxiv.org/html/2510.13161v2#S2.E3) (the latter includes the teacher-forced roll-forward through accepted tokens).
Because verification cannot begin before speculation is available, and the *next* speculation cannot begin before the final acceptance decision at step $t$ is known, the happen-before relation is

$$ $\hat{y}_{t+1:t+\gamma}\;\prec\;\text{(verification at $t$)}\;\prec\;\hat{y}^{\text{next}}_{t+1:t+\gamma},$ $$

yielding a *serial* step latency

$$ $T_{\mathrm{SD}}(\gamma;\phi,\theta)\;=\;T_{\text{draft}}(\gamma;\phi)\;+\;T_{\text{target}}(\gamma;\theta).$ (5) $$

Increasing draft capacity (larger $\gamma$, deeper/wider $f_{d}$) typically *increases* $\rho$ but also increases $T_{\text{draft}}$, while tiny drafts reduce $T_{\text{draft}}$ but suffer low $\rho$ (leviathan2023fast; chen2023accelerating).
[Equation 5](https://arxiv.org/html/2510.13161v2#S2.E5) exposes the core limitation: improvements in acceptance must compensate for the added draft latency, intrinsically coupling acceptance with latency.

Figure: Figure 1: Mirror-SD verification and reuse (example with $\gamma=3,\,\kappa=1$). At early exit, the target (blue) emits $\mathcal{M}_{t}=\{m_{1},\dots,m_{4}\}$ and continues to the final layer. The draft (orange) expands $\mathcal{M}_{t}$ into branch-complete continuations $y^{\prime}_{i0{:}i3}$ (grid). After verification, the target accepts $\hat{y}_{0},\hat{y}_{1}$ and issues correction $y_{2}$ at depth $\tau=2$. Reuse is possible if there exists a precomputed branch whose prefix matches the accepted tokens $(\hat{y}_{0},\hat{y}_{1})$ and whose node at depth $\tau$ equals $y_{2}$ (green). Otherwise, speculation is recomputed (See [Section 3.1](https://arxiv.org/html/2510.13161v2#S3.SS1) for the formal rule).
Refer to caption: x1.png

### 3 Mirror Speculative Decoding

We propose *Mirror Speculative Decoding* (Mirror SD), a systems–algorithm co-design that enables parallel draft-target execution by conditioning the draft on *intermediate* target-layer distributions and reconciling via a bandwidth-light token channel. This section develops the method end-to-end—formal semantics, latency models, and a realizable tensor-parallel implementation.

#### 3.1 Early-Exit Proxies and Branch-Complete Concurrent Speculation.

Consider a target transformer of depth $N$ with layers $\mathcal{L}_{1},\dots,\mathcal{L}_{N}$ and intermediate representations $h_{t}^{(\ell)}$ at step $t$.
Applying the LM head $W_{\mathrm{LM}}$ to an intermediate state yields a proxy next-token distribution

$$ $p^{(\ell)}(\cdot\mid y_{<t},x)\;=\;\mathrm{Softmax}\!\big(W_{\mathrm{LM}}\,h_{t}^{(\ell)}\big),\qquad\ell<N,$ (6) $$

which is typically strongly correlated with the final distribution $p^{(N)}(\cdot\mid y_{<t},x)$  (Pal2023FutureLA).
We designate an *early-exit* layer $\ell_{e}\in\{1,\dots,N-1\}$ and expose a low-bandwidth *token channel*:

$$ $\mathcal{M}_{t}\;=\;\mathrm{Top}\text{-}\kappa\!\big(p^{(\ell_{e})}(\cdot\mid y_{<t},x)\big)\;=\;\{(v_{i},\log\tilde{p}_{i})\}_{i=1}^{\kappa},\qquad v_{i}\in\mathcal{V},$ (7) $$

containing only the top-$\kappa$ candidate tokens and their log-probabilities.
While this message is sent, the target continues its verification pass through $\mathcal{L}_{\ell_{e}+1},\dots,\mathcal{L}_{N}$ to form the full next-token distribution $p^{(N)}(\cdot\mid y_{<t},x)$.
Let $\gamma\in\mathbb{N}$ denote the *speculative window length*.

Given $\mathcal{M}_{t}$, the draft begins a *branch-complete* rollout in parallel: for each candidate $v_{i}$ and for every prefix length $r\leq\gamma$, it prepares a speculative continuation for the *next step* of decoding starting from $v_{i}$,

$$ $\forall i\in\{1,\dots,\kappa\},\ \forall r\in\{1,\dots,\gamma\}:\qquad\hat{y}^{\prime(i)}_{t+1:t+r}\ \sim\ f_{d}\!\left(\cdot\,\middle|\,y_{<t},x,\ \tilde{y}_{t+1}=v_{i}\right).$ (8) $$

While the draft’s batched branches run, the target finishes verification against the currently selected draft path under the standard speculative rule and determines the first mismatch (the *correction*).
Formally, let

$$ $A_{t}\;\triangleq\;\max\Big\{r\in\{0,\dots,\gamma\}:\ \hat{y}_{t+j}=y_{t+j}^{\text{targ}}\ \ \forall\,j\leq r\Big\}$ $$

be the accepted prefix length, where $y_{t+j}^{\text{targ}}$ are the target’s tokens obtained from $p^{(N)}(\cdot\mid y_{<t+j-1},x)$ (greedy/stochastic sampling).
If $A_{t}<\gamma$, the correction occurs at index $\tau\!=\!A_{t}\!+\!1$ with token

$$ $c_{t+\tau}\;\triangleq\;y_{t+\tau}^{\text{targ}}\sim p^{(N)}(\cdot\mid y_{<t+\tau-1},x).$ $$

Let $\mathcal{T}_{t}$ be the hypothesis tree built at early exit from the top-$\kappa$ roots $\{v_{i}\}$, whose nodes at depth $r$ store the token at position $t+r$ and its precomputed continuation.

Verification vs. reuse criterion.
At step $t$, the target accepts a prefix of length $A_{t}$ and issues a correction at $\tau=A_{t}{+}1$ with token $c_{t+\tau}$.
The early-exit message $\mathcal{M}_{t}$ induces a hypothesis tree $\mathcal{T}_{t}$ rooted at the top-$\kappa$ candidates, with $\mathrm{Paths}_{r}(\mathcal{T}_{t})$ denoting all root-to-depth-$r$ prefixes, which serve as anchors for speculative continuations.
The accepted prefix is $\Pi_{t}=(y^{\mathrm{targ}}_{t+1},\ldots,y^{\mathrm{targ}}_{t+A_{t}})$, and the corrected prefix extends it with the correction token, $\Pi_{t}^{+}=(\Pi_{t},\,c_{t+\tau})$.
Reuse occurs whenever this corrected prefix already appears as a path in $\mathcal{T}_{t}$, i.e.

$$ $\Pi_{t}^{+}\in\mathrm{Paths}_{\tau}(\mathcal{T}_{t}),$ $$

so that only the correction must be checked while the accepted positions $1{:}A_{t}$ remain fixed.

Operational selection of the next window.

$$ $\hat{y}^{\prime}_{t+1:t+\gamma}=\begin{cases}\text{branch rooted at }c_{t+1},&A_{t}=0\ \land\ \exists\,i:\ v_{i}=c_{t+1},\[4.0pt] \text{precomputed continuation at depth }\tau\ \text{along }\Pi_{t},&A_{t}\geq 1\ \land\ \Pi_{t}^{+}\in\mathrm{Paths}_{\tau}(\mathcal{T}_{t}),\[4.0pt] \text{fresh rollout from }(y_{1:t+A_{t}},\,c_{t+\tau}),&\text{otherwise.}\end{cases}$ $$

In all cases, the committed output is $y^{\mathrm{targ}}_{t+1:t+A_{t}}$, after which decoding advances to the next step.

Effect of sampling width at early exit.
Let $q(\cdot)=p^{(N)}(\cdot\mid h_{t})$ and $\tilde{p}(\cdot)=p^{(\ell_{e})}(\cdot\mid h_{t})$.
We denote the top-$\kappa$ mass overlap as:

$$ $\Omega_{\kappa}\;=\;\sum_{y\in\mathrm{Top}\text{-}\kappa(\tilde{p})}q(y).$ (9) $$

It follows that $\mathbb{P}\!\big(y_{t+1}\in\mathrm{Top}\text{-}\kappa(\tilde{p})\big)=\Omega_{\kappa}$, which is nondecreasing in $\kappa$ and satisfies $\lim_{\kappa\to|\mathcal{V}|}\Omega_{\kappa}=1$.
Larger $\kappa$ therefore reduces fallbacks requiring speculation recomputation and improves throughput, while leaving acceptance semantics intact (See  [Appendix B](https://arxiv.org/html/2510.13161v2#A2)).

#### 3.2 Draft execution with Speculative Streaming

For the draft model $f_{d}$, we employ *Speculative Streaming* (SS) (bhendawade2024speculative), a speculative mechanism that *verifies* previously proposed tokens while *generating* new speculative tokens *in the same forward pass* using multi-stream attention. Applying SS to the target would modify its decoding dynamics and alter the final distribution $p^{(N)}(\cdot\mid y_{<t},x)$ (bhendawade2024speculative), breaking the lossless guarantee established in  [Appendix B](https://arxiv.org/html/2510.13161v2#A2). In contrast, using SS on the draft accelerates speculation generation without changing acceptance semantics, since all commitments still require verification against the unchanged target. This design leverages SS precisely where it yields additional concurrency while preserving correctness (See  [Appendix B](https://arxiv.org/html/2510.13161v2#A2)).  [Section D.2](https://arxiv.org/html/2510.13161v2#A4.SS2) illustrates the SS mechanism and compares draft-only speedups between vanilla and SS drafts.

##### Multi-stream attention (MSA) factorization.

Let $M_{t}^{(\ell)}$ denote the main-stream hidden state at layer $\ell$ and step $t$, and $S_{t,j}^{(\ell)}$ the hidden state of lookahead stream $j\!\in\!\{1,\dots,\gamma\}$. Speculative streaming (SS) constructs attention masks so that each $S_{t,j}$ attends to the verified prefix and to lower-index lookahead streams $\{S_{t,1},\dots,S_{t,j}\}$, while the main stream $M_{t}$ attends only to the verified prefix. At the top layer, a *shared* LM head $W_{\mathrm{LM}}^{(d)}$ projects these hidden states to token logits:

$$ $W_{\mathrm{LM}}^{(d)}\,M_{t}^{(N)}\;\mapsto\;p_{d}(\cdot\mid h_{t})\quad\text{and}\quad W_{\mathrm{LM}}^{(d)}\,S_{t,j}^{(N)}\;\mapsto\;p_{d}(\cdot\mid h_{t},\,j),\ \ j=1,\dots,\gamma.$ $$

so a single forward pass yields both the distribution used to *verify* the prior draft and the distributions needed to *grow* the next speculative window across multiple lookahead depths. SS trains these streams with a future $n$-gram prediction objective without introducing additional heads.

##### Work-conserving draft generation within Mirror-SD.

Within each Mirror-SD step, the draft must furnish a branch-complete speculative window of length $\gamma$ at the rendezvous ( [Section 3.1](https://arxiv.org/html/2510.13161v2#S3.SS1)). Under SS, a single draft *internal* step can emit $\eta_{j}\!\geq\!1$ tokens by verifying the prior proposal and predicting multiple future tokens in one pass (bhendawade2024speculative). Consequently, the number of draft steps $J$ required to materialize $\gamma$ tokens satisfies

$$ $J\leq\Big\lceil\tfrac{\gamma}{\bar{\eta}}\Big\rceil,\qquad\bar{\eta}=\tfrac{1}{J}\sum_{j=1}^{J}\eta_{j}.$ $$

Figure: Figure 2: Heterogeneous sharding in Mirror-SD. The *target* (blue) uses Megatron-style TP with two collectives per MHA/MLP block, while the *draft* (orange) uses SPD-style sharding across $G_{D}$ NPUs with only two synchronizations per step. This design reduces sync cost, enlarges draft capacity, and improves acceptance without raising critical-path latency. *Note:* The beige bands labeled “All-Reduce Draft + Target” are a visual shorthand: the draft and target perform *separate* all-reduces within their own device groups, with no cross-collective coupling.
Refer to caption: x2.png

#### 3.3 Heterogeneous Sharding of Mirror-SD

We co-schedule a depth–$N$ *target* on $G_{T}{=}8$ GPUs and a depth–$N_{\mathrm{d}}$ *draft* on $G_{D}{=}8$ NPUs. The target is a pre-trained model and thus kept in its standard Megatron-style tensor parallel (TP) form (Shoeybi2019MegatronLMTM), ensuring compatibility with existing inference stacks and KV-cache layouts. In contrast, the draft is trained from scratch using the SPD architecture  (Kim2025SPDSD) and deployed on NPUs. We write $S$ for per–microbatch sequence length, $B$ for microbatch size, and $|\mathcal{V}|$ for vocabulary size.  [Figure 2](https://arxiv.org/html/2510.13161v2#S3.F2) illustrates the heterogeneous sharding setup with an example configuration (target of 8 layers, draft of 4 layers); in practice, both target and draft may use different depths based on the experiment configuration.

##### Target sharding

We use Megatron-style TP on the target: column-parallel $W_{qkv}$ and $W_{o}$ in MHA, and column/row-parallel $W_{1},W_{2}$ in the MLP. Each transformer block performs the standard two TP collectives (attention and MLP). At early exit $\ell_{e}$, the target emits $\mathrm{Top}\text{-}\kappa\!\big(p^{(\ell_{e})}\big)$ over the token channel while continuing the verification phase; acceptance remains decided against $p^{(N)}$ and is therefore unchanged relative to vanilla SD (See  [Appendix B](https://arxiv.org/html/2510.13161v2#A2)).

##### Draft sharding.

The draft is trained with SPD architecture  (Kim2025SPDSD). We divide the $N_{\mathrm{d}}$ layers into two contiguous segments. Within each segment we instantiate $G_{D}$ parallel tracks; track $g\in\{1,\dots,G_{D}\}$ is pinned to NPU $g$ and advances through all layers of its segment using a resident weight shard. There is no inter-NPU traffic inside a segment (See  [Figure 2](https://arxiv.org/html/2510.13161v2#S3.F2)). At the segment boundary, all tracks perform a single global synchronization to re-align tensor partitions, and a second synchronization occurs at the end of the forward pass to assemble full-width logits for the main and lookahead streams. Each internal draft step executes two all-reduce collectives on activation shards while weights remain sharded. This replaces per-layer synchronization with a fixed two-collective cost, reducing latency and enabling more parameters to be sharded across NPUs. In practice, this expands draft capacity and improves acceptance rates $\rho(\gamma;\phi,\theta)$ without increasing critical-path latency.

##### Cross-accelerator rendezvous.

Mirror-SD performs two token-level exchanges per step: early-exit ($\ell_{e}$) and final verification ($N$). These exchanges carry $O(B\,\kappa)$ small items (IDs and log-probabilities) and are negligible in practice (microseconds) compared to millisecond-scale target/draft compute; they are accounted for by $T_{\mathrm{rv}}$ in the latency model.

#### 3.4 Latency Analysis

Let the target early-exit at layer $\ell_{e}$ in a depth-$N$ stack with per–layer times $c_{\ell}$, and write

$$ $T_{\text{target}}^{\,1:\ell_{e}}=\sum_{\ell=1}^{\ell_{e}}c_{\ell},\qquad T_{\text{target}}^{\,\ell_{e}+1:N}=\sum_{\ell=\ell_{e}+1}^{N}c_{\ell}.$ $$

Let $\gamma$ be the speculative window length and let $T_{\text{draft}}^{\mathrm{gen}}(\gamma)$ denote the time to produce a branch-complete draft window (absorbing any multi-token SS steps). We account for the two rendezvous overheads at early exit and final verification,

$$ $T_{\mathrm{rv}}^{(\mathrm{ee})},\qquad T_{\mathrm{rv}}^{(\mathrm{fv})},\qquad T_{\mathrm{rv}}\triangleq T_{\mathrm{rv}}^{(\mathrm{ee})}+T_{\mathrm{rv}}^{(\mathrm{fv})},$ $$

where the GPU$\leftrightarrow$NPU token exchanges carry only $O(B\kappa)$ IDs/log-probabilities.

A single Mirror-SD step consists of (i) target prefix, (ii) early-exit rendezvous, (iii) a parallel region where the target suffix overlaps the draft generation, and (iv) final rendezvous. The step latency is

$$ $T_{\mathrm{Mirror}}=T_{\text{target}}^{\,1:\ell_{e}}+T_{\mathrm{rv}}^{(\mathrm{ee})}+\max\!\big\{T_{\text{target}}^{\,\ell_{e}+1:N},\,T_{\text{draft}}^{\mathrm{gen}}(\gamma)\big\}+T_{\mathrm{rv}}^{(\mathrm{fv})}.$ (10) $$

Let the *overlap budget* be $\Delta\triangleq T_{\text{target}}^{\,\ell_{e}+1:N}$. If $T_{\text{draft}}^{\mathrm{gen}}(\gamma)\leq\Delta$, the entire draft generation is hidden under the target suffix and

$$ $T_{\mathrm{Mirror}}=T_{\text{target}}+T_{\mathrm{rv}}.$ $$

Otherwise the draft dominates the parallel region and

$$ $T_{\mathrm{Mirror}}=T_{\text{target}}^{\,1:\ell_{e}}+T_{\mathrm{rv}}^{(\mathrm{ee})}+T_{\text{draft}}^{\mathrm{gen}}(\gamma)+T_{\mathrm{rv}}^{(\mathrm{fv})}.$ $$

Thus, scaling the draft that only increases overlapped $T_{\text{draft}}^{\mathrm{gen}}(\gamma)$ is *free* up to budget $\Delta$, while the token-channel transfers remain a small $O(B\kappa)$ term. We provide a full accounting of sampling/transfer costs, multi-step SS, and synchronization in  [Appendix C](https://arxiv.org/html/2510.13161v2#A3).

### 4 Experiments

We evaluate Mirror-SD on a broad suite of generation workloads under realistic serving constraints, using server-scale decoder-only LLMs that are routinely deployed in production inference stacks across mid to large capacities, and we compare against strong speculative-decoding baselines.

#### 4.1 Evaluation protocol

Datasets and tasks. We integrate our approach with the open-source SpecBench framework (xia-etal-2024-unlocking) to ensure a fair, reproducible comparison against prior methods. SpecBench provides standardized prompts and pre/post-processing, sampling settings and released configs and seeds (xia-etal-2024-unlocking). We report results on multi-turn interactive conversation (MT Bench), translation, summarization, mathematical reasoning, machine translation and retrieval-augmented generation (RAG). Context and generation lengths follow the SpecBench protocol (xia-etal-2024-unlocking).

Models and baselines.
We evaluate Mirror-SD on server-scale targets that are deployable in production inference stacks: Qwen3-14B and Qwen3-32B (qwen3technicalreport), Mistral-24B (mistral_small_3_2501_2025), and OPT-66B (zhang2022opt).
For Qwen targets, we train a 0.6B-parameter draft with 2 segments and 8 tracks and deploy it on 8 NPUs as described in [Section 3.3](https://arxiv.org/html/2510.13161v2#S3.SS3).
For Mistral we train a 0.5B draft, and for OPT we train a 200M draft, both sharded as in [Section 3.3](https://arxiv.org/html/2510.13161v2#S3.SS3) to optimize synchronization cost.
All draft models are trained with SS objective described in (bhendawade2024speculative) on UltraChat  (ding2023enhancing). Across all target models, drafts are launched from the mid-layer early exit ($\frac{1}{2}$ of total depth) with top-$\kappa$=8 under batch size 1. Please refer to [Appendix E](https://arxiv.org/html/2510.13161v2#A5) for the effects of early-exit depth and $\kappa$. Baselines include vanilla SD, Medusa (Cai2024MedusaSL), Hydra (hydra), EAGLE 2/3 (eagle2; eagle3), Recycling (luo2024turning), PLD (saxena2023pld), SpS (gante2023assisted), REST (he-etal-2024-rest), and Lookahead (fu2023lookahead). All baselines have public implementations in SpecBench (xia-etal-2024-unlocking), and we use the corresponding implementations.

Metrics.
We focus solely on efficiency, without reporting accuracy metrics, since Mirror-SD is lossless and guarantees identical outputs to the target model under the same decoding process (see  [Appendix B](https://arxiv.org/html/2510.13161v2#A2)). Our two key metrics are:
(i) end-to-end wall-time speedup over target-only autoregressive decoding, reported as a speedup factor; and
(ii) *acceptance length*, the expected number of tokens accepted per speculative window, averaged across steps and prompts. We report greedy decoding with temperature $\tau=0$ and stochastic decoding with $\tau=1$. The same decoding hyperparameters are used for all methods.

Serving configuration and reproducibility.
Target models are distributed across eight M2 Ultra GPUs using Megatron-style tensor parallelism ([Section 3.3](https://arxiv.org/html/2510.13161v2#S3.SS3)), while the draft runs on eight NPUs (apple_m2_ultra). All evaluations use a fixed batch size of 1 and speculative window length $\gamma{=}7$; please refer to [Section D.1](https://arxiv.org/html/2510.13161v2#A4.SS1) for analysis of batching effects. The token channel transmits only the top-$\kappa$ token IDs and log-probabilities in bf16. For determinism, interconnects are pinned and frequency scaling is disabled. Timings include compute, collectives, and rendezvous overhead.

#### 4.2 Tri-objective analysis with an MT-Bench diagnostic

Speculative decoding couples three quantities: the speculative window $\gamma$, the acceptance length $\mathbb{E}[A_{t}]=\gamma\,\rho(\gamma;\phi,\theta)$, and the per-speculation-step latency. In vanilla SD, enlarging $\gamma$ typically boosts acceptance but also increases draft construction time, yielding an upward-sloping latency curve. For Mirror-SD, the step latency follows the model in  [Section 3.4](https://arxiv.org/html/2510.13161v2#S3.SS4) ([Equation 10](https://arxiv.org/html/2510.13161v2#S3.E10)): as long as $T_{\text{draft}}^{\mathrm{gen}}(\gamma)\!\leq\!\Delta$ with $\Delta=T_{\text{target}}^{\,\ell_{e}+1:N}$, increasing $\gamma$ (and thus $\mathbb{E}[A_{t}]$) adds *no* marginal latency; once $T_{\text{draft}}^{\mathrm{gen}}(\gamma)\!>\!\Delta$, latency grows by the excess beyond $\Delta$. Acceptance semantics remain unchanged ( [Appendix B](https://arxiv.org/html/2510.13161v2#A2)). We validate these hypotheses on MT-Bench (bai-etal-2024-mt) by sweeping $\gamma$, measuring $\mathbb{E}[A_{t}]$ and the observed draft construction time, and comparing three methods that share the same target: (i) vanilla SD with autoregressive drafts from 12M to 1.7B parameters, (ii) Mirror-SD with a $0.6$M draft, and (iii) Mirror-SD with a speculative-streaming draft ([Section 3.2](https://arxiv.org/html/2510.13161v2#S3.SS2)) of $0.6$B.  [Figure 3(a)](https://arxiv.org/html/2510.13161v2#S4.F3.sf1) places $\gamma$ on the x-axis, $\mathbb{E}[A_{t}]$ on the y-axis, and draft construction latency on the z-axis.

##### Findings.

Vanilla SD traces an ascending surface: larger drafts increase $\mathbb{E}[A_{t}]$ but raise step latency commensurately. Mirror-SD shifts this surface downward by overlapping draft generation on NPUs with target verification on GPUs, revealing a near-zero-slope regime wherever $T_{\text{draft}}^{\mathrm{gen}}(\gamma)\!\leq\!\Delta$. Adding speculative streaming further reduces $T_{\text{draft}}^{\mathrm{gen}}(\gamma)$ by requiring fewer internal draft steps $J$ to cover the same window length $\gamma$, which extends the near-zero-slope region and pushes the surface down again. Across $\gamma$, Mirror-SD and Mirror-SD+SS dominate the Pareto frontier—achieving higher $\mathbb{E}[A_{t}]$ at a given latency, lower latency at a given $\mathbb{E}[A_{t}]$, and a wider feasible range before saturating the overlap budget defined in  [Section 3.4](https://arxiv.org/html/2510.13161v2#S3.SS4).

Figure: (a) Tri-objective diagnostic on MT-Bench.
Refer to caption: x3.png

**Table 1: SpecBench wall-time speedups. Mirror-SD outperforms prior methods across models, tasks, and decoding temperatures, showing consistent improvements.**
| Model | Task | EAGLE3 | EAGLE2 | Hydra | Recycling | Medusa | Vanilla-SD | PLD | SpS | REST | Lookahead | Mirror-SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-14B (T=0) | Translation | 2.53x | 1.98x | 2.03x | 1.86x | 1.65x | 2.34x | 1.18x | 1.15x | 1.21x | 1.09x | 4.13x |
| Summarization | 2.91x | 2.19x | 2.00x | 2.30x | 1.55x | 1.76x | 2.12x | 1.87x | 1.38x | 1.30x | 3.07x |  |
| Question Answering | 3.09x | 2.39x | 2.19x | 2.13x | 1.62x | 1.81x | 1.14x | 1.31x | 1.61x | 1.27x | 3.18x |  |
| Mathematical Reasoning | 3.36x | 2.75x | 2.53x | 2.58x | 2.12x | 2.80x | 1.67x | 1.59x | 1.15x | 1.70x | 5.32x |  |
| Retrieval Aug. Generation | 2.66x | 2.13x | 2.04x | 2.06x | 1.64x | 2.02x | 1.67x | 1.75x | 1.57x | 1.32x | 3.49x |  |
| Multi-turn Conversation | 3.29x | 3.05x | 2.45x | 2.44x | 1.93x | 2.07x | 1.63x | 1.81x | 1.49x | 1.35x | 3.70x |  |
| Qwen3-14B (T=1) | Translation | 1.92x | 1.81x | 1.81x | 1.78x | 1.54x | 2.19x | 1.07x | 1.04x | 1.08x | 1.03x | 3.89x |
| Summarization | 2.84x | 2.05x | 1.66x | 1.84x | 1.40x | 1.50x | 1.86x | 1.40x | 1.20x | 1.13x | 2.81x |  |
| Question Answering | 2.61x | 2.00x | 1.85x | 1.84x | 1.37x | 1.36x | 1.04x | 1.18x | 1.28x | 1.15x | 2.80x |  |
| Mathematical Reasoning | 3.25x | 2.54x | 2.42x | 2.29x | 2.01x | 2.53x | 1.49x | 1.42x | 1.05x | 1.39x | 5.02x |  |
| Retrieval Aug. Generation | 2.53x | 1.86x | 1.59x | 1.89x | 1.47x | 1.68x | 1.56x | 1.60x | 1.30x | 1.07x | 2.95x |  |
| Multi-turn Conversation | 3.05x | 2.78x | 2.16x | 2.15x | 1.81x | 1.98x | 1.42x | 1.41x | 1.37x | 1.24x | 3.48x |  |
| Qwen3-32B (T=0) | Translation | 2.52x | 2.10x | 2.14x | 1.57x | 1.56x | 2.74x | 1.09x | 1.24x | 1.15x | 1.12x | 3.72x |
| Summarization | 2.98x | 2.59x | 1.98x | 1.98x | 1.56x | 2.07x | 1.82x | 1.62x | 1.38x | 1.26x | 3.14x |  |
| Question Answering | 2.76x | 2.26x | 2.17x | 1.63x | 1.81x | 2.06x | 1.17x | 1.59x | 1.70x | 1.13x | 3.04x |  |
| Mathematical Reasoning | 3.77x | 3.49x | 2.52x | 1.95x | 2.23x | 3.33x | 1.68x | 1.70x | 1.33x | 1.49x | 5.84x |  |
| Retrieval Aug. Generation | 2.65x | 2.22x | 1.92x | 1.61x | 1.59x | 2.33x | 1.42x | 1.69x | 1.76x | 1.15x | 3.42x |  |
| Multi-turn Conversation | 3.29x | 3.24x | 2.75x | 1.79x | 1.92x | 2.67x | 1.53x | 1.65x | 1.63x | 1.33x | 3.59x |  |
| Qwen3-32B (T=1) | Translation | 2.36x | 1.79x | 1.90x | 1.40x | 1.42x | 2.43x | 1.03x | 1.09x | 1.03x | 1.05x | 3.15x |
| Summarization | 2.79x | 2.22x | 1.75x | 1.48x | 1.45x | 1.92x | 1.59x | 1.43x | 1.16x | 1.17x | 2.92x |  |
| Question Answering | 2.34x | 2.09x | 1.72x | 1.46x | 1.61x | 1.89x | 1.04x | 1.37x | 1.44x | 1.04x | 2.90x |  |
| Mathematical Reasoning | 3.45x | 3.13x | 2.35x | 1.80x | 1.66x | 2.88x | 1.36x | 1.59x | 1.20x | 1.28x | 5.08x |  |
| Retrieval Aug. Generation | 2.34x | 1.96x | 1.79x | 1.50x | 1.35x | 2.08x | 1.28x | 1.35x | 1.48x | 1.07x | 3.33x |  |
| Multi-turn Conversation | 3.14x | 2.58x | 2.29x | 1.63x | 1.73x | 2.39x | 1.34x | 1.48x | 1.47x | 1.17x | 3.28x |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |

#### 4.3 Effectiveness

[Table 1](https://arxiv.org/html/2510.13161v2#S4.T1) reports end-to-end wall-time speedups across SpecBench (xia-etal-2024-unlocking) tasks. A clear pattern emerges: Mirror-SD shows improvements over baselines across model sizes, temperatures, and workloads. On Qwen3-14B, Mirror-SD averages $3.8\times$ acceleration with greedy sampling, compared to $2.97\times$ for the strongest prior methods; on Qwen3-32B, the average rises to $3.78\times$, eclipsing baselines at roughly $3\times$. The gains are most pronounced on long-horizon workloads (e.g., mathematical reasoning), where Mirror-SD reaches up to $5.84\times$ speedup. The improvement is driven primarily by a larger acceptance length $\mathbb{E}[A_{t}]$: Mirror-SD lets us scale the draft and apply speculative streaming without paying proportional step latency, which increases the number of tokens committed per target step. Since throughput scales roughly with the expected tokens accepted per step, $S\propto 1+\mathbb{E}[A_{t}]$, these acceptance gains translate directly into wall-time speedups. Retrieval-augmented generation shows a similar effect, benefitting from stable intermediate distributions that allow the draft to sustain long accepted prefixes. Even on high-entropy domains such as multi-turn conversation, where acceptance is intrinsically harder, Mirror-SD consistently delivers $3.3$–$3.7\times$ acceleration compared to the $1.8$-$2.4\times$ range of Hydra, Recycling or Medusa. In translation and QA, the margin is steadier but no less striking: Mirror-SD maintains a speedup edge across both greedy and stochastic decoding, validating that its improvements are insensitive to decoding regime. For an intuition grounded in the concurrency model and scaling laws behind [Figure 3(a)](https://arxiv.org/html/2510.13161v2#S4.F3.sf1), see [Appendix C](https://arxiv.org/html/2510.13161v2#A3).

Figure: (a) OPT
Refer to caption: x5.png

#### 4.4 Generalizability across model families

To test whether the gains of Mirror-SD extend beyond Qwen, we repeat the study on two server-scale decoder-only families: Mistral-24B and OPT-66B. For each target, we hold decoding hyperparameters and draft capacity fixed and compare four variants: (1) standard speculative decoding with an autoregressive draft, (2) standard speculative decoding with a speculative-streaming draft, (3) Mirror-SD with an autoregressive draft, and (4) Mirror-SD with a speculative-streaming draft. [Figure 4](https://arxiv.org/html/2510.13161v2#S4.F4) reports end-to-end speedups over target-only decoding for translation, summarization, and multi-turn conversation under $\tau=0$ and $\tau=1$ regimes. Across both families and all tasks, the vanilla SD baseline with autoregressive-draft generation yields the smallest gains; adding speculative streaming increases throughput; switching to Mirror-SD produces a further jump; combining Mirror-SD with speculative streaming delivers the largest speedups. This progression matches the analysis in  [Sections 3.2](https://arxiv.org/html/2510.13161v2#S3.SS2) and [3.4](https://arxiv.org/html/2510.13161v2#S3.SS4): Mirror-SD shortens the critical path by overlapping draft generation with the target suffix, while speculative streaming reduces the draft generation time $T_{\text{draft}}^{\text{gen}}(\gamma)$ by emitting multiple tokens per internal draft step. Together, these effects allow larger acceptance lengths $E[A_{t}]$ without additional step latency until the overlap budget is reached, and the target’s output distribution remains unchanged by construction. These results show that pairing Mirror-SD with a speculative-streaming draft generalizes across model families, delivering higher throughput without altering the base architecture or quality.

### 5 Conclusion

We introduced *Mirror Speculative Decoding* (Mirror-SD), a systems–algorithm co-design that overlaps target and draft computation, reduces draft synchronizations, and confines cross-accelerator traffic to a lightweight token channel. Deployed on heterogeneous GPU–NPU setups, Mirror-SD consistently accelerates decoding by 2.8X to 5.8X while preserving correctness. By reducing serial bottlenecks and leveraging multi-accelerator SoCs, Mirror-SD demonstrates a practical low-latency approach for large-scale LLM serving.

## Appendix

### Appendix Contents

### Appendix A Related Works

##### Speculative decoding with draft models.

The original speculative decoding paradigm accelerates autoregressive generation by pairing a small, fast *draft* model with a larger *target* model, which verifies proposed tokens (chen2023accelerating; leviathan2023fast). This approach achieves substantial wall-time savings whenever the draft is hardware-efficient and closely aligned with the target. Domain-specialized drafts trained via distillation further improve acceptance in task-specific settings (hong2025training). Recent variants explore parallelization strategies, such as batch-axis speculation (sun2023spectr) and tree-structured drafts (miao2023specinfer; spector2023accelerating), to raise acceptance rates and amortize draft cost.

##### Single-model approaches.

An alternative line of work removes the explicit draft model and equips the target itself with speculative capacity.
Medusa predicts multiple tokens in parallel via extra heads (medusa), while Hydra enforces autoregressive coupling across those heads to raise acceptance (hydra).
EAGLE introduces a dedicated speculation layer (eagle), with EAGLE-2 enabling dynamic tree retries (eagle2) and EAGLE-3 moving to token-level prediction with multi-layer fusion (eagle3).
Prompt-lookup decoding (PLD) and Lookahead propose suffixes by retrieval rather than generation (saxena2023pld; fu2023lookahead), which is effective when prefix–continuation correlations are strong.
Recycling reduces wasted work by reusing intermediate activations when speculative branches are invalidated, instead of recomputing full forwards (luo2024turning). Other recent advances include structured or retrieval-based decoding policies (yi2024generation; he-etal-2024-rest). Across the single-model designs, speculative capacity is integrated into the target stack, so larger or wider modules increase acceptance but still add work on the target’s critical path; by contrast, Mirror-SD runs draft and target on heterogeneous devices and overlaps draft within the target’s suffix window, converting added draft capacity into acceptance gains without inflating per-step latency proportionally.

##### Dynamic and adaptive decoding.

Beyond speculation, a range of methods accelerate inference by adapting compute during decoding.
CALM (schuster2022confident) and related early-exit methods reduce cost by exiting tokens at shallow layers, while skip decoding (SkipDecode) mitigates key-value cache mismatch via position-dependent layer skipping.
Mixture-of-Depths (MoD) (raposo2024mixture) routes only a subset of tokens through full blocks, yielding non-uniform FLOP allocation.
Other strategies include token merging (bolya2023token) to reduce sequence length dynamically, adaptive span models (sukhbaatar2019adaptive) that learn context windows per token, and CoLT5 (ainslie2023colt5) which routes tokens through heavy or light pathways.
More recently, M2R2 (bhendawade2025m2r2) introduces accelerated residual streams to improve early alignment and efficiency.
Together, these approaches trade fixed per-token compute for dynamic allocation, complementing speculative decoding’s strategy of parallelizing token generation.

##### Positioning.

Mirror-SD builds on these advances but takes a distinct perspective: it is a systems–algorithm co-design aimed at minimizing the *critical path* in speculative decoding. By launching drafts from intermediate target layers, overlapping draft and target compute, and confining cross-accelerator communication to lightweight token exchanges, Mirror-SD complements prior algorithmic improvements and makes speculation more effective in heterogeneous GPU–NPU deployments.

### Appendix B Correctness: Acceptance and Distribution

Let $\gamma$ be the speculative window length, $N$ the number of transformer layers in the target, and let $A_{t}\!\in\!\{0,\dots,\gamma\}$ denote the accepted-prefix length at step $t$.
Recall that the target’s final next-token distribution is $p^{(N)}(\cdot\mid h_{\cdot})$ and that verification commits the longest prefix of the draft that matches the target’s tokens.

##### Acceptance operator (rule-level equivalence).

For any realized draft proposal $\hat{y}_{t+1:t+\gamma}$ and realized target tokens $y^{\mathrm{target}}_{t+1:t+\gamma}$ (obtained by rolling the target with teacher forcing along the agreed prefix and stopping at the first mismatch), both vanilla SD and Mirror-SD compute

$$ $A_{t}\;=\;\max\Big\{r\leq\gamma:\ \hat{y}_{t+j}=y^{\mathrm{target}}_{t+j}\ \ \forall j\leq r\Big\}.$ (11) $$

[Equation 11](https://arxiv.org/html/2510.13161v2#A2.E11) is the *same* acceptance operator in both algorithms: Mirror-SD never commits a token that was not verified against $p^{(N)}$, and any commit is exactly the longest verified prefix. Thus, Mirror-SD changes only the *schedule* by which draft proposals are produced (overlapping with target compute), not the acceptance rule.

##### Distributional equivalence (when the verified draft path is identically distributed).

Fix the models $(f_{draft},f_{target})$ and window $\gamma$.
Let $\mathcal{C}_{t}$ be the decoding context at step $t$ (prompt and previously committed tokens), and let $\zeta_{draft},\zeta_{target}$ collect all random seeds for draft and target sampling.
Define the function

$$ $\mathcal{S}(\hat{y}_{t+1:t+\gamma},\,y^{\mathrm{target}}_{t+1:t+\gamma})\;=\;\max\{r\leq\gamma:\ \hat{y}_{t+j}=y^{\mathrm{target}}_{t+j}\ \forall j\leq r\},$ $$

so that $A_{t}=\mathcal{S}(\hat{y},y^{\mathrm{target}})$ in both procedures.

Assume the draft sequence actually presented to verification in Mirror-SD, denoted $\hat{y}^{\mathrm{Mir}}_{t+1:t+\gamma}$, has the same conditional distribution as the vanilla draft sequence $\hat{y}^{\mathrm{Van}}_{t+1:t+\gamma}$ given $\mathcal{C}_{t}$:

$$ $\hat{y}^{\mathrm{Mir}}_{t+1:t+\gamma}\ \stackrel{{\scriptstyle d}}{{=}}\ \hat{y}^{\mathrm{Van}}_{t+1:t+\gamma}\ \mid\ \mathcal{C}_{t}.$ (12) $$

Then, under a common coupling of $(\zeta_{d},\zeta_{t})$,

$$ $\displaystyle\mathbb{P}_{\mathrm{Mirror}}\!\left(A_{t}=r\right)$ $\displaystyle=\mathbb{P}\!\big(\mathcal{S}(\hat{y}^{\mathrm{Mir}},y^{\mathrm{targ}})=r\big)$ (13) $\displaystyle=\mathbb{P}\!\big(\mathcal{S}(\hat{y}^{\mathrm{Van}},y^{\mathrm{targ}})=r\big)$ $\displaystyle=\mathbb{P}_{\mathrm{Vanilla}}\!\left(A_{t}=r\right),\quad\forall r\in\{0,\dots,\gamma\}.$ $$

Hence the acceptance-rate statistic
$\rho(\gamma;\phi,\theta)=\mathbb{E}[A_{t}]/\gamma$
coincides between Mirror-SD and vanilla SD.

##### Sufficient condition for equation 12 .

Condition equation [12](https://arxiv.org/html/2510.13161v2#A2.E12) holds if the draft path used for verification in Mirror-SD is sampled from $f_{draft}(\cdot\mid h_{t})$ exactly as in vanilla SD, or more generally if the branch-selection policy induces the same conditional law for the verified draft sequence as vanilla SD.
Under this mild parity condition, Mirror-SD is *distributionally* identical to vanilla SD with respect to $A_{t}$, while still enjoying the latency benefits of overlapping draft computation with the target’s suffix.

Figure: (a) Qwen-14B
Refer to caption: x7.png

### Appendix C Latency and Communication Analysis

This appendix consolidates the latency model of Mirror-SD with its tensor-parallel (TP) communication costs.

Draft and Target Latencies
Within one Mirror-SD step, the draft may take $J\!\geq\!1$ *internal* steps. With speculative streaming (SS), step $j$ emits $\eta_{j}\!\geq\!1$ tokens so that $\sum_{j=1}^{J}\eta_{j}\geq\gamma$, with average $\bar{\eta}=\tfrac{1}{J}\sum_{j}\eta_{j}$ and

$$ $T_{\text{draft}}^{\mathrm{gen}}(\gamma)=\sum_{j=1}^{J}(u^{\mathrm{d}}_{j}+s^{\mathrm{d}}_{j}),\qquad J\leq\Big\lceil\tfrac{\gamma}{\bar{\eta}}\Big\rceil.$ $$

Here $u^{\mathrm{d}}_{j}$ is device-local compute and $s^{\mathrm{d}}_{j}$ draft synchronization. For the target, each layer $\ell$ incurs $c_{\ell}=u^{\mathrm{t}}_{\ell}+s^{\mathrm{t}}_{\ell}$, giving

$$ $T_{\text{target}}^{\,1:\ell_{e}}=\sum_{\ell=1}^{\ell_{e}}c_{\ell},\qquad T_{\text{target}}^{\,\ell_{e}+1:N}=\sum_{\ell=\ell_{e}+1}^{N}c_{\ell}.$ $$

At early exit and final verification, rendezvous costs decompose as

$$ $T_{\mathrm{rv}}^{(\mathrm{ee})}=T_{\text{samp}}^{(\mathrm{ee})}+T_{\text{xfer}}^{(\mathrm{ee})},\quad T_{\mathrm{rv}}^{(\mathrm{fv})}=T_{\text{samp}}^{(\mathrm{fv})}+T_{\text{xfer}}^{(\mathrm{fv})},\quad T_{\mathrm{rv}}=T_{\mathrm{rv}}^{(\mathrm{ee})}+T_{\mathrm{rv}}^{(\mathrm{fv})},$ $$

where transfers involve only $O(B\kappa)$ IDs/log-probs and are negligible compared with compute.

Mirror-SD Latency Law
The per-step latency is

$$ $T_{\mathrm{Mirror}}=T_{\text{target}}^{\,1:\ell_{e}}+T_{\mathrm{rv}}^{(\mathrm{ee})}+\max\!\{T_{\text{target}}^{\,\ell_{e}+1:N},\,T_{\text{draft}}^{\mathrm{gen}}(\gamma)\}+T_{\mathrm{rv}}^{(\mathrm{fv})}.$ (14) $$

Let $\Delta=T_{\text{target}}^{\,\ell_{e}+1:N}$. If $T_{\text{draft}}^{\mathrm{gen}}(\gamma)\leq\Delta$, draft work is fully hidden: $T_{\mathrm{Mirror}}=T_{\text{target}}+T_{\mathrm{rv}}$. Otherwise, draft cost dominates the parallel region: $T_{\mathrm{Mirror}}=T_{\text{target}}^{\,1:\ell_{e}}+T_{\text{draft}}^{\mathrm{gen}}(\gamma)+T_{\mathrm{rv}}$.
Compared to vanilla SD,

$$ $T_{\mathrm{SD}}=T_{\text{target}}^{\,1:\ell_{e}}+T_{\text{target}}^{\,\ell_{e}+1:N}+T_{\text{draft}}^{\mathrm{gen}}(\gamma),$ $$

Mirror-SD hides draft work up to $\Delta$, leaving only lightweight rendezvous terms on the critical path.

##### Comparison to vanilla SD (per step).

Vanilla SD executes draft and target serially:

$$ $T_{\mathrm{SD}}=T_{\text{target}}^{\,1:\ell_{e}}+T_{\text{target}}^{\,\ell_{e}+1:N}+T_{\text{draft}}^{\mathrm{gen}}(\gamma)\;=\;T_{\text{target}}+T_{\text{draft}}^{\mathrm{gen}}(\gamma),$ $$

where we write $\Delta\!\stackrel{{\scriptstyle\mathrm{def}}}{{=}}\!T_{\text{target}}^{\,\ell_{e}+1:N}$ for the *overlap budget*.
Using the Mirror-SD law above,

$$ $T_{\mathrm{Mirror}}=T_{\text{target}}^{\,1:\ell_{e}}+T_{\mathrm{rv}}^{(\mathrm{ee})}+\max\!\{\Delta,\,T_{\text{draft}}^{\mathrm{gen}}(\gamma)\}+T_{\mathrm{rv}}^{(\mathrm{fv})}\;=\;T_{\text{target}}+T_{\mathrm{rv}},\quad\text{if }T_{\text{draft}}^{\mathrm{gen}}(\gamma)\leq\Delta,$ $$

and

$$ $T_{\mathrm{Mirror}}=T_{\text{target}}^{\,1:\ell_{e}}+T_{\text{draft}}^{\mathrm{gen}}(\gamma)+T_{\mathrm{rv}},\quad\text{if }T_{\text{draft}}^{\mathrm{gen}}(\gamma)>\Delta,$ $$

with $T_{\mathrm{rv}}\!=\!T_{\mathrm{rv}}^{(\mathrm{ee})}\!+\!T_{\mathrm{rv}}^{(\mathrm{fv})}$.

Per-step time saved. The improvement is

$$ $\Delta T\;\stackrel{{\scriptstyle\mathrm{def}}}{{=}}\;T_{\mathrm{SD}}-T_{\mathrm{Mirror}}\;=\;\big(\min\{\Delta,\,T_{\text{draft}}^{\mathrm{gen}}(\gamma)\}\big)-T_{\mathrm{rv}},$ $$

i.e., Mirror-SD hides up to the smaller of the overlap budget and the draft time, minus lightweight rendezvous. Thus Mirror-SD is strictly faster whenever

$$ $T_{\mathrm{rv}}\;<\;\min\{\Delta,\,T_{\text{draft}}^{\mathrm{gen}}(\gamma)\}.$ $$

Per-step speedup. The piecewise speedup $S\!=\!T_{\mathrm{SD}}/T_{\mathrm{Mirror}}$ is

$$ $S\;=\;\begin{cases}\dfrac{T_{\text{target}}+T_{\text{draft}}^{\mathrm{gen}}(\gamma)}{T_{\text{target}}+T_{\mathrm{rv}}},&\text{if }T_{\text{draft}}^{\mathrm{gen}}(\gamma)\leq\Delta,\[5.38193pt] \dfrac{T_{\text{target}}^{\,1:\ell_{e}}+\Delta+T_{\text{draft}}^{\mathrm{gen}}(\gamma)}{T_{\text{target}}^{\,1:\ell_{e}}+T_{\text{draft}}^{\mathrm{gen}}(\gamma)+T_{\mathrm{rv}}},&\text{if }T_{\text{draft}}^{\mathrm{gen}}(\gamma)>\Delta.\end{cases}$ $$

In practice $T_{\mathrm{rv}}$ is $O(B\kappa)$ token/log-prob exchange and sampling, i.e., microsecond-scale, so the conditions above are typically satisfied; speculative streaming (larger $\bar{\eta}$) further reduces $J$ and $T_{\text{draft}}^{\mathrm{gen}}(\gamma)$, making full hiding ($T_{\text{draft}}^{\mathrm{gen}}(\gamma)\!\leq\!\Delta$) common.

Communication Costs under TP
For $G$ devices and message size $M$ (per rank), AllReduce cost is

$$ $T_{\mathrm{allreduce}}(M;G)=\alpha\log G+\beta M,$ $$

with $\alpha$ per-hop latency and $\beta$ per-word transfer time.

Target:
Let $H_{\mathrm{T}}$ be the target hidden width, $G_{\mathrm{T}}$ its TP degree, and $S_{\mathrm{T}}$ the effective tokens per collective.
Each of the $N$ blocks performs two collectives on shards of size
$M_{\mathrm{T}}=\frac{B\,S_{\mathrm{T}}\,H_{\mathrm{T}}}{G_{\mathrm{T}}}$, giving

$$ $T^{\mathrm{comm}}_{\mathrm{target}}=2N\cdot T_{\mathrm{allreduce}}\!\big(M_{\mathrm{T}};\,G_{\mathrm{T}}\big).$ $$

Draft:
Let $H_{\mathrm{D}}$ be the draft hidden width, $G_{\mathrm{D}}$ its TP degree, and $S_{\mathrm{D}}$ the effective tokens per draft collective.
Each draft *internal* step performs two collectives on shards of size
$M_{\mathrm{D}}=\frac{B\,S_{\mathrm{D}}\,H_{\mathrm{D}}}{G_{\mathrm{D}}}$, so

$$ $T^{\mathrm{comm}}_{\text{draft-step}}=2\,T_{\mathrm{allreduce}}\!\big(M_{\mathrm{D}};\,G_{\mathrm{D}}\big),\qquad T^{\mathrm{comm}}_{\text{draft (over $J$ steps)}}=2J\,T_{\mathrm{allreduce}}\!\big(M_{\mathrm{D}};\,G_{\mathrm{D}}\big),$ $$

which is included in $T_{\text{draft}}^{\mathrm{gen}}(\gamma)$.

Cross-accelerator:
Token-channel exchanges remain $O(B\kappa)$ IDs/log-probs and are microsecond-scale.

### Appendix D Extended Ablations & Empirical Analysis

#### D.1 Batching Effects

In deployment, batching is often enabled to improve throughput and amortize GPU compute, but it is not universal: many interactive or privacy-sensitive settings prioritize per-request latency and avoid batching. To ensure completeness, we therefore also evaluate Mirror-SD under batched inference. The key question is whether speculative decoding, and Mirror-SD in particular, retains its gains when batching is enabled, or whether draft overhead grows to the point of erasing speedup. To bound the growth of draft-side computation with increasing batch size and to keep draft execution maximally hidden under the target, we *scale the draft hyperparameters with $B$*: as $B$ increases, we reduce both Top-$\kappa$ and the number of SS lookahead streams so that aggregate draft cost and the token-channel payload remain controlled. Concretely, we use $\kappa{=}8$ with two SS streams for $B\in\{1,8\}$; from $B{=}16$ onward we use a single SS stream and progressively reduce $\kappa$: $\kappa{=}4$ for $B{=}16$, $\kappa{=}2$ for $B{=}32$, and $\kappa{=}1$ for $B\geq 64$.

Observed trends. We find that vanilla SD speedup declines steadily as batch size $B$ increases ([Figure 5(b)](https://arxiv.org/html/2510.13161v2#A2.F5.sf2)). Larger batches lengthen the target verification phase both because more sequences must be processed in parallel and because batching introduces additional padding and synchronization under tensor-parallel execution. Mirror-SD also shows a downward trend with $B$, but consistently outperforms vanilla SD ([Figure 5(b)](https://arxiv.org/html/2510.13161v2#A2.F5.sf2), [Figure 5(a)](https://arxiv.org/html/2510.13161v2#A2.F5.sf1)). As $B$ grows, the draft must evaluate top-$\kappa$ candidates across $\gamma$ positions for each sequence, which increases draft compute and intra-NPU communication and pushes the draft path toward a compute-bound regime. Consequently, its ability to overlap with target verification diminishes. This decreased yet positive overlap is sufficient for Mirror-SD to maintain a consistent speedup lead over vanilla SD as batching increases. In practice, batching introduces several intertwined effects: (i) the *target* takes longer, enlarging the potential overlap window; (ii) the *draft* also takes longer, and its relative overhead grows with the $\kappa\times\gamma$ expansion; (iii) autoregressive baselines slow as $B$ increases; (iv) speculative decoding slows even more, as it inherits both AR’s slowdown and the draft’s added work; and (v) under tensor-parallel sharding, both SD variants lose relative speedup, but Mirror-SD maintains a consistent lead by exploiting concurrency across heterogeneous accelerators.

Relative draft overhead. We also report a normalized “relative draft overhead” in  [Figure 5](https://arxiv.org/html/2510.13161v2#A2.F5), defined as the fraction of draft speculation time that cannot be hidden under target verification, normalized against the total overhead of vanilla SD. This metric is dimensionless and directly reveals how much of the draft path remains exposed on the critical path. As batch size $B$ increases, the verification phase grows longer, but draft compute and intra-NPU communication grow even faster (since each sequence requires top-$\kappa$ rollouts across $\gamma$ positions). Consequently, relative draft overhead rises with $B$, aligning with the decreasing speedups observed in our batching experiments.

#### D.2 Draft-side speedups with speculative streaming

We quantify the internal draft gains from Speculative Streaming (SS) under the same targets and decoding settings as our main experiments. As described in  [Section 3.2](https://arxiv.org/html/2510.13161v2#S3.SS2), SS verifies previously proposed tokens while producing multiple new lookahead tokens in a single forward pass via multi-stream attention. Empirically, this reduces the number of draft internal steps $J$ needed to materialize a window of length $\gamma$, typically yielding $J\ll\gamma$ and a corresponding reduction in draft generation time $T_{\text{draft}}^{\mathrm{gen}}(\gamma)$.  [Figure 6(b)](https://arxiv.org/html/2510.13161v2#A4.F6.sf2) reports the draft-only speedup of SS over a plain autoregressive draft across translation, summarization, QA, mathematical reasoning, RAG, and MT-Bench. The effect is consistent across workloads: SS achieves substantially fewer internal steps for the same $\gamma$ and, consequently, shorter $T_{\text{draft}}^{\mathrm{gen}}(\gamma)$. When composed with Mirror-SD’s overlap ([Section 3.4](https://arxiv.org/html/2510.13161v2#S3.SS4)), this pushes the operating point further into the zero-slope region where increases in $\gamma$ raise acceptance length $\mathbb{E}[A_{t}]=\gamma\,\rho(\gamma;\phi,\theta)$ without increasing step latency. Because acceptance semantics are unchanged ( [Appendix B](https://arxiv.org/html/2510.13161v2#A2)), all end-to-end gains are purely systems-level.

Figure: (a) Speculative Streaming (SS): each draft step proposes multiple tokens via lookahead streams; accepted tokens extend the prefix, rejected ones are dropped.
Refer to caption: x9.png

Figure: (a) Humanities
Refer to caption: x11.png

### Appendix E Fallback Dynamics: Influence of Top- κ \kappa and Early-Exit Depth

#### E.1 Setup and definitions

At decoding step $t$, let the target’s final next-token distribution be
$q(\cdot)=p^{(N)}(\cdot\mid y_{<t},x)$ and the early-exit proxy be
$\tilde{p}(\cdot)=p^{(\ell_{e})}(\cdot\mid y_{<t},x)$.
The target accepts a prefix of length $A_{t}$ and, if a mismatch occurs, issues a correction at index $\tau\!=\!A_{t}{+}1$ with token $c_{t+\tau}$.
The draft precomputes a branch-complete window conditioned on the early-exit Top-$\kappa$ set
$M_{t}=\{(v_{i},\log\tilde{p}_{i})\}_{i=1}^{\kappa}$.
Reuse succeeds iff the target’s correction lies on a precomputed path,

$$ $\Pi_{t}^{+}\in\mathrm{Paths}_{\tau}(T_{t}),$ $$

otherwise we *fallback* (re-initialize the draft from the corrected context).
Let $F_{t}=\mathbb{1}\{\Pi_{t}^{+}\notin\mathrm{Paths}_{\tau}(T_{t})\}$ and $\mathsf{FF}\equiv\mathbb{E}[F_{t}]$.
Define the *overlap mass*

$$ $\Omega_{\kappa}(\ell_{e})\;\stackrel{{\scriptstyle\mathrm{def}}}{{=}}\;\sum_{y\in\mathrm{Top\text{-}\kappa}(\tilde{p})}q(y),$ $$

i.e., the probability under $q$ that the next token lies in the early-exit Top-$\kappa$ set.

#### E.2 Monotonicity in k k

Proposition 1 (Top-$\kappa$ reduces fallback).
For a fixed early-exit layer $\ell_{e}$, the fallback frequency $\mathsf{FF}(\ell_{e},\kappa)$ is nonincreasing in the integer $\kappa$ and vanishes as $\kappa\!\to\!|V|$:

$$ $\kappa_{2}\geq\kappa_{1}\;\Longrightarrow\;\mathsf{FF}(\ell_{e},\kappa_{2})\leq\mathsf{FF}(\ell_{e},\kappa_{1}),\qquad\lim_{\kappa\to|V|}\mathsf{FF}(\ell_{e},\kappa)=0.$ $$

*Proof.*
If $A_{t}\!=\!0$ (mismatch on the first token), reuse succeeds iff $y_{t+1}\!\in\!\mathrm{Top\text{-}\kappa}(p^{(\ell_{e})})$, so
$\Pr[F_{t}\!=\!1\mid A_{t}\!=\!0]=1-\Omega_{\kappa}(\ell_{e})$.
If $A_{t}\!\geq\!1$, the root matches $y_{t+1}$ and reuse at depth $\tau$ requires $c_{t+\tau}$ to appear on some branch of the hypothesis tree $T_{t}$ seeded by $\mathrm{Top\text{-}\kappa}(p^{(\ell_{e})})$. Increasing $\kappa$ only adds roots/paths and never removes existing ones, so $\{\Pi_{t}^{+}\!\in\!\mathrm{Paths}_{\tau}(T_{t})\}$ is monotone in $\kappa$. Taking expectations over $t$ yields the claim. The limit follows because $\Omega_{\kappa}(\ell_{e})\!\to\!1$ as $\kappa\!\to\!|V|$, at which point the hypothesis tree contains all needed paths.

A useful corollary is

$$ $\mathsf{FF}(\ell_{e},\kappa)\;\leq\;1-\Omega_{\kappa}(\ell_{e}),$ $$

which is tight when most fallbacks occur at $\tau\!=\!1$ (high-entropy regimes).

#### E.3 Monotonicity in early-exit depth

Proposition 2 (Deeper exit reduces fallback).
Fix $\kappa$. As the early-exit layer $\ell_{e}$ moves deeper (toward $N$), the overlap mass

$$ $\Omega_{\kappa}(\ell_{e})\;=\;\sum_{y\in\mathrm{Top\text{-}\kappa}(p^{(\ell_{e})})}q(y)$ $$

converges to its maximal value $q(S^{\star})$ with $S^{\star}=\mathrm{Top\text{-}\kappa}(q)$; consequently $\mathsf{FF}(\ell_{e},\kappa)\leq 1-\Omega_{\kappa}(\ell_{e})$ decreases with depth and stabilizes at its minimum for sufficiently deep exits.

*Proof.*
As the layer index $\ell$ increases, the distributions $p^{(\ell)}$ approach $q$; write
$\varepsilon_{\ell}\!\stackrel{{\scriptstyle\mathrm{def}}}{{=}}\!\|p^{(\ell)}-q\|_{\infty}\to 0$.
Let $S_{\ell}=\mathrm{Top\text{-}\kappa}(p^{(\ell)})$ and $S^{\star}=\mathrm{Top\text{-}\kappa}(q)$.
Because $S_{\ell}$ maximizes $p^{(\ell)}$-mass among all size-$\kappa$ sets, and any such set $A$ satisfies
$|q(A)-p^{(\ell)}(A)|\leq\kappa\,\varepsilon_{\ell}$, we have

$$ $\Omega_{\kappa}(\ell)=q(S_{\ell})\;\geq\;q(S^{\star})-2\kappa\,\varepsilon_{\ell}\;\xrightarrow[\ell\uparrow N]{}\;q(S^{\star}).$ $$

If the Top-$\kappa$ boundary of $q$ has margin $\Delta_{\kappa}>0$, then whenever $\varepsilon_{\ell}<\Delta_{\kappa}/2$
the Top-$\kappa$ set stabilizes ($S_{\ell}=S^{\star}$) for all deeper layers, so $\Omega_{\kappa}(\ell)=q(S^{\star})$ thereafter.
Since reuse probability is monotone in the $q$-mass captured by the seed set, the bound
$\mathsf{FF}(\ell_{e},\kappa)\leq 1-\Omega_{\kappa}(\ell_{e})$ implies a (weakly) decreasing $\mathsf{FF}$ with depth and
eventual stabilization at its minimum.

#### E.4 Empirical confirmation

[Figure 7](https://arxiv.org/html/2510.13161v2#A4.F7) reports fallback frequency as a function of $k$ for early exits at 1/4, 1/2, and 3/4 of depth across six tasks. Two consistent trends emerge:

- •
Top-$\kappa$ effect. Increasing $k$ monotonically lowers fallback, with diminishing returns once $\Omega_{\kappa}$ saturates. This matches the bound $\mathsf{FF}\leq 1-\Omega_{\kappa}(\ell_{e})$ and reflects a higher probability that the draft’s precomputed path already contains the target’s correction.
- •
Early-exit effect. Holding $k$ fixed, moving the exit deeper (1/4 $\to$ 1/2 $\to$ 3/4) lowers fallback across tasks. Deeper exits raise $\Omega_{\kappa}$ by improving agreement between the early-exit proxy and the final distribution, so the correction token more often lies on a precomputed branch.

#### E.5 Practical recommendation

Unless otherwise noted, across all SpecBench experiments reported in  [Table 1](https://arxiv.org/html/2510.13161v2#S4.T1) we set the Top-$\kappa$ width to $\kappa=8$ and fix the early exit to the middle of the network ($\ell_{e}=N/2$, “Exit 1/2”). In practice, this mid-depth, $k{=}8$ configuration works well across most setups, balancing fallback probability and the overlap budget for draft precomputation.

Choosing $k$ and $\ell_{e}$ trades a small token-channel payload and longer precomputation for fewer fallbacks and, consequently, longer accepted prefixes per step. In Mirror-SD, the channel payload is $O(B\kappa)$ and the precomputation runs in parallel under the target suffix; thus, within the overlap budget, increasing $k$ or moving $\ell_{e}$ deeper reduces fallback *without* adding step latency, directly improving end-to-end throughput via larger expected acceptance length. For bandwidth-constrained deployments, $\kappa{=}8$, $\ell_{e}{=}N/2$ is a robust default; when acceptance is still low, increase $\kappa$ or move the exit slightly deeper (subject to the overlap budget), and when channel or memory is tight, reduce $\kappa$ or use a slightly shallower exit.

### Appendix F Additional Experimental Details

#### F.1 Target and Draft Sharding

For the experiments in [Section 4](https://arxiv.org/html/2510.13161v2#S4), both target and draft models were distributed across *eight Apple M2 Ultra systems*  (apple_m2_ultra), each integrating a high-throughput GPU and a dedicated Neural Engine (NPU).
We allocate the target to GPUs using Megatron-style tensor parallelism and the draft to NPUs using SPD-style sharding (see [Section 3.3](https://arxiv.org/html/2510.13161v2#S3.SS3)). Each M2 Ultra consists of a dual-die package connected internally by *UltraFusion*, a die-to-die interconnect providing up to 2.5 TB/s of bandwidth while presenting the system as a single logical GPU/NPU pair (apple_m2_ultra).
Across machines, we organize the 8 nodes into groups of 2, linked by Thunderbolt 5 interconnects (up to 120 Gbps peak bandwidth) (apple_thunderbolt5).
Groups are further connected through a high-speed network fabric, providing sufficient bandwidth for inter-group synchronization with sub-millisecond latency.

In this setup, cross-accelerator token-channel communication consists only of $O(B\kappa)$ items (token IDs and a few log-probabilities), transferred via GPU$\rightarrow$CPU$\rightarrow$NPU copies. These messages remain negligible compared to inter-layer collectives and draft compute, consistent with the latency analysis in [Section 3.4](https://arxiv.org/html/2510.13161v2#S3.SS4).

#### F.2 Draft Model Configuration

The draft used in our experiments is a 0.6B-parameter model trained with the SPD architecture (Kim2025SPDSD).
It is organized into 16 transformer layers, divided into two contiguous segments of 8 layers each.
Within every segment we instantiate $G_{D}{=}8$ parallel tracks, where track $g\in\{1,\dots,G_{D}\}$ is pinned to NPU $g$ and advances through its resident shard of the segment.
Each track operates with a hidden size of 256 per shard. As in [Section 3.3](https://arxiv.org/html/2510.13161v2#S3.SS3), there is no inter-NPU traffic within a segment. Synchronization occurs only twice per forward pass: once at the segment boundary to re-align tensor partitions, and once at the output to assemble logits for both main and lookahead streams.

### Appendix G LLM Usage Statement

In preparing this manuscript, we used AI-assisted tools to check grammar and to rephrase some sentences for clarity and readability. No content, results, or analysis were generated by AI systems; all scientific contributions and conclusions are our own.
