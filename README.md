# Let-BERT-SPEAK
Training-Free Block Diffusion Language Model with BERT

Code: https://github.com/trotsky1997/Let-BERT-SPEAK/blob/main/generate.py

Blog: https://trotsky1997.notion.site/Let-BERT-SPEAK-Training-Free-Block-Diffusion-Language-Model-with-BERT-2a2bbfcc4cdf802aa67dcba6a02a0c9f

## Completion
![b2834a4a02aeebe47dbb74f79a5cc008](https://github.com/user-attachments/assets/da9e4402-4808-460d-84cd-c827aaf25582)

## Instruction Following / Chat




## 3 Method

### 3.1 Overview

We propose **Blockwise Diffusion Generation (BDG)** — a training-free text generation framework that transforms masked language models (MLMs) such as **BERT** or **RoBERTa** into autoregressive-like generators. Instead of fine-tuning the model for next-token prediction, BDG iteratively refines masked segments of text using the model’s native masked token prediction capability.

At each iteration, the model fills in several consecutive masked tokens (a *block*), evaluates token confidence, and re-masks uncertain positions, thereby forming a *diffusion-like refinement process* over discrete token space.

------

### 3.2 Blockwise Masked Generation

Given an input sequence ( x =$$x_1, \dots, x_t] ), we append a block of ( B ) mask tokens (`[MASK]`) to the sequence:
$$
x' =$$x_1, \dots, x_t, \underbrace{[\text{MASK}], \dots,$$\text{MASK}]}*{B}]
$$
 This extended sequence is fed into the MLM to predict the probability distribution ( P*\theta(v|x') ) over the vocabulary at each masked position.

We then perform **Top-k**, **Top-p (nucleus)**, and **temperature-scaled** sampling to select tokens for replacement:
$$
p_i = \text{softmax}\left( \frac{\text{logits}_i}{T} \right)
$$
 where ( T ) denotes temperature. The sampling is restricted to the top-k or top-p subset of candidate tokens to maintain diversity and coherence.

------

### 3.3 Token Sampling and Constraints

To ensure controllability, we introduce a **banned-token filtering mechanism**, excluding undesired words (e.g., `[UNK]`, “bot”, or user-defined terms). Before sampling, all corresponding token IDs are masked out from the logits by assigning them (-\infty).

The final token for each masked position is drawn from the normalized probability mass of the allowed tokens.

------

### 3.4 Remasking Strategy

After each sampling round, BDG applies a **remasking step** to refine uncertain or low-confidence tokens.
 Let ( p(v^*) ) denote the predicted probability of the sampled token ( v^* ).
 A token is **accepted** if ( p(v^*) > \tau_\text{high} ) and **re-masked** if ( p(v^*) < \tau_\text{low} ), where ( \tau_\text{high}, \tau_\text{low} \in$$0,1] ) are confidence thresholds.

Additionally, we randomly re-mask a subset of tokens at each step following a decaying *remask ratio* schedule ( r_t ), linearly decreasing across iterations:
$$
 r_t = \text{linspace}(0, r_{\max}, T_\text{steps})[-t]
$$
 This mechanism promotes diversity early on and stability in later steps, mimicking diffusion’s progressive denoising process.

------

### 3.5 Iterative Block Diffusion

The generation process proceeds for ( N ) *refinement steps* per block.
 At each step:

1. Predict token logits using the MLM.
2. Sample new tokens under top-k/top-p/temperature control.
3. Replace confident predictions.
4. Re-mask low-confidence tokens and random subsets.

This iterative replacement gradually converges to a stable token configuration, producing coherent continuations even without autoregressive training.

------

### 3.6 Arbitrary-Length Generation

To produce long texts beyond the MLM’s native receptive field, we adopt a **rolling-block mechanism**.
 After each block refinement, the newly generated tokens are appended to the input sequence, and a new block of `[MASK]` tokens is added for the next iteration.
 This allows the model to generate arbitrarily long sequences:
$$
 x^{(t+1)} = \text{concat}(x^{(t)}, \text{BlockwiseGenerate}(x^{(t)}))
$$

The process continues until a target length or an end-of-sequence condition (e.g., `[SEP]`) is reached.

------

### 3.7 Summary

The proposed BDG framework thus enables *training-free text generation* with BERT-style MLMs through:

- **Blockwise masked token diffusion** for multi-token generation;
- **Confidence-aware remasking** for iterative refinement;
- **Rolling-block extension** for long-text synthesis; and
- **Lexical control** via banned-token filtering.

This simple yet effective paradigm leverages pre-trained MLMs’ bidirectional context understanding while emulating autoregressive decoding behavior.

