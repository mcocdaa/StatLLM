# 📊 StatLLM: Black-Box Statistical Large Language Model Fingerprinting & Attribution

> **Scientific Black-Box LLM Attribution via Discrete Multinomial Bias, Dirichlet-Smoothed Likelihood, Central Limit Theorem Confidence Convergence, and 2D PCA Cluster Projection.**  
> *Benchmark across 15+ Frontier Flagship Models with 1,190+ Empirical Authentic Samples under Contextual Prompt Perturbations.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![CI](https://github.com/mcocdaa/StatLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/mcocdaa/StatLLM/actions/workflows/ci.yml)
[![Docker Publish](https://github.com/mcocdaa/StatLLM/actions/workflows/docker.yml/badge.svg)](https://github.com/mcocdaa/StatLLM/actions/workflows/docker.yml)

[English](README.md) | [中文文档](README_CN.md)

---

## 🌟 Overview & Mission

In today's generative AI ecosystem, model-as-a-service (MaaS) gateways, aggregators, and commercial API wrappers frequently suffer from **model swapping**, **silent quantization degradation**, or **false base-model claims**. Traditional stylometric or watermarking approaches are either subjective, brittle to prompting, or require white-box model weights and logprobs that commercial providers keep private.

**StatLLM** demonstrates that **you don't need weights or logits to fingerprint an LLM**. Because autoregressive transformers inherently lack physical true randomness, their tokenizer vocabulary, pre-training corpus distributions, and RLHF alignment carve permanent, model-specific discrete biases:
- **Zero White-box Assumptions**: Evaluates purely on public inputs and output text arrays.
- **15 Frontier Flagships Covered**: Empirical reference profiles for DeepSeek, Zhipu GLM, Meta, Google Gemma/Gemini, OpenAI, Anthropic, xAI, Alibaba, MiniMax, and Moonshot.
- **Contextual Noise Immunity**: Over 500+ perturbed samples testing resistance against system personas, chit-chat history, and business text wrappers.
- **Strict Mathematical Rigor**: Computes Dirichlet-smoothed posterior probabilities, Null Hypothesis baseline likelihoods, and 95%/68% confidence intervals.
- **Sample-Size Uncertainty Advisory**: Guides users when low sample size ($N \le 5$) widens confidence intervals, showing precise mathematical predictions for how collecting $N \ge 15$ narrows error bounds by $40\%+$.

---

## 🔬 Benchmark Matrix (15 Models, 1,190 Authentic Samples)

Every reference sample in `statllm.db` is harvested through authentic API endpoints under randomized prompt perturbations ($T \in [0.70, 0.95]$):

| AI Frontier Lab | Flagship Model | Endpoint / Channel | Samples | Key Discrete Fingerprint Traits |
|:---|:---|:---|:---:|:---|
| **DeepSeek** | **DeepSeek-V4.1-Flash** | `openrouter` | **105** | Pronounced favorite integers (7, 23, 41); distinct permutation inversion profile |
| **Zhipu AI** | **GLM-5.3** | `opencode` / `openrouter` | **74** | Strong affinity for 27, 84, 15; extreme color preference for Purple/Cyan |
| **Meta AI** | **Meta Muse-Spark 1.3** | `opencode` | **69** | Distinct preference for 27, 84, 33; strong permutation 4/5-lead biases |
| **Meta AI (Open)** | **Llama-3.3-70B** | `openrouter` | **74** | Uniform integer spread; characteristic cyclic RPS patterns |
| **Google (4-Gen)** | **Gemma 4 31B** | `openrouter` | **60** | Latest 4-series architecture; heavy clustering around 23, 87, 12, 56, 91 |
| **Google (Cloud)** | **Gemini Pro (Latest)** | `openrouter` | **70** | CoT thinking reasoning traces; distinct 42, 17, 88, 5, 73 preference pattern |
| **Google (Fast)** | **Gemini 2.5 Flash** | `openrouter` | **85** | Fast latency; tight letter clustering (G, P, K, D) |
| **OpenAI** | **GPT-6-Astra** | `openrouter` | **85** | Strong integer affinity (17, 42, 64, 92); letter preference for B, Q, L |
| **OpenAI** | **GPT-5.6-Luna** | `openrouter` | **80** | Extreme initial-token integer bias (17 at >90%); high compliance |
| **OpenAI** | **GPT-6-Luna** | `openrouter` | **60** | High mathematical precision with distinct deterministic clusters |
| **Anthropic** | **Claude-Sonnet-5.5** | `openrouter` | **85** | High entropy across color and integer probes with characteristic blue/cyan leads |
| **xAI** | **Grok-4.7** | `openrouter` | **90** | High affinity for 23, 37, 82; prominent Rock bias in game-theoretic RPS |
| **Alibaba Cloud** | **Qwen-3.8-Max** | `openrouter` | **85** | Strong Chinese native representation; Cyan/Orange bias in color probes |
| **MiniMax** | **MiniMax-M3** | `openrouter` | **85** | Unique tokenizer boundary alignments; Orange/Purple initial selections |
| **Moonshot AI** | **Kimi-K3** | `openrouter` | **83** | High long-context noise resistance; characteristic letter clusters (F, L, T, Z) |

---

## 📐 Mathematical Formulation

### 1. Dirichlet-Smoothed Probability
For candidate model $M_k$ on probe $q$, the probability of discrete token $s$ is estimated with Jeffreys prior ($\alpha = 0.5$):
$$P(s \mid q, M_k) = \frac{C_{k, q}(s) + \alpha}{N_{k, q} + \alpha \cdot |\mathcal{S}_q|}$$

### 2. Bayesian Log-Likelihood & Relative Posterior
Given a submission set $D = \{(q_1, \mathbf{s}_1), \dots, (q_n, \mathbf{s}_n)\}$:
$$\log \mathcal{L}(M_k \mid D) = \sum_{i=1}^{n} \sum_{t \in \mathbf{s}_i} \log P(t \mid q_i, M_k) + \lambda \sum_{\tau \in \text{traits}} \log P(\tau \mid q_i, M_k)$$
$$P(M_k \mid D) = \frac{\exp\big(\log \mathcal{L}(M_k \mid D) - \max_j \log \mathcal{L}(M_j \mid D)\big)}{\sum_{l} \exp\big(\log \mathcal{L}(M_l \mid D) - \max_j \log \mathcal{L}(M_j \mid D)\big)}$$

### 3. Independent Fitness vs. Null Baseline
To prevent deceptive overconfidence when none of the candidates match, StatLLM calculates an absolute fitness score against a uniform Null Baseline $\mathcal{H}_0$:
$$\text{Fitness}(M_k) = \frac{1}{1 + \exp\left(-2.5 \cdot \frac{\log \mathcal{L}(M_k) - \log \mathcal{L}(\mathcal{H}_0)}{|\log \mathcal{L}(\mathcal{H}_0)|}\right)}$$

### 4. Uncertainty & Central Limit Error Convergence
Standard error shrinks inversely with sample count:
$$\text{SE} \propto \frac{\sigma}{\sqrt{n}}$$
When $N \le 5$, the 95% confidence interval spans roughly $\pm 23\%$. Expanding to $N \ge 15$ compresses the error span by over $40\%$, directly reported in the UI diagnostic card.

---

## 🎯 The 5 Standardized Probes

1. **`arr_int5` (Integer Bias Probe)**: Request 5 random integers between 1 and 100 in JSON format. Uncovers numeric distribution preferences and duplicate avoidance.
2. **`arr_color5` (Finite Set Selection Probe)**: Pick 5 colors from rainbow colors. Exposes multinomial lexical ordering biases.
3. **`arr_rps5` (Game-Theoretic RPS Probe)**: Play 5 rounds of Rock-Paper-Scissors. Measures symmetry-breaking and transition preferences.
4. **`arr_perm5` (Permutation Inversion Probe)**: Randomly shuffle $[1, 2, 3, 4, 5]$. Tests inversion count distribution and fixed-point probabilities.
5. **`arr_letter5` (Alphabet Probe)**: Pick 5 capital letters (A-Z). Reveals tokenizer-level frequency artifacts.

---

## 🚀 Quickstart

### Option 1: Docker (Fastest)

Run the prebuilt multi-architecture container image directly:
```bash
docker run -d -p 8008:8008 --name statllm ghcr.io/mcocdaa/statllm:latest
```

Or clone and run with Docker Compose:
```bash
git clone https://github.com/mcocdaa/StatLLM.git
cd StatLLM
docker compose up -d
```
Open **`http://localhost:8008`** in your browser.

### Option 2: Local Python Environment

```bash
# Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and package
pip install -r requirements.txt
pip install -e .

# Run unit tests
pytest -v tests/

# Launch local server
statllm serve --port 8008
```

---

## 💻 CLI Usage

```bash
# View all standard discrete probes
statllm probes

# Inspect database summary across all 15 models
statllm stats

# Evaluate a quick response directly from CLI
statllm eval --probe arr_int5 "[42, 17, 88, 5, 73]"
```

---

## 🌐 Web Interface Features

- **Interactive Test Lab**: One-click probe prompt copy, dynamic response cards, multi-probe batch evaluation.
- **Dynamic Forest Plots**: Displays 68% (1-sigma) core and 95% (2-sigma) conservative confidence intervals with live hover metrics.
- **2D PCA Projection Map**: Interactive scatter cloud computed from normalized multi-probe vectors with interactive **"★ You Are Here"** user positioning.
- **Sample-Size Advisory Card**: Real-time diagnostic recommendations for narrowing confidence intervals based on error propagation.
- **Strict Bilingual Engine**: 100% pure Chinese & English modes with zero character leakage (verified by Playwright).

---

## 🔒 Security & Privacy

- **No Secret Leaks**: StatLLM never commits or logs API keys. Copy `.env.example` to `.env` to configure your keys.
- **Local-First Execution**: Evaluation runs entirely locally on your machine or private server.
- Review our [SECURITY.md](SECURITY.md) for details on responsible vulnerability reporting.

---

## 📄 License & Citation

StatLLM is licensed under the [MIT License](LICENSE).

```bibtex
@software{statllm2026,
  title = {StatLLM: Black-Box Statistical Large Language Model Fingerprinting & Attribution},
  author = {mcocdaa},
  year = {2026},
  url = {https://github.com/mcocdaa/StatLLM}
}
```
