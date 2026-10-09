# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.1] - 2026-10-09

### Fixed
- **Prevent Silent Guessing Contamination**: Eradicated implicit auto-attribution that silently assigned unverified anonymous evaluation data to the predicted `top_model`. If no explicit model is claimed, data is strictly rejected from contaminating baseline distributions.

### Added
- **Explicit Ground-Truth Model Declaration**: Added interactive ground-truth model input and datalist in the evaluation workbench, requiring explicit model identification before saving crowdsourced samples.
- **Direct Sample Contribution Modal**: Added standalone sample contribution modal in the Benchmark Database view for explicit single-probe submissions with automatic community model registration.

---

## [0.1.0] - 2026-10-09

### Added
- **15 Frontier Flagship LLMs Benchmark**: Empirical baselines covering OpenAI (`GPT-6-Astra`, `GPT-5.6-Luna`, `GPT-6-Luna`), Anthropic (`Claude-Sonnet-5.5`), Google (`Gemini-Pro-Latest`, `Gemma-4-31B`, `Gemini-2.5-Flash`), Meta (`Meta-Muse-Spark-1.3`, `Llama-3.3-70B`), DeepSeek (`DeepSeek-V4.1-Flash`), Zhipu AI (`GLM-5.3`), xAI (`Grok-4.7`), Alibaba (`Qwen-3.8-Max`), Moonshot (`Kimi-K3`), and MiniMax (`MiniMax-M3`).
- **1,190 Authentic API Fingerprints**: Zero-synthetic baseline dataset with over 503 perturbed samples across Persona, Chit-chat, and long-context noise variations.
- **5 Discrete Mathematical Probes**: Standardized probe suite (`arr_int5`, `arr_color5`, `arr_rps5`, `arr_perm5`, `arr_letter5`) with robust regex extractors for non-intrusive black-box evaluation.
- **Bayesian Log-Likelihood Engine**: Non-exclusive and exclusive likelihood estimators with Dirichlet-multinomial prior smoothing and Sigmoid Fitness scoring against a uniform null hypothesis.
- **CLT Uncertainty & Sample Size Advisory**: Interactive diagnostic system calculating standard error ($SE \propto 1/\sqrt{n}$) with KaTeX math rendering and sample size convergence recommendations.
- **Interactive 2D PCA Cluster Projection**: Dimensionality reduction mapping model fingerprints onto an interactive 2D canvas with empirical bootstrap confidence regions.
- **Zero-Leakage Bilingual SPA**: Fully localized English and Chinese interface with Chart.js Forest Plots and dynamic diagnostic cards.
- **Automated Playwright i18n Audit**: CI/CD script (`scripts/audit_i18n.py`) strictly enforcing 0 untranslated Chinese characters in English mode.
- **GitHub Actions CI/CD & GHCR Container Packaging**: Multi-architecture (`linux/amd64`, `linux/arm64`) automated Docker image publishing pipeline (`ghcr.io/mcocdaa/statllm`).
- **Open-Source Governance Standards**: Comprehensive suite of `AGENTS.md`, `CONTRIBUTING.md`, `SECURITY.md`, and `.env.example`.
