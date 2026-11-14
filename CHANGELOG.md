# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.2] - 2026-11-14

### Fixed & Changed
- **Packaging**: Include `web` and `web/static/*` assets inside the Python wheel (`pyproject.toml`), sync FastAPI `version` in `web/app.py` with `statllm.__version__`, and add `httpx>=0.24.0` to `dev` and `test` optional dependencies.
- **Docker & Deployment**: Parameterize PyPI index via `ARG PIP_INDEX_URL=https://pypi.org/simple` in `Dockerfile`, update `docker-compose.yml` image to `v0.2.2` without overriding container code via default bind mounts, sanitize `deploy/nginx.conf.example` and `deploy/README.md` with `statllm.example.com` and `$connection_upgrade` map, and sync deployment scripts (`deploy/init.sh`, `deploy/.env.example`) to `v0.2.2`.
- **CLI & Docs**: Fix `-p 8008:8000` Docker port examples, update `--probe` help text (`arr_int5, arr_color5`), reconcile model/probe/sample counts across `README.md`, `README_CN.md`, `AGENTS.md`, and `CONTRIBUTING.md`, untrack unused legacy sample JSON files, and add `.github/dependabot.yml`.

---

## [0.2.1] - 2026-10-13

### Added & Improved
- **Academic Methodology Paper & Lab**: Added Academic Methodology Paper (§0–§10) and Single-Probe Quantile Lab with KaTeX rendering, theme toggle, and i18n polish in the Web UI.
- **Statistical Engine & Performance**: Added precomputed `probe_matrices` SQLite caching, refactored attribution/likelihood helpers, fixed P05/`R_p` & credible-interval math, added Condorcet-monotonic leaderboard ranking, added 2D PCA user projection, normalized `source_type` attribution (`6,578` active official samples), and fixed sub-feature denominator inflation.

---

## [0.2.0] - 2026-10-12

### Added
- **Stability-Weighted Robust Attribution Diagnostic**: Novel adaptive likelihood weighting algorithm ($w_j = 0.35 \cdot S_{\text{global}} + 0.65 \cdot S_{\text{contest}}$) based on empirical downside inversion rate ($P(\Delta < 0)$) to suppress single-sample distillation outliers and shared prefix collisions.
- **Precomputed 12-Probe Empirical Stability Matrix**: `statllm/stability_data.py` storing 156 model-pair quantile spreads and downside inversion risks.
- **Dedicated Robust Attribution Interface**: Distinct card contrasting Raw MLE baseline against Robust Adjusted verdict with mechanistic collision diagnostics and full per-probe weight breakdown table.
- **Tail Inversion Warning Badge**: Dynamic warning badge on the Raw MLE card alerting users when single-sample volatility inversion is detected.
- **Strict Anti-Cache Deployment**: Server-side `Cache-Control: no-cache, no-store, must-revalidate` and client script version query (`app.js?v=20261012b`) ensuring instant update delivery across all browsers.

### Changed
- **Disambiguated Raw Baseline from Robust Verdict**: Card 1 re-anchored as "Raw Empirical Fitness (MLE)" to decouple empirical observation from the weighted robust verdict.

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
