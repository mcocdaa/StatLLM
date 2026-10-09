# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Onboarded 15 AI frontier flagship models across all major AI labs (DeepSeek V4.1, Zhipu GLM-5.3, Meta Muse-Spark 1.3, Google Gemma 4 31B, Gemini Pro Latest, GPT-6 Astra, Claude Sonnet 5.5, Grok 4.7, etc.).
- Expanded empirical reference database to 1,190 authentic samples, with over 503 multi-modal prompt perturbations (Persona, Chit-chat, Context Noise).
- Built Sample Size & Confidence Interval Convergence Advisory Card based on Central Limit Theorem error propagation ($SE \propto 1/\sqrt{n}$).
- Created bilingual Playwright end-to-end automated audit suite (`scripts/audit_i18n.py`) guaranteeing 0 Chinese characters in English mode.
- Added GitHub Actions CI/CD workflows for automated pytest testing and multi-arch Docker image packaging to GHCR (`ghcr.io/mcocdaa/statllm`).
- Created `AGENTS.md`, `CONTRIBUTING.md`, `SECURITY.md`, and `.env.example`.

### Changed
- Refitted 2D PCA cluster projector on updated 15-model empirical vector space.
- Reorganized root exploration scripts into `scripts/` directory for clean repository topology.
- Sanitized all scripts to read credentials strictly from environment variables.

---

## [1.0.0] - 2026-10-08

### Added
- Initial release of StatLLM black-box statistical attribution engine.
- 5 standardized discrete mathematical probes (`arr_int5`, `arr_color5`, `arr_rps5`, `arr_perm5`, `arr_letter5`).
- Bayesian log-likelihood evaluator with Dirichlet-multinomial smoothing.
- Interactive Forest Plot for 95% and 68% confidence intervals.
- Docker Compose local deployment support.
