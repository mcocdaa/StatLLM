# Contributing to StatLLM

Thank you for your interest in contributing to **StatLLM**! We welcome community contributions, including new discrete mathematical probes, empirical model fingerprint data, bug fixes, UI improvements, and documentation enhancements.

---

## 1. Code of Conduct & Core Invariants

Before contributing, please review our core architectural invariants:
- **Zero-Synthetic Data**: Benchmark fingerprint samples must come exclusively from verified, authentic API responses from LLM providers. Never commit synthetic or hallucinated data directly to `statllm.db`.
- **Zero Secret Leaks**: Never commit API keys, tokens, or credentials to Git. Use environment variables (via `.env`) and check `.gitignore`.
- **Strict Bilingual Separation**: The Web UI supports both English (`en`) and Chinese (`zh`). Ensure that English mode contains zero untranslated Chinese characters. Run `python scripts/audit_i18n.py` to verify.

---

## 2. Development Setup

```bash
# 1. Fork and clone repository
git clone https://github.com/mcocdaa/StatLLM.git
cd StatLLM

# 2. Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install development dependencies
pip install -r requirements.txt
pip install -e ".[test]"

# 4. Run tests
pytest -v tests/
```

---

## 3. Contributing New Probes

1. Create or extend a `Probe` class in `statllm/probes.py`.
2. Implement strict extraction logic in `parse(raw_text)` returning:
   - `parsed_tokens`: List of extracted string tokens.
   - `traits`: Dictionary of behavioral traits (e.g., `first_token`, `has_duplicates`).
   - `is_valid`: Boolean indicating valid array structure.
   - `strictly_complied`: Boolean indicating zero extra markdown formatting.
3. Add probe metadata to `PROBES` registry.
4. Add corresponding i18n keys in `web/static/app.js` (`I18N.en` and `I18N.zh`).
5. Add unit tests in `tests/test_probes.py`.

---

## 4. Submitting Pull Requests

1. Create a feature branch: `git checkout -b feat/your-feature-name`.
2. Commit your changes following Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`).
3. Ensure all tests pass: `pytest -v tests/`.
4. Run the Playwright bilingual audit: `python scripts/audit_i18n.py`.
5. Push to your fork and submit a Pull Request to `main`.
