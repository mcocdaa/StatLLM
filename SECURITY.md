# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |

---

## Reporting a Vulnerability

The StatLLM project takes security and data privacy seriously.

If you discover a security vulnerability, sensitive credential leak, or algorithmic exploit, please **DO NOT** open a public issue.

Instead, please report it privately:
- **Email**: `2021137961@qq.com` (Maintainer)
- **Subject**: `[SECURITY VULNERABILITY] StatLLM - <Brief Description>`

Please include:
1. Detailed description of the vulnerability.
2. Steps to reproduce or proof-of-concept.
3. Potential impact and attack vectors.
4. Suggested remediation if known.

We will acknowledge receipt within 48 hours and work with you on a responsible disclosure timeline and prompt fix.

---

## API Key Protection Policy

- StatLLM is an open-source research and black-box evaluation framework.
- The framework never stores user API keys on the server or in client cookies.
- All evaluation routines operate locally or on user-configured endpoints.
- If you use scripts in `scripts/`, store keys exclusively in `.env` (which is ignored by Git).
