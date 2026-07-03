# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of ApexRAG seriously. If you believe you've found a
security vulnerability, please report it to us as described below.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please send an email to [abinivas.gs@gmail.com](mailto:abinivas.gs@gmail.com)
with a detailed description of the issue.

You should receive a response within 48 hours. If for some reason you do not,
please follow up via email to ensure we received your original message.

### What to include

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

## Preferred Languages

We prefer all communications to be in English.

## Policy

We will:
- Acknowledge receipt of your vulnerability report within 48 hours
- Provide an initial assessment within 5 business days
- Keep you informed of our progress toward a fix
- Publicly acknowledge your responsible disclosure (if desired)

## Security Considerations

### API Key Authentication

When deploying the ApexRAG API server, always set `APEX_API_KEY` to a strong,
random value. The API key is checked via the `X-API-Key` header on every request
(except health check endpoints).

### Database Credentials

For PostgreSQL deployments, use strong passwords and restrict network access
to trusted clients only. Never commit database credentials to version control.
Use environment variables or a `.env` file (which is gitignored).

### Rate Limiting

The built-in rate limiter protects against brute-force and DoS attacks.
Configure `APEX_RATE_LIMIT` appropriately for your deployment.

### Dependencies

ApexRAG audits its dependencies for known vulnerabilities. Run
`pip-audit` regularly in your CI pipeline:

```bash
pip install pip-audit
pip-audit
```
