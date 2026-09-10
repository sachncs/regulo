# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it
responsibly. **Do not open a public GitHub issue for security vulnerabilities.**

### How to Report

1. **Email**: Send a description to **sachncs@gmail.com**
2. **Subject line**: `[SECURITY] Brief description of vulnerability`
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
- **Assessment**: We will investigate and assess the severity within 7 days
- **Resolution**: We will work on a fix and coordinate disclosure with you
- **Credit**: We will credit you in the security advisory (unless you prefer
  anonymity)

## Disclosure Policy

- We follow a **responsible disclosure** process
- We will not disclose vulnerability details publicly until a fix is available
- We will publish a security advisory on GitHub once the fix is released
- We ask that you do not publicly disclose the issue until we have had a chance
  to address it

## Security Best Practices

When using this library:

- **Dependencies**: Keep all dependencies up to date (`pip install --upgrade`)
- **Virtual environments**: Always use a virtual environment
- **Data handling**: Do not pass sensitive data to demo scripts without proper
  safeguards
- **Pinning**: Pin dependency versions in production deployments
- **Code review**: Review all code before running in sensitive environments

## Scope

This security policy applies to the code in this repository only. It does not
cover:

- Third-party dependencies (report upstream)
- The original paper's theoretical security properties
- Deployment environments and infrastructure

## Contact

For security-related inquiries, please contact:
- **Email**: sachncs@gmail.com
- **GitHub**: Open a private security advisory via GitHub's "Report a
  vulnerability" feature
