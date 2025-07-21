# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within Model Audit Copilot, please follow these steps:

1. **DO NOT** open a public issue.
2. Email your findings to tobyliu.sw@outlook.com
3. Include the following information:
   - Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
   - Full paths of source file(s) related to the manifestation of the issue
   - The location of the affected source code (tag/branch/commit or direct URL)
   - Any special configuration required to reproduce the issue
   - Step-by-step instructions to reproduce the issue
   - Proof-of-concept or exploit code (if possible)
   - Impact of the issue, including how an attacker might exploit the issue

## Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 5 business days
- **Resolution Target**: 
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 90 days

## Security Best Practices

When using Model Audit Copilot:

1. **API Keys and Secrets**:
   - Never commit API keys or secrets to the repository
   - Use environment variables for sensitive configuration
   - Rotate credentials regularly

2. **Model Security**:
   - Validate all model inputs
   - Implement proper access controls
   - Monitor for adversarial attacks
   - Keep audit logs secure and tamper-proof

3. **Data Protection**:
   - Encrypt sensitive data at rest and in transit
   - Follow data retention policies
   - Implement proper data anonymization when necessary

4. **Docker Security**:
   - Always use specific version tags, not `latest`
   - Run containers with minimal privileges
   - Keep base images updated
   - Use secrets management for sensitive data

5. **Dependencies**:
   - Regularly update dependencies
   - Monitor for known vulnerabilities
   - Use tools like `pip-audit` or `safety`

## Disclosure Policy

When we receive a security report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find similar problems
3. Prepare fixes for all supported versions
4. Release new versions with security patches
5. Credit the reporter in release notes (unless anonymity is requested)

## Comments on this Policy

If you have suggestions on how this process could be improved, please submit a pull request.