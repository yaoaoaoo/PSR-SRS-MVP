# Security Policy

## Reporting a Vulnerability

If you discover a security issue, please use the repository's private
security reporting feature if enabled. If not available, contact the
project maintainers directly.

## Scope

This is a **local-only MVP** for research and demonstration purposes.

- **No real user data**: All data in `data/sample/` is synthetic.
- **No network services**: No API server, no database, no external connections.
- **No production use**: This project is not intended for production deployment.

## What We Do Not Accept

- This project is not responsible for any production security incidents.
- The synthetic data generator does not handle PII or credentials.
- The evaluation pipeline does not connect to external services.

## Best Practices for Users

- Do not add real user data to `data/sample/`.
- Do not expose the project directory via a web server.
- Review any third-party dependencies before installation.
