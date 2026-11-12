# Security policy

Please report vulnerabilities privately via GitHub's **"Report a vulnerability"** (Security tab →
Advisories) rather than a public issue. We aim to acknowledge reports within a few days.

The threat model and current limits are in the README ("Least-privilege auth", "Transport and
identity"). In short: a LAN coordinator serves TLS, workers pin its certificate, each worker holds
a credential bound to its own id, and relay checkpoints are treated as untrusted input to the
next worker's prompt. Known gaps: results from a compromised worker are not cross-checked, and
credentials are revoked by rotating the master worker token.
