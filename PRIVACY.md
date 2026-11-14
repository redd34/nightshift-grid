# Privacy policy

_Last updated: 14 November 2026. Applies to Nightshift Grid (the `nightshift-grid` package, its MCP server and the Claude Code plugin and skill in this repository)._

Nightshift Grid is self-hosted software. There is no Nightshift service: the project does not run servers, does not have accounts, and receives no data from you.

## What we collect

Nothing. The software contains no telemetry, analytics, crash reporting or update checks, and makes no network requests to the project or its maintainers.

## Where your data goes when you use it

Everything stays on machines you control:

- **Your documents** are read from the folders you allow (`NIGHTSHIFT_MCP_ROOTS`, by default the project directory). Secret files such as `.env`, keys and credentials are refused.
- **Chunks of document text** are sent to the coordinator you run, and from there to the worker machines on your grid. On a LAN, that traffic uses TLS and per-worker credentials. If you point the MCP server at a coordinator that isn't on your own machine, every tool result carries a warning that text is leaving the machine.
- **Inference** runs in Ollama on each worker machine. Nightshift does not call any cloud model API.
- **What is stored, and where:**
  - the coordinator's job journal (job and chunk state, including chunk text and results) in a SQLite file in its state directory (`~/.nightshift` for an installed copy);
  - extracted entities and relationships, with provenance back to your files, in the output directory (`output_graph.jsonl`, `graph.json`, a manifest);
  - optionally, if you configure it, a Qdrant instance you run (only collections prefixed `nightshift_`).

  Extracted entities can include names or other personal data that appear in your documents. You control and can delete all of these files at any time; nothing is retained anywhere else.

## The Claude Code plugin and skill

The skill is instructions for Claude. It does not collect or transmit anything itself. When Claude uses the Nightshift tools, document excerpts and extracted facts appear in your Claude conversation, and Anthropic handles that conversation under its own terms and privacy policy.

## Third parties

You install and run Ollama, models, Python packages and (optionally) Qdrant yourself; each has its own terms. Installing from PyPI or GitHub is subject to their policies.

## Children

Nightshift Grid is a developer tool and is not directed at children under 18.

## Contact

Questions: open an issue at https://github.com/redd34/nightshift-grid/issues. Security issues: see [SECURITY.md](SECURITY.md).
