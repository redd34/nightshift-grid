#!/usr/bin/env bash
# Generate strong random grid tokens (256 bits, URL-safe).
#   deploy/gen_token.sh            -> one token on stdout
#   deploy/gen_token.sh --env      -> coordinator.env lines (separate worker + client tokens)
# Equivalent one-liner: python3 -c 'import secrets;print(secrets.token_urlsafe(32))'
set -euo pipefail
gen() { python3 -c 'import secrets; print(secrets.token_urlsafe(32))'; }
case "${1:-}" in
  --env) printf 'GRID_WORKER_TOKEN=%s\nGRID_CLIENT_TOKEN=%s\n' "$(gen)" "$(gen)" ;;
  "")    gen ;;
  *)     echo "usage: $0 [--env]" >&2; exit 2 ;;
esac
