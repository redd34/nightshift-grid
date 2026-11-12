#!/usr/bin/env bash
# Nightshift Grid worker installer for macOS (LaunchAgent) and Linux (systemd --user).
#
# Installs a per-user, auto-restarting, low-priority worker service. Idempotent: re-running
# updates code, deps and the service file in place. Never needs root. See deploy/README.md.
#
#   deploy/install_worker.sh --coordinator http://grid:8000 --model qwen3:4b --token-stdin < token.txt
#   deploy/install_worker.sh --dry-run --coordinator http://grid:8000
#   deploy/install_worker.sh --uninstall [--purge]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------- defaults (env overridable)
COORDINATOR="${NIGHTSHIFT_COORDINATOR:-}"
WORKER_ID="${NIGHTSHIFT_WORKER_ID:-}"
MODEL="${NIGHTSHIFT_MODEL:-qwen3:4b}"
TIER="${NIGHTSHIFT_TIER:-}"
INSTALL_DIR="${NIGHTSHIFT_INSTALL_DIR:-${HOME}/.local/share/nightshift}"
CONFIG_DIR="${NIGHTSHIFT_CONFIG_DIR:-${XDG_CONFIG_HOME:-${HOME}/.config}/nightshift}"
NATIVE_IDLE_SECONDS="${NIGHTSHIFT_NATIVE_IDLE_SECONDS:-60}"
CPU_BUSY_PERCENT="${NIGHTSHIFT_CPU_BUSY_PERCENT:-}"
OLLAMA_URL="${NIGHTSHIFT_OLLAMA_URL:-http://127.0.0.1:11434}"
MEMORY_HIGH="${NIGHTSHIFT_MEMORY_HIGH:-}"
THROTTLE_SECONDS="${NIGHTSHIFT_THROTTLE_SECONDS:-60}"
PYTHON_BIN="${NIGHTSHIFT_PYTHON:-}"
TOKEN_SRC_FILE=""
TLS_FINGERPRINT="${NIGHTSHIFT_TLS_FINGERPRINT:-}"
CA_FILE=""
TOKEN_STDIN=0
ALLOW_BATTERY=0
ALLOW_NO_IDLE_SENSOR=0
MIN_FREE_RAM_GB=""
PULL_MODEL=0
DRY_RUN=0
UNINSTALL=0
PURGE=0
NO_START=0
RENDER_TO=""
EXTRA_ARGS=()

LABEL="com.nightshift.worker"
UNIT_NAME="nightshift-worker.service"

usage() {
  cat <<'EOF'
Usage: install_worker.sh [options]

Required (unless --uninstall):
  --coordinator URL        Coordinator base URL, e.g. https://10.0.0.5:8000  [env NIGHTSHIFT_COORDINATOR]

TLS (a LAN coordinator serves HTTPS; it prints its fingerprint at startup):
  --tls-fingerprint SHA    Pin the coordinator certificate (sha256:<hex>)  [env NIGHTSHIFT_TLS_FINGERPRINT]
  --ca-file PATH           Or verify it against this CA / certificate file (absolute path)

Worker:
  --id ID                  Worker id (default: node-<sanitized short hostname>)
  --model NAME             Ollama model (default: qwen3:4b)
  --tier 1|2|3             Override capability tier (default: auto-detect)
  --native-idle-seconds N  Yield while HID idle < N s (default: 60; 0 disables)
  --max-host-cpu-pct P     Yield while non-worker CPU load > P% (worker default: 65; 0 disables)
  --cpu-busy-percent P     Deprecated alias for --max-host-cpu-pct
  --min-free-ram-gb G      Yield while free RAM < G GB (worker default: 2.0; 0 disables)
  --allow-no-idle-sensor   Run even if no HID idle sensor works (headless / Wayland without D-Bus);
                           without it the worker refuses to start there (fail-closed)
  --allow-battery          Harvest even when on battery
  --worker-arg ARG         Extra argument passed to worker_daemon.py (repeatable)
  --ollama URL             Ollama URL to check (default: http://127.0.0.1:11434)

Credential (written to <config-dir>/worker.token, mode 0600; kept if already present).
Use a per-worker credential minted on the coordinator host, not the master worker token:
  python grid_auth.py issue --id <worker-id> --worker-token-file <state>/worker.token --out <id>.credential
  --token-file PATH        Read the credential from PATH
  --token-stdin            Read the credential from stdin (first line)

Install:
  --install-dir DIR        Code + venv location (default: ~/.local/share/nightshift)
  --config-dir DIR         Token location (default: ~/.config/nightshift)
  --python PATH            Python >= 3.11 to build the venv with (default: auto)
  --pull-model             Run `ollama pull MODEL` if the model is missing
  --memory-high SIZE       Linux only: systemd MemoryHigh= guard, e.g. 2G
  --throttle-seconds N     Min seconds between restarts (default: 60)
  --no-start               Install the service file but do not (re)start it

Modes:
  --dry-run                Print every action and rendered file; change nothing
  --render-to DIR          Only render the service file into DIR (for review/CI), then exit
  --uninstall              Stop and remove the service and install dir (token kept)
  --purge                  With --uninstall, also delete the token/config dir
  -h, --help               Show this help
EOF
}

die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
warn() { printf 'warning: %s\n' "$*" >&2; }
info() { printf '==> %s\n' "$*"; }

# run CMD...: execute, or just print it under --dry-run.
run() {
  if [ "${DRY_RUN}" -eq 1 ]; then
    printf '[dry-run] %s\n' "$(printf '%q ' "$@")"
  else
    "$@"
  fi
}

need_arg() { [ "$#" -ge 2 ] && [ -n "$2" ] || die "option $1 needs a value"; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --coordinator)         need_arg "$@"; COORDINATOR="$2"; shift 2 ;;
    --id)                  need_arg "$@"; WORKER_ID="$2"; shift 2 ;;
    --model)               need_arg "$@"; MODEL="$2"; shift 2 ;;
    --tier)                need_arg "$@"; TIER="$2"; shift 2 ;;
    --native-idle-seconds) need_arg "$@"; NATIVE_IDLE_SECONDS="$2"; shift 2 ;;
    --cpu-busy-percent|--max-host-cpu-pct) need_arg "$@"; CPU_BUSY_PERCENT="$2"; shift 2 ;;
    --min-free-ram-gb)     need_arg "$@"; MIN_FREE_RAM_GB="$2"; shift 2 ;;
    --allow-no-idle-sensor) ALLOW_NO_IDLE_SENSOR=1; shift ;;
    --allow-battery)       ALLOW_BATTERY=1; shift ;;
    --worker-arg)          need_arg "$@"; EXTRA_ARGS+=("$2"); shift 2 ;;
    --ollama)              need_arg "$@"; OLLAMA_URL="$2"; shift 2 ;;
    --token-file)          need_arg "$@"; TOKEN_SRC_FILE="$2"; shift 2 ;;
    --token-stdin)         TOKEN_STDIN=1; shift ;;
    --tls-fingerprint)     need_arg "$@"; TLS_FINGERPRINT="$2"; shift 2 ;;
    --ca-file)             need_arg "$@"; CA_FILE="$2"; shift 2 ;;
    --install-dir)         need_arg "$@"; INSTALL_DIR="$2"; shift 2 ;;
    --config-dir)          need_arg "$@"; CONFIG_DIR="$2"; shift 2 ;;
    --python)              need_arg "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --pull-model)          PULL_MODEL=1; shift ;;
    --memory-high)         need_arg "$@"; MEMORY_HIGH="$2"; shift 2 ;;
    --throttle-seconds)    need_arg "$@"; THROTTLE_SECONDS="$2"; shift 2 ;;
    --no-start)            NO_START=1; shift ;;
    --dry-run)             DRY_RUN=1; shift ;;
    --render-to)           need_arg "$@"; RENDER_TO="$2"; shift 2 ;;
    --uninstall)           UNINSTALL=1; shift ;;
    --purge)               PURGE=1; shift ;;
    -h|--help)             usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
done

# ---------------------------------------------------------------- platform
# NIGHTSHIFT_OS=Linux|Darwin lets you --render-to/--dry-run another platform's unit for review.
OS="${NIGHTSHIFT_OS:-$(uname -s)}"
case "${OS}" in
  Darwin)
    SERVICE_KIND="launchd"
    SERVICE_DIR="${HOME}/Library/LaunchAgents"
    SERVICE_FILE="${SERVICE_DIR}/${LABEL}.plist"
    TEMPLATE="${SCRIPT_DIR}/macos/${LABEL}.plist.template"
    LOG_FILE="${HOME}/Library/Logs/nightshift-worker.log"
    ;;
  Linux)
    SERVICE_KIND="systemd"
    SERVICE_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
    SERVICE_FILE="${SERVICE_DIR}/${UNIT_NAME}"
    TEMPLATE="${SCRIPT_DIR}/linux/nightshift-worker.service.template"
    LOG_FILE="journalctl --user -u ${UNIT_NAME}"
    ;;
  *) die "unsupported OS ${OS}; on Windows use deploy/windows/install_worker.ps1" ;;
esac
# NIGHTSHIFT_SERVICE_DIR redirects where the service file is written (testing / staging only).
if [ -n "${NIGHTSHIFT_SERVICE_DIR:-}" ]; then
  SERVICE_DIR="${NIGHTSHIFT_SERVICE_DIR}"
  SERVICE_FILE="${SERVICE_DIR}/$(basename "${SERVICE_FILE}")"
fi
TOKEN_FILE="${CONFIG_DIR}/worker.token"
VENV_PY="${INSTALL_DIR}/.venv/bin/python"

# ---------------------------------------------------------------- service control
service_stop_remove() {
  if [ "${SERVICE_KIND}" = "launchd" ]; then
    # bootout sends SIGTERM -> worker drains and hands its lease back.
    if launchctl print "gui/$(id -u)/${LABEL}" >/dev/null 2>&1 || [ "${DRY_RUN}" -eq 1 ]; then
      run launchctl bootout "gui/$(id -u)/${LABEL}" || true
    fi
  else
    if command -v systemctl >/dev/null 2>&1; then
      run systemctl --user disable --now "${UNIT_NAME}" || true
    fi
  fi
  if [ -e "${SERVICE_FILE}" ] || [ "${DRY_RUN}" -eq 1 ]; then
    run rm -f "${SERVICE_FILE}"
  fi
  if [ "${SERVICE_KIND}" = "systemd" ] && command -v systemctl >/dev/null 2>&1; then
    run systemctl --user daemon-reload || true
  fi
}

service_start() {
  if [ "${SERVICE_KIND}" = "launchd" ]; then
    # Re-bootstrap so an updated plist takes effect (idempotent).
    if launchctl print "gui/$(id -u)/${LABEL}" >/dev/null 2>&1; then
      run launchctl bootout "gui/$(id -u)/${LABEL}" || true
    fi
    run launchctl bootstrap "gui/$(id -u)" "${SERVICE_FILE}"
  else
    command -v systemctl >/dev/null 2>&1 || die "systemctl not found; start the worker manually"
    run systemctl --user daemon-reload
    run systemctl --user enable "${UNIT_NAME}"
    run systemctl --user restart "${UNIT_NAME}"
  fi
}

# ---------------------------------------------------------------- uninstall
if [ "${UNINSTALL}" -eq 1 ]; then
  info "Uninstalling Nightshift worker (${SERVICE_KIND})"
  service_stop_remove
  case "${INSTALL_DIR}" in
    ""|"/"|"${HOME}"|"${HOME}/") die "refusing to delete install dir '${INSTALL_DIR}'" ;;
  esac
  if [ -d "${INSTALL_DIR}" ] || [ "${DRY_RUN}" -eq 1 ]; then
    run rm -rf "${INSTALL_DIR}"
  fi
  if [ "${PURGE}" -eq 1 ]; then
    case "${CONFIG_DIR}" in
      ""|"/"|"${HOME}"|"${HOME}/") die "refusing to delete config dir '${CONFIG_DIR}'" ;;
    esac
    run rm -rf "${CONFIG_DIR}"
  else
    info "Token kept at ${TOKEN_FILE} (use --purge to delete it)"
  fi
  info "Done."
  exit 0
fi

# ---------------------------------------------------------------- validate inputs
[ -n "${COORDINATOR}" ] || { usage >&2; die "--coordinator is required"; }
case "${COORDINATOR}" in http://*|https://*) ;; *) die "--coordinator must start with http:// or https://" ;; esac
COORDINATOR="${COORDINATOR%/}"
case "${COORDINATOR}" in
  http://127.0.0.1*|http://localhost*|http://\[::1\]*|https://*) ;;
  http://*) warn "plain-http coordinator ${COORDINATOR}: tokens cross the network unencrypted (fine only on Tailscale/WireGuard)" ;;
esac
if [ -n "${TLS_FINGERPRINT}${CA_FILE}" ]; then
  case "${COORDINATOR}" in https://*) ;; *) die "--tls-fingerprint/--ca-file need an https:// coordinator" ;; esac
fi
if [ -n "${CA_FILE}" ]; then
  case "${CA_FILE}" in /*) ;; *) die "--ca-file must be an absolute path (the service runs from another directory)" ;; esac
fi

if [ -z "${WORKER_ID}" ]; then
  host=""
  if [ "${OS}" = "Darwin" ]; then host="$(scutil --get LocalHostName 2>/dev/null || true)"; fi
  [ -n "${host}" ] || host="$(hostname -s 2>/dev/null || hostname)"
  host="$(printf '%s' "${host}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-' | sed -e 's/--*/-/g' -e 's/^-//' -e 's/-$//')"
  WORKER_ID="node-${host:-unknown}"
fi
printf '%s' "${WORKER_ID}" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$' \
  || die "worker id '${WORKER_ID}' must match [A-Za-z0-9][A-Za-z0-9._-]{0,63} (it names /tmp/preempt_<id>)"
case "${TIER}" in ""|1|2|3) ;; *) die "--tier must be 1, 2 or 3" ;; esac
case "${NATIVE_IDLE_SECONDS}" in ''|*[!0-9.]*) die "--native-idle-seconds must be a number" ;; esac
case "${THROTTLE_SECONDS}" in ''|*[!0-9]*) die "--throttle-seconds must be an integer" ;; esac
if [ -n "${CPU_BUSY_PERCENT}" ]; then
  case "${CPU_BUSY_PERCENT}" in *[!0-9.]*) die "--max-host-cpu-pct must be a number" ;; esac
fi
if [ -n "${MIN_FREE_RAM_GB}" ]; then
  case "${MIN_FREE_RAM_GB}" in *[!0-9.]*) die "--min-free-ram-gb must be a number" ;; esac
fi
if [ -n "${MEMORY_HIGH}" ]; then
  printf '%s' "${MEMORY_HIGH}" | grep -Eq '^[0-9]+[KMGT]?$|^[0-9]+%$' || die "--memory-high must look like 2G or 50%"
fi
[ -f "${TEMPLATE}" ] || die "template not found: ${TEMPLATE}"
[ -f "${SRC_DIR}/worker_daemon.py" ] || die "worker_daemon.py not found in ${SRC_DIR}"

# ---------------------------------------------------------------- python >= 3.11
python_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; }
if [ -n "${PYTHON_BIN}" ]; then
  python_ok "${PYTHON_BIN}" || die "${PYTHON_BIN} is not Python >= 3.11"
else
  for cand in python3.13 python3.12 python3.11 python3; do
    if command -v "${cand}" >/dev/null 2>&1 && python_ok "$(command -v "${cand}")"; then
      PYTHON_BIN="$(command -v "${cand}")"; break
    fi
  done
  [ -n "${PYTHON_BIN}" ] || die "Python >= 3.11 not found (install it, or pass --python PATH)"
fi
info "Using $("${PYTHON_BIN}" --version 2>&1) at ${PYTHON_BIN}"

# ---------------------------------------------------------------- worker argv
WORKER_ARGV=("${VENV_PY}" "${INSTALL_DIR}/worker_daemon.py"
  --id "${WORKER_ID}" --coordinator "${COORDINATOR}" --model "${MODEL}"
  --ollama "${OLLAMA_URL}" --token-file "${TOKEN_FILE}")
if [ -n "${TIER}" ]; then WORKER_ARGV+=(--tier "${TIER}"); fi
if [ -n "${TLS_FINGERPRINT}" ]; then WORKER_ARGV+=(--tls-fingerprint "${TLS_FINGERPRINT}"); fi
if [ -n "${CA_FILE}" ]; then WORKER_ARGV+=(--ca-file "${CA_FILE}"); fi
if [ "${NATIVE_IDLE_SECONDS}" != "0" ]; then WORKER_ARGV+=(--native-idle-seconds "${NATIVE_IDLE_SECONDS}"); fi
if [ -n "${CPU_BUSY_PERCENT}" ]; then WORKER_ARGV+=(--max-host-cpu-pct "${CPU_BUSY_PERCENT}"); fi
if [ -n "${MIN_FREE_RAM_GB}" ]; then WORKER_ARGV+=(--min-free-ram-gb "${MIN_FREE_RAM_GB}"); fi
if [ "${ALLOW_NO_IDLE_SENSOR}" -eq 1 ]; then WORKER_ARGV+=(--allow-no-idle-sensor); fi
if [ "${ALLOW_BATTERY}" -eq 1 ]; then WORKER_ARGV+=(--allow-battery); fi
WORKER_ARGV+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

render_service() {
  if [ "${SERVICE_KIND}" = "launchd" ]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/render_template.py" plist "${TEMPLATE}" \
      "LABEL=${LABEL}" "INSTALL_DIR=${INSTALL_DIR}" "LOG_FILE=${LOG_FILE}" \
      "THROTTLE_SECONDS=${THROTTLE_SECONDS}" -- "${WORKER_ARGV[@]}"
  else
    local mem_line="# MemoryHigh= not set (install with --memory-high 2G to throttle this unit under memory pressure)"
    if [ -n "${MEMORY_HIGH}" ]; then mem_line="MemoryHigh=${MEMORY_HIGH}"; fi
    "${PYTHON_BIN}" "${SCRIPT_DIR}/render_template.py" systemd "${TEMPLATE}" \
      "WORKER_ID=${WORKER_ID}" "INSTALL_DIR=${INSTALL_DIR}" "DISPLAY=${DISPLAY:-:0}" \
      "THROTTLE_SECONDS=${THROTTLE_SECONDS}" "MEMORY_HIGH_LINE=${mem_line}" -- "${WORKER_ARGV[@]}"
  fi
}

if [ -n "${RENDER_TO}" ]; then
  mkdir -p "${RENDER_TO}"
  out="${RENDER_TO}/$(basename "${SERVICE_FILE}")"
  render_service > "${out}"
  info "Rendered ${out}"
  exit 0
fi

# ---------------------------------------------------------------- token
NEW_TOKEN=""
if [ "${TOKEN_STDIN}" -eq 1 ]; then
  IFS= read -r NEW_TOKEN || true
  [ -n "${NEW_TOKEN}" ] || die "no token on stdin"
elif [ -n "${TOKEN_SRC_FILE}" ]; then
  [ -r "${TOKEN_SRC_FILE}" ] || die "cannot read token file ${TOKEN_SRC_FILE}"
  IFS= read -r NEW_TOKEN < "${TOKEN_SRC_FILE}" || true
  [ -n "${NEW_TOKEN}" ] || die "token file ${TOKEN_SRC_FILE} is empty"
fi
NEW_TOKEN="$(printf '%s' "${NEW_TOKEN}" | tr -d '[:space:]')"
if [ -z "${NEW_TOKEN}" ] && [ ! -s "${TOKEN_FILE}" ]; then
  if [ "${DRY_RUN}" -eq 1 ]; then
    warn "no token given and ${TOKEN_FILE} missing (a real install would fail here)"
  else
    die "no token: pass --token-stdin or --token-file (generate one with deploy/gen_token.sh)"
  fi
fi

info "Install plan"
printf '    %-14s %s\n' "worker id" "${WORKER_ID}" "coordinator" "${COORDINATOR}" "model" "${MODEL}" \
  "tier" "${TIER:-auto}" "install dir" "${INSTALL_DIR}" "token file" "${TOKEN_FILE}" \
  "service" "${SERVICE_FILE}" "logs" "${LOG_FILE}"

if [ -n "${NEW_TOKEN}" ]; then
  run mkdir -p "${CONFIG_DIR}"
  run chmod 700 "${CONFIG_DIR}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    printf '[dry-run] write token (%d chars) to %s with mode 0600\n' "${#NEW_TOKEN}" "${TOKEN_FILE}"
  else
    tmp_tok="$(umask 077 && mktemp "${CONFIG_DIR}/.worker.token.XXXXXX")"
    printf '%s\n' "${NEW_TOKEN}" > "${tmp_tok}"
    chmod 600 "${tmp_tok}"
    mv -f "${tmp_tok}" "${TOKEN_FILE}"
  fi
elif [ -s "${TOKEN_FILE}" ]; then
  info "Keeping existing token ${TOKEN_FILE}"
  run chmod 600 "${TOKEN_FILE}"
fi

# ---------------------------------------------------------------- code + venv
info "Installing code into ${INSTALL_DIR}"
run mkdir -p "${INSTALL_DIR}"
for f in "${SRC_DIR}"/*.py "${SRC_DIR}/requirements.txt"; do
  if [ -f "${f}" ]; then run cp -f "${f}" "${INSTALL_DIR}/"; fi
done
run mkdir -p "${INSTALL_DIR}/deploy"
run cp -f "${SCRIPT_DIR}/README.md" "${INSTALL_DIR}/deploy/README.md"
if [ ! -x "${VENV_PY}" ] || ! python_ok "${VENV_PY}"; then
  run "${PYTHON_BIN}" -m venv --clear "${INSTALL_DIR}/.venv"
fi
run "${VENV_PY}" -m pip install --quiet --disable-pip-version-check --upgrade pip
# One retry: a transient index/network hiccup can surface as ResolutionImpossible.
run "${VENV_PY}" -m pip install --quiet --disable-pip-version-check -r "${INSTALL_DIR}/requirements.txt" \
  || { warn "pip install failed; retrying once"; run "${VENV_PY}" -m pip install --disable-pip-version-check -r "${INSTALL_DIR}/requirements.txt"; }

# ---------------------------------------------------------------- ollama + model
info "Checking Ollama at ${OLLAMA_URL}"
tags_json=""
if command -v curl >/dev/null 2>&1; then
  tags_json="$(curl -fsS --max-time 3 "${OLLAMA_URL%/}/api/tags" 2>/dev/null || true)"
fi
if [ -z "${tags_json}" ]; then
  warn "Ollama is not reachable at ${OLLAMA_URL}. Install it from https://ollama.com and start it"
  warn "(macOS: open the Ollama app; Linux: systemctl enable --now ollama), then: ollama pull ${MODEL}"
else
  if printf '%s' "${tags_json}" | "${PYTHON_BIN}" -c '
import json, sys
want = sys.argv[1]
names = {m.get("name", "") for m in json.load(sys.stdin).get("models", [])}
names |= {n[:-len(":latest")] for n in names if n.endswith(":latest")}
sys.exit(0 if want in names else 1)' "${MODEL}"; then
    info "Model ${MODEL} is available"
  elif [ "${PULL_MODEL}" -eq 1 ]; then
    command -v ollama >/dev/null 2>&1 || die "--pull-model given but the ollama CLI is not on PATH"
    run ollama pull "${MODEL}"
  else
    warn "Model ${MODEL} is not pulled. Run: ollama pull ${MODEL}   (or re-run with --pull-model)"
  fi
fi

# ---------------------------------------------------------------- service file
info "Writing ${SERVICE_KIND} service ${SERVICE_FILE}"
run mkdir -p "${SERVICE_DIR}"
if [ "${SERVICE_KIND}" = "launchd" ]; then
  run mkdir -p "$(dirname "${LOG_FILE}")"
fi
if [ "${DRY_RUN}" -eq 1 ]; then
  printf '[dry-run] would write %s:\n' "${SERVICE_FILE}"
  printf -- '-----8<-----\n'
  render_service
  printf -- '----->8-----\n'
else
  tmp_svc="$(mktemp "${SERVICE_DIR}/.nightshift.XXXXXX")"
  render_service > "${tmp_svc}"
  if [ "${SERVICE_KIND}" = "launchd" ]; then
    plutil -lint "${tmp_svc}" >/dev/null || { rm -f "${tmp_svc}"; die "rendered plist failed plutil -lint"; }
  elif command -v systemd-analyze >/dev/null 2>&1; then
    systemd-analyze --user verify "${tmp_svc}" 2>/dev/null || warn "systemd-analyze reported issues (see above)"
  fi
  chmod 644 "${tmp_svc}"
  mv -f "${tmp_svc}" "${SERVICE_FILE}"
fi

if [ "${NO_START}" -eq 1 ]; then
  info "--no-start: service installed but not started"
else
  service_start
fi

# ---------------------------------------------------------------- next steps
if [ "${DRY_RUN}" -eq 1 ]; then info "Dry run complete: nothing was changed."; else info "Installed. Logs: ${LOG_FILE}"; fi
if [ "${SERVICE_KIND}" = "launchd" ]; then
  echo "    status:  launchctl print gui/$(id -u)/${LABEL} | head -20"
  echo "    logs:    tail -f '${LOG_FILE}'"
else
  echo "    status:  systemctl --user status ${UNIT_NAME}"
  echo "    logs:    journalctl --user -u ${UNIT_NAME} -f"
  if command -v loginctl >/dev/null 2>&1 && ! loginctl show-user "$(id -un)" -p Linger 2>/dev/null | grep -q 'yes'; then
    echo "    note:    user services stop at logout. For a headless box run: sudo loginctl enable-linger $(id -un)"
  fi
  command -v xprintidle >/dev/null 2>&1 || warn "xprintidle not installed: HID idle detection is disabled (apt install xprintidle)"
fi
echo "    chaos:   touch /tmp/preempt_${WORKER_ID}   (rm it to release)"
