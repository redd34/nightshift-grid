# Deploying Nightshift Grid

One coordinator (Linux system service) plus any number of workers. Each worker is a **per-user**
service on a machine that also runs Ollama, so it can see that user's keyboard/mouse activity and
yield to them.

| File | Purpose |
|---|---|
| `gen_token.sh` | Generate 256-bit tokens |
| `install_worker.sh` | macOS (LaunchAgent) / Linux (systemd `--user`) worker installer |
| `windows/install_worker.ps1` | Windows worker installer (Task Scheduler, at logon) |
| `linux/nightshift-coordinator.service.template` | Hardened system unit for the coordinator |
| `macos/…plist.template`, `linux/nightshift-worker.service.template` | Worker templates, rendered by the installer |
| `render_template.py` | Escaping-safe template renderer used by the installer |

## 1. Tokens

```bash
deploy/gen_token.sh --env > coordinator.env     # GRID_WORKER_TOKEN=… / GRID_CLIENT_TOKEN=…
# or a single token for both roles:
python3 -c 'import secrets;print(secrets.token_urlsafe(32))'
```

Workers get `GRID_WORKER_TOKEN`; `demo_submit.py`, dashboards and other clients get `GRID_CLIENT_TOKEN`.
Tokens never go on a command line (they'd be visible in `ps`): workers read them from a 0600 file
via `--token-file`.

## 2. Coordinator (Linux, systemd)

```bash
sudo useradd --system --home-dir /var/lib/nightshift --shell /usr/sbin/nologin nightshift
sudo mkdir -p /opt/nightshift /etc/nightshift
sudo cp *.py requirements.txt /opt/nightshift/
sudo python3 -m venv /opt/nightshift/.venv && sudo /opt/nightshift/.venv/bin/pip install -r /opt/nightshift/requirements.txt
sudo install -m 0640 -o root -g nightshift coordinator.env /etc/nightshift/coordinator.env
python3 deploy/render_template.py systemd deploy/linux/nightshift-coordinator.service.template \
    INSTALL_DIR=/opt/nightshift PORT=8000 -- | sudo tee /etc/systemd/system/nightshift-coordinator.service >/dev/null
sudo systemctl daemon-reload && sudo systemctl enable --now nightshift-coordinator
curl -fsS http://127.0.0.1:8000/healthz
```

State (`grid_state.db`, `output_graph.jsonl`) lives in `/var/lib/nightshift` (`StateDirectory=`).
The unit runs as user `nightshift` with `ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`,
`NoNewPrivileges` and a `@system-service` syscall filter. Open TCP 8000 to the worker subnet only.
Logs: `journalctl -u nightshift-coordinator -f`. To try it without systemd, run
`GRID_WORKER_TOKEN=… GRID_CLIENT_TOKEN=… python coordinator.py --host 0.0.0.0 --port 8000 --output output_graph.jsonl --db grid_state.db`.

## 3. Workers

Prerequisite on every worker: Ollama running on `127.0.0.1:11434` with the model pulled
(`ollama pull qwen3:4b`). The installer checks this and prints a hint; pass `--pull-model` to pull.

### macOS / Linux

```bash
# preview everything (writes nothing):
deploy/install_worker.sh --dry-run --coordinator http://10.0.0.5:8000 --model qwen3:4b
# install (token from stdin so it never hits argv or shell history):
printf '%s\n' "$GRID_WORKER_TOKEN" | deploy/install_worker.sh --coordinator http://10.0.0.5:8000 \
    --model qwen3:4b --tier 2 --token-stdin
```

Defaults: id `node-<hostname>`, code + venv in `~/.local/share/nightshift`, token in
`~/.config/nightshift/worker.token` (0600, dir 0700), `--native-idle-seconds 60`, restart
throttle 60 s. Re-running is safe: it refreshes code, deps and the service file and keeps an
existing token. `--render-to DIR` only renders the service file for review. Run `--help` for all flags.

- **macOS** installs `~/Library/LaunchAgents/com.nightshift.worker.plist` (`KeepAlive` on
  non-zero exit, `ThrottleInterval`, `ProcessType=Background`, `LowPriorityIO`, `Nice=10`).
  It is a **LaunchAgent, not a LaunchDaemon**: agents run inside the logged-in user's GUI session,
  so IOKit `HIDIdleTime` reflects the person at the keyboard, and the worker stops at logout. A
  root daemon would run at the login window and under fast-user-switching, harvesting from a
  session whose human it can't see.
- **Linux** installs `~/.config/systemd/user/nightshift-worker.service` (`Restart=on-failure`,
  `RestartSec`, `StartLimitBurst=5/10min`, `Nice=10`, `CPUWeight/IOWeight=20`, idle IO class,
  optional `--memory-high 2G`). It deliberately has **no `PrivateTmp`** because the chaos trigger
  is `/tmp/preempt_<id>`. User services stop at logout; for headless boxes run
  `sudo loginctl enable-linger $USER`. HID idle uses `xprintidle` (X11 only; install it, and
  `systemctl --user import-environment DISPLAY XAUTHORITY` if your display isn't `:0`).

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File deploy\windows\install_worker.ps1 `
    -Coordinator http://10.0.0.5:8000 -Model qwen3:4b -TokenFile C:\path\worker.token -WhatIf
# drop -WhatIf to install; -Uninstall [-Purge] to remove
```

Creates a Task Scheduler task **at logon for the current user** (restart on failure, below-normal
priority, no time limit), not a Windows service. Services run in session 0, where
`GetLastInputInfo` never sees the interactive user's input, so the worker would never yield.
The token is copied to `%APPDATA%\nightshift\worker.token` with an ACL granting only the user.
Logs: `%LOCALAPPDATA%\nightshift\worker.log`. Chaos trigger: `%TEMP%\preempt_<id>`.

## 4. Logs

| OS | Command |
|---|---|
| macOS | `tail -f ~/Library/Logs/nightshift-worker.log`; `launchctl print gui/$(id -u)/com.nightshift.worker` |
| Linux | `journalctl --user -u nightshift-worker -f`; `systemctl --user status nightshift-worker` |
| Windows | `Get-Content -Wait $env:LOCALAPPDATA\nightshift\worker.log`; `Get-ScheduledTask NightshiftWorker` |

## 5. Uninstall

```bash
deploy/install_worker.sh --uninstall           # stops (graceful drain), removes service + install dir
deploy/install_worker.sh --uninstall --purge   # also deletes ~/.config/nightshift (token)
```

Windows: `install_worker.ps1 -Uninstall [-Purge]`. Coordinator:
`sudo systemctl disable --now nightshift-coordinator` and remove the unit, `/opt/nightshift`,
`/etc/nightshift` (and `/var/lib/nightshift` if you don't need the results).

## 6. Troubleshooting

- **Worker exits with 401 / restarts every minute.** Token mismatch. Compare
  `~/.config/nightshift/worker.token` with `GRID_WORKER_TOKEN` (or `GRID_TOKEN`) in
  `/etc/nightshift/coordinator.env`, rewrite it with `--token-stdin`, then restart. Restarts are
  throttled (launchd `ThrottleInterval`, systemd `StartLimitBurst`, Task Scheduler interval),
  so a bad token never hot-loops. systemd gives up after 5 tries: `systemctl --user reset-failed nightshift-worker`.
- **Ollama missing / model not found.** `curl http://127.0.0.1:11434/api/tags` must list the
  model. Start Ollama (macOS app; `systemctl enable --now ollama`; Windows tray app) and
  `ollama pull <model>`. The installer only warns about this. The worker needs it at runtime.
- **Worker registers but never harvests.** The good-citizen checks are working as designed:
  - *HID active*: someone is using the machine, or HID idle is under `--native-idle-seconds`
    (default 60). Wait, or reinstall with a smaller value (`0` disables it).
  - *On battery*: laptops don't harvest on battery. Plug in, or install with `--allow-battery`.
  - *CPU busy*: other load is above `--cpu-busy-percent`.
  - *Chaos file*: a leftover `/tmp/preempt_<id>` pauses the worker. `rm` it.
  - *Wrong session*: the worker was started as a daemon/service or over SSH without a GUI
    session. Use the per-user installers above.
- **Can't reach coordinator.** `curl -fsS http://<coordinator>:8000/healthz` from the worker; check
  the firewall and that the coordinator binds `0.0.0.0`.
