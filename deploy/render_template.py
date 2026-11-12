#!/usr/bin/env python3
"""Render a Nightshift service template with safe escaping.

Usage:
    render_template.py FORMAT TEMPLATE [KEY=VALUE ...] -- ARG0 ARG1 ...

FORMAT is "plist" (values XML-escaped, @PROGRAM_ARGUMENTS@ becomes <string> lines) or
"systemd" (values have '%' doubled, @EXEC_START@ becomes a quoted ExecStart command line).
Every @KEY@ placeholder in the template must be supplied; unknown leftovers are an error, so a
typo can never produce a half-rendered unit. Output goes to stdout.

Kept in Python (not sed) because install paths may contain spaces, '&', '%' or '/', all of
which break naive sed substitution or systemd/plist syntax.
"""
from __future__ import annotations

import re
import sys
from xml.sax.saxutils import escape as xml_escape


def systemd_quote(arg: str) -> str:
    # systemd splits ExecStart on whitespace and honours C-style double quotes; '%' starts a
    # specifier, so it must be doubled. '$' would trigger env expansion, so escape it too.
    s = arg.replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"').replace("$", "$$")
    return f'"{s}"' if (not s or re.search(r"\s|;|'", s)) else s


def main(argv: list[str]) -> int:
    if len(argv) < 3 or "--" not in argv:
        print(__doc__, file=sys.stderr)
        return 2
    fmt, template = argv[1], argv[2]
    sep = argv.index("--")
    pairs, program = argv[3:sep], argv[sep + 1:]
    if fmt not in ("plist", "systemd"):
        print(f"unknown format {fmt!r}", file=sys.stderr)
        return 2

    values: dict[str, str] = {}
    for p in pairs:
        key, eq, val = p.partition("=")
        if not eq or not re.fullmatch(r"[A-Z0-9_]+", key):
            print(f"bad KEY=VALUE pair: {p!r}", file=sys.stderr)
            return 2
        values[key] = val

    with open(template, encoding="utf-8") as fh:
        text = fh.read()

    if fmt == "plist":
        values = {k: xml_escape(v) for k, v in values.items()}
        values["PROGRAM_ARGUMENTS"] = "\n".join(
            f"        <string>{xml_escape(a)}</string>" for a in program)
    else:
        # Values may be raw unit lines (e.g. MEMORY_HIGH_LINE), so only escape specifiers.
        values = {k: v.replace("%", "%%") if not k.endswith("_LINE") else v for k, v in values.items()}
        values["EXEC_START"] = " ".join(systemd_quote(a) for a in program)

    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in values:
            raise KeyError(key)
        return values[key]

    try:
        out = re.sub(r"@([A-Z0-9_]+)@", sub, text)
    except KeyError as exc:
        print(f"template {template} needs a value for @{exc.args[0]}@", file=sys.stderr)
        return 3
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
