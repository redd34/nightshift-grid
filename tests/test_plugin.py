"""The Claude Code plugin ships in two shapes: plugin/ (skill + bundled MCP server, installed from this
repo's marketplace) and directory-plugin/ (skill only, for the Claude directory, which doesn't accept a
bundled local MCP server from unverified publishers). Their skill and metadata must not drift."""
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FULL, DIRECTORY = ROOT / "plugin", ROOT / "directory-plugin"


def test_skill_copies_are_identical():
    rel = Path("skills/nightshift-grid/SKILL.md")
    assert (FULL / rel).read_bytes() == (DIRECTORY / rel).read_bytes()
    assert (FULL / ".claude-plugin/icon.png").read_bytes() == (DIRECTORY / ".claude-plugin/icon.png").read_bytes()


def test_plugin_versions_match_package():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    for d in (FULL, DIRECTORY):
        assert json.loads((d / ".claude-plugin/plugin.json").read_text())["version"] == version, d


def test_directory_plugin_bundles_no_mcp_server_and_full_one_does():
    assert "mcpServers" not in json.loads((DIRECTORY / ".claude-plugin/plugin.json").read_text())
    assert json.loads((FULL / ".claude-plugin/plugin.json").read_text())["mcpServers"]["nightshift"]


def test_marketplace_points_at_full_plugin():
    market = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    assert [p["source"] for p in market["plugins"]] == ["./plugin"]
