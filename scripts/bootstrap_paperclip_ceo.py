from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

BASE_URL = "https://raw.githubusercontent.com/paperclipai/companies/main/default/ceo"
FILES = ("AGENTS.md", "HEARTBEAT.md", "SOUL.md", "TOOLS.md")


def download(url: str) -> str:
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def write_file(path: Path, content: str, force: bool) -> str:
    if path.exists() and not force:
        return f"skipped {path} (already exists)"
    path.write_text(content, encoding="utf-8")
    return f"wrote   {path}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap the Paperclip CEO persona files into a local agent directory "
            "for OpenClaw/Codex-style agent setups."
        )
    )
    parser.add_argument(
        "--agent-dir",
        default="agents/ceo",
        help="Directory where the CEO agent files should be stored. Default: %(default)s",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files instead of skipping them.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    agent_dir = Path(args.agent_dir).expanduser().resolve()
    agent_dir.mkdir(parents=True, exist_ok=True)

    print(f"Bootstrapping CEO files into: {agent_dir}")
    for filename in FILES:
        content = download(f"{BASE_URL}/{filename}")
        print(write_file(agent_dir / filename, content, force=args.force))

    print("\nUse these values in the OpenClaw / agent creation form:")
    print("- Adapter type: Codex (local) or Claude Code (local)")
    print(f"- Working directory: {Path.cwd().resolve()}")
    print(f"- Agent instructions file: {agent_dir / 'AGENTS.md'}")
    print("- Prompt template: leave the default template unless you have a custom heartbeat flow")
    print(
        "- First task: Create the CEO HEARTBEAT.md, verify the persona files, then hire a Founding Engineer"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
