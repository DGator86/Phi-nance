#!/usr/bin/env python3
"""
Create all Phi Capital agents in Paperclip AI.

Usage:
    python3 scripts/create_paperclip_agents.py [--probe] [--workspace WORKSPACE_ID]

Options:
    --probe         Just probe API endpoints to discover the right ones
    --workspace     Override the workspace ID (auto-detected if not provided)
    --base-url      Base URL for Paperclip (default: http://localhost:3100)
    --dry-run       Print the agents that would be created without creating them
"""

import json
import sys
import os
import argparse
import urllib.request
import urllib.error
import urllib.parse

BASE_URL = "http://localhost:3100"
WORKSPACE_DIR = os.path.expanduser("~/.paperclip/instances/default/workspaces")
AGENTS_DIR = "/root/Phi-nance/.github/agents"

# Agent definitions extracted from .github/agents/*.agent.md
AGENTS = [
    {
        "name": "Advisor",
        "description": "Portfolio and risk advisor agent that synthesizes market context, memory, and risk analytics into actionable guidance.",
        "model": "claude-sonnet-4-6",
        "color": "teal",
        "emoji": "🧭",
        "instructions_file": ".github/agents/advisor.agent.md",
    },
    {
        "name": "Chief Trader",
        "description": "Chief Trading Officer who owns execution strategy, final trade decisions, and firm-wide P&L accountability for the Phi Capital options trading firm.",
        "model": "claude-opus-4-6",
        "color": "red",
        "emoji": "🦅",
        "instructions_file": ".github/agents/chief-trader.agent.md",
    },
    {
        "name": "Compliance Officer",
        "description": "Compliance officer who enforces trading rules, position limits, regulatory constraints, and firm policy. The final gate before live capital is risked.",
        "model": "claude-sonnet-4-6",
        "color": "indigo",
        "emoji": "🛡️",
        "instructions_file": ".github/agents/compliance-officer.agent.md",
    },
    {
        "name": "Market Analyst",
        "description": "Market analyst who synthesizes macro context, volatility regimes, sector flows, and real-time market structure into actionable daily briefings for the trading team.",
        "model": "claude-sonnet-4-6",
        "color": "purple",
        "emoji": "🌐",
        "instructions_file": ".github/agents/market-analyst.agent.md",
    },
    {
        "name": "Options Trader",
        "description": "Executes options strategies using Phi-nance, manages live positions, structures multi-leg trades, and maintains Greeks within approved limits.",
        "model": "claude-sonnet-4-6",
        "color": "cyan",
        "emoji": "📊",
        "instructions_file": ".github/agents/options-trader.agent.md",
    },
    {
        "name": "Orchestrator",
        "description": "Planning and coordination agent that decomposes Phi-nance initiatives into structured RPI execution tracks.",
        "model": "claude-opus-4-6",
        "color": "white",
        "emoji": "🎯",
        "instructions_file": ".github/agents/orchestrator.agent.md",
    },
    {
        "name": "Portfolio Manager",
        "description": "Portfolio manager who owns position sizing, capital allocation, P&L attribution, and firm-level portfolio construction across all active strategies.",
        "model": "claude-sonnet-4-6",
        "color": "green",
        "emoji": "⚖️",
        "instructions_file": ".github/agents/portfolio-manager.agent.md",
    },
    {
        "name": "Quant Analyst",
        "description": "Quantitative analyst who builds, validates, and monitors trading signals, pricing models, and strategy backtests using the Phi-nance research stack.",
        "model": "claude-sonnet-4-6",
        "color": "blue",
        "emoji": "🔬",
        "instructions_file": ".github/agents/quant-analyst.agent.md",
    },
    {
        "name": "Risk Monitor",
        "description": "RL-powered risk monitor agent that converts portfolio state into dynamic risk limits and hedge posture.",
        "model": "claude-sonnet-4-6",
        "color": "orange",
        "emoji": "🚨",
        "instructions_file": ".github/agents/risk-monitor.agent.md",
    },
    {
        "name": "Software Engineer",
        "description": "Software engineer who owns Phi-nance development, debugging, infrastructure, and data pipeline reliability. Keeps the firm's trading technology running and improving.",
        "model": "claude-sonnet-4-6",
        "color": "gray",
        "emoji": "⚙️",
        "instructions_file": ".github/agents/software-engineer.agent.md",
    },
    {
        "name": "Strategy R&D",
        "description": "Research + RL strategy-discovery agent for hypothesis generation, template search, and experiment definition.",
        "model": "claude-opus-4-6",
        "color": "magenta",
        "emoji": "🧪",
        "instructions_file": ".github/agents/strategy-rd.agent.md",
    },
]


def get_workspace_id():
    """Auto-detect workspace ID from the filesystem."""
    if os.path.isdir(WORKSPACE_DIR):
        entries = os.listdir(WORKSPACE_DIR)
        if entries:
            return entries[0]
    return None


def api_request(path, method="GET", data=None, base_url=BASE_URL):
    """Make an API request to Paperclip."""
    url = base_url + path
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read()
            return resp.status, content.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        content = e.read().decode("utf-8", errors="replace")
        return e.code, content
    except Exception as e:
        return None, str(e)


def probe_api(base_url):
    """Probe Paperclip API endpoints to find what's available."""
    paths = [
        # tRPC-style endpoints
        "/trpc/agent.list",
        "/trpc/agent.create",
        "/trpc/workspace.list",
        "/api/trpc/agent.list",
        "/api/trpc/workspace.list",
        # REST-style
        "/api/v1/agents",
        "/api/v1/workspaces",
        "/api/agents",
        "/api/workspaces",
        # OpenClaw / Paperclip
        "/api/v1/companies",
        "/api/company",
        "/health",
        "/api/health",
        "/",
    ]
    print(f"Probing Paperclip API at {base_url}\n")
    for path in paths:
        status, body = api_request(path, base_url=base_url)
        snippet = body[:150].replace("\n", " ") if body else ""
        print(f"  {status or 'ERR'} {path}: {snippet}")
    print()


def read_instructions(agent_def, repo_root="/root/Phi-nance"):
    """Read the instructions markdown from the agent file (body after frontmatter)."""
    filepath = os.path.join(repo_root, agent_def["instructions_file"])
    if not os.path.exists(filepath):
        return f"# {agent_def['name']}\n\n{agent_def['description']}"
    with open(filepath, "r") as f:
        content = f.read()
    # Strip YAML frontmatter (between --- delimiters)
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3:].strip()
    return content


def create_agents_trpc(workspace_id, base_url, dry_run, repo_root):
    """Attempt to create agents via tRPC batch endpoint."""
    # Try tRPC batch format
    print(f"Using workspace: {workspace_id}")
    print(f"Repo root: {repo_root}\n")

    success = 0
    fail = 0
    for agent in AGENTS:
        instructions = read_instructions(agent, repo_root)
        payload = {
            "workspaceId": workspace_id,
            "name": agent["name"],
            "description": agent["description"],
            "model": agent["model"],
            "color": agent["color"],
            "emoji": agent["emoji"],
            "systemPrompt": instructions,
        }

        if dry_run:
            print(f"  [dry-run] Would create: {agent['emoji']} {agent['name']} ({agent['model']})")
            success += 1
            continue

        # Try multiple endpoint formats
        endpoints_to_try = [
            ("/api/trpc/agent.create", "POST", {"0": {"json": payload}}),
            ("/trpc/agent.create", "POST", {"0": {"json": payload}}),
            ("/api/v1/agents", "POST", payload),
            ("/api/agents", "POST", payload),
        ]

        created = False
        for path, method, data in endpoints_to_try:
            status, body = api_request(path, method=method, data=data, base_url=base_url)
            if status and 200 <= status < 300:
                print(f"  ✓ Created: {agent['emoji']} {agent['name']} via {path}")
                created = True
                success += 1
                break
            elif status == 404:
                continue  # Try next endpoint
            else:
                print(f"  ? {agent['name']} via {path}: HTTP {status}: {body[:100]}")

        if not created and not dry_run:
            print(f"  ✗ Failed: {agent['emoji']} {agent['name']} - no working endpoint found")
            fail += 1

    print(f"\nResults: {success} created, {fail} failed")
    return fail == 0


def create_agents_postgres(workspace_id, repo_root):
    """Create agents by inserting directly into the Paperclip PostgreSQL database."""
    try:
        import subprocess
        # Check if psql is available
        result = subprocess.run(["which", "psql"], capture_output=True)
        if result.returncode != 0:
            print("psql not found. Trying pg_isready...")

        # Find the paperclip postgres socket
        pg_port = 54329
        pg_host = "127.0.0.1"
        pg_user = "paperclip"
        pg_db = "paperclip"

        print(f"Attempting direct PostgreSQL connection on port {pg_port}...")

        import uuid
        now = "NOW()"

        success = 0
        for agent in AGENTS:
            instructions = read_instructions(agent, repo_root)
            agent_id = str(uuid.uuid4())

            sql = f"""
INSERT INTO agents (id, workspace_id, name, description, model, color, emoji, system_prompt, created_at, updated_at)
VALUES (
    '{agent_id}',
    '{workspace_id}',
    {json.dumps(agent['name'])},
    {json.dumps(agent['description'])},
    '{agent['model']}',
    '{agent['color']}',
    {json.dumps(agent['emoji'])},
    {json.dumps(instructions)},
    NOW(), NOW()
)
ON CONFLICT (workspace_id, name) DO UPDATE SET
    description = EXCLUDED.description,
    model = EXCLUDED.model,
    color = EXCLUDED.color,
    emoji = EXCLUDED.emoji,
    system_prompt = EXCLUDED.system_prompt,
    updated_at = NOW();
"""
            cmd = [
                "psql",
                f"-h{pg_host}",
                f"-p{pg_port}",
                f"-U{pg_user}",
                f"-d{pg_db}",
                "-c", sql.strip()
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                print(f"  ✓ DB inserted: {agent['emoji']} {agent['name']}")
                success += 1
            else:
                print(f"  ✗ DB failed: {agent['name']}: {result.stderr[:150]}")

        print(f"\nDB Results: {success}/{len(AGENTS)} inserted")
        return success == len(AGENTS)
    except Exception as e:
        print(f"PostgreSQL approach failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Create Phi Capital agents in Paperclip AI")
    parser.add_argument("--probe", action="store_true", help="Probe API endpoints only")
    parser.add_argument("--workspace", help="Workspace ID override")
    parser.add_argument("--base-url", default=BASE_URL, help=f"Paperclip base URL (default: {BASE_URL})")
    parser.add_argument("--dry-run", action="store_true", help="Print agents without creating")
    parser.add_argument("--db", action="store_true", help="Use direct PostgreSQL insertion instead of API")
    parser.add_argument("--repo-root", default="/root/Phi-nance", help="Path to Phi-nance repo")
    args = parser.parse_args()

    if args.probe:
        probe_api(args.base_url)
        return

    workspace_id = args.workspace or get_workspace_id()
    if not workspace_id:
        print("ERROR: Could not auto-detect workspace ID.")
        print(f"  Check: ls {WORKSPACE_DIR}")
        print("  Or pass --workspace YOUR_WORKSPACE_ID")
        sys.exit(1)

    print(f"Phi Capital Agent Creator")
    print(f"=========================")
    print(f"Workspace ID: {workspace_id}")
    print(f"Repo root:    {args.repo_root}")
    print(f"Paperclip:    {args.base_url}")
    print()

    if args.db:
        create_agents_postgres(workspace_id, args.repo_root)
    elif args.dry_run:
        create_agents_trpc(workspace_id, args.base_url, dry_run=True, repo_root=args.repo_root)
    else:
        # Try API first, fall back to Postgres if all endpoints fail
        print("Step 1: Probing API endpoints...")
        probe_api(args.base_url)
        print("Step 2: Attempting agent creation via API...")
        ok = create_agents_trpc(workspace_id, args.base_url, dry_run=False, repo_root=args.repo_root)
        if not ok:
            print("\nAPI creation incomplete. Trying PostgreSQL direct insertion...")
            create_agents_postgres(workspace_id, args.repo_root)


if __name__ == "__main__":
    main()
