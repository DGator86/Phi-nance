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
    --db            Use direct PostgreSQL insertion instead of API
"""

import json
import sys
import os
import argparse
import urllib.request
import urllib.error
import subprocess

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
        "command": "claude",
        "instructions_file": ".github/agents/advisor.agent.md",
    },
    {
        "name": "Chief Trader",
        "description": "Chief Trading Officer who owns execution strategy, final trade decisions, and firm-wide P&L accountability for the Phi Capital options trading firm.",
        "model": "claude-opus-4-6",
        "color": "red",
        "emoji": "🦅",
        "command": "claude",
        "instructions_file": ".github/agents/chief-trader.agent.md",
    },
    {
        "name": "Compliance Officer",
        "description": "Compliance officer who enforces trading rules, position limits, regulatory constraints, and firm policy. The final gate before live capital is risked.",
        "model": "claude-sonnet-4-6",
        "color": "indigo",
        "emoji": "🛡️",
        "command": "claude",
        "instructions_file": ".github/agents/compliance-officer.agent.md",
    },
    {
        "name": "Market Analyst",
        "description": "Market analyst who synthesizes macro context, volatility regimes, sector flows, and real-time market structure into actionable daily briefings for the trading team.",
        "model": "claude-sonnet-4-6",
        "color": "purple",
        "emoji": "🌐",
        "command": "claude",
        "instructions_file": ".github/agents/market-analyst.agent.md",
    },
    {
        "name": "Options Trader",
        "description": "Executes options strategies using Phi-nance, manages live positions, structures multi-leg trades, and maintains Greeks within approved limits.",
        "model": "claude-sonnet-4-6",
        "color": "cyan",
        "emoji": "📊",
        "command": "claude",
        "instructions_file": ".github/agents/options-trader.agent.md",
    },
    {
        "name": "Orchestrator",
        "description": "Planning and coordination agent that decomposes Phi-nance initiatives into structured RPI execution tracks.",
        "model": "claude-opus-4-6",
        "color": "white",
        "emoji": "🎯",
        "command": "claude",
        "instructions_file": ".github/agents/orchestrator.agent.md",
    },
    {
        "name": "Portfolio Manager",
        "description": "Portfolio manager who owns position sizing, capital allocation, P&L attribution, and firm-level portfolio construction across all active strategies.",
        "model": "claude-sonnet-4-6",
        "color": "green",
        "emoji": "⚖️",
        "command": "claude",
        "instructions_file": ".github/agents/portfolio-manager.agent.md",
    },
    {
        "name": "Quant Analyst",
        "description": "Quantitative analyst who builds, validates, and monitors trading signals, pricing models, and strategy backtests using the Phi-nance research stack.",
        "model": "claude-sonnet-4-6",
        "color": "blue",
        "emoji": "🔬",
        "command": "claude",
        "instructions_file": ".github/agents/quant-analyst.agent.md",
    },
    {
        "name": "Risk Monitor",
        "description": "RL-powered risk monitor agent that converts portfolio state into dynamic risk limits and hedge posture.",
        "model": "claude-sonnet-4-6",
        "color": "orange",
        "emoji": "🚨",
        "command": "claude",
        "instructions_file": ".github/agents/risk-monitor.agent.md",
    },
    {
        "name": "Software Engineer",
        "description": "Software engineer who owns Phi-nance development, debugging, infrastructure, and data pipeline reliability. Keeps the firm's trading technology running and improving.",
        "model": "claude-sonnet-4-6",
        "color": "gray",
        "emoji": "⚙️",
        "command": "claude",
        "instructions_file": ".github/agents/software-engineer.agent.md",
    },
    {
        "name": "Strategy R&D",
        "description": "Research + RL strategy-discovery agent for hypothesis generation, template search, and experiment definition.",
        "model": "claude-opus-4-6",
        "color": "magenta",
        "emoji": "🧪",
        "command": "claude",
        "instructions_file": ".github/agents/strategy-rd.agent.md",
    },
]


def get_workspace_id(base_url=BASE_URL):
    """Auto-detect workspace ID from the filesystem or API."""
    # Search common locations where Paperclip may store workspaces
    candidates = [
        WORKSPACE_DIR,
        os.path.expanduser("~/.paperclip/instances/default/workspaces"),
        "/root/.paperclip/instances/default/workspaces",
    ]
    # Also search all home directories
    if os.path.isdir("/home"):
        for user_dir in os.listdir("/home"):
            candidates.append(f"/home/{user_dir}/.paperclip/instances/default/workspaces")

    for ws_dir in candidates:
        if os.path.isdir(ws_dir):
            entries = [e for e in os.listdir(ws_dir) if not e.startswith('.')]
            if entries:
                return entries[0]

    # Fallback: try API endpoints to discover workspace ID
    for path in ["/api/workspaces", "/api/workspace"]:
        status, body = api_request(path, base_url=base_url)
        if status == 200:
            try:
                data = json.loads(body)
                if isinstance(data, list) and data:
                    ws = data[0]
                    return ws.get("id") or ws.get("workspaceId")
                if isinstance(data, dict):
                    return data.get("id") or data.get("workspaceId")
            except Exception:
                pass

    # Try via company
    company_id, _ = get_company_id(base_url)
    if company_id:
        for path in [f"/api/workspaces?companyId={company_id}", f"/api/companies/{company_id}/workspaces"]:
            status, body = api_request(path, base_url=base_url)
            if status == 200:
                try:
                    data = json.loads(body)
                    if isinstance(data, list) and data:
                        ws = data[0]
                        return ws.get("id") or ws.get("workspaceId")
                    if isinstance(data, dict):
                        return data.get("id") or data.get("workspaceId")
                except Exception:
                    pass

    return None


def api_request(path, method="GET", data=None, base_url=BASE_URL, extra_headers=None):
    """Make an API request to Paperclip."""
    url = base_url + path
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
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


def get_company_id(base_url):
    """Get company ID from /api/companies."""
    status, body = api_request("/api/companies", base_url=base_url)
    if status == 200:
        try:
            companies = json.loads(body)
            if companies and isinstance(companies, list):
                return companies[0]["id"], companies[0].get("name", "")
        except Exception:
            pass
    return None, None


def probe_api(base_url, company_id=None):
    """Probe Paperclip API endpoints."""
    cid = company_id or "COMPANY_ID"
    paths_and_methods = [
        ("GET", "/api/health"),
        ("GET", "/api/companies"),
        # Try GET listing with companyId
        ("GET", f"/api/agents?companyId={cid}"),
        # Try POST to /api/agents (create)
        ("POST", "/api/agents"),
        # Company-scoped
        ("GET", f"/api/companies/{cid}"),
        ("GET", f"/api/companies/{cid}/agents"),
        ("POST", f"/api/companies/{cid}/agents"),
        # Shortname lookup pattern we know exists
        ("GET", f"/api/agents/advisor?companyId={cid}"),
    ]
    print(f"Probing Paperclip API at {base_url}\n")
    for method, path in paths_and_methods:
        if method == "POST":
            status, body = api_request(path, method="POST",
                data={"companyId": cid, "name": "test", "shortname": "test"},
                base_url=base_url)
        else:
            status, body = api_request(path, base_url=base_url)
        is_json = body and (body.strip().startswith('{') or body.strip().startswith('['))
        snippet = (body or "")[:150].replace("\n", " ")
        marker = "JSON" if is_json else "HTML" if (body and "<!DOCTYPE" in body) else "text"
        print(f"  {status or 'ERR'} {method} {marker} {path}: {snippet[:120]}")
    print()


def find_psql_binary():
    """Find the psql binary — including Paperclip's embedded postgres."""
    candidates = [
        "/usr/bin/psql",
        "/usr/local/bin/psql",
        "/home/paperclip/.paperclip/bin/psql",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # Find embedded postgres binary, then look for psql in the same dir
    result = subprocess.run(
        ["find", "/home/paperclip/.npm", "-name", "postgres", "-type", "f"],
        capture_output=True, text=True, timeout=10
    )
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Check for psql in the same bin/ directory
        bin_dir = os.path.dirname(line)
        psql_path = os.path.join(bin_dir, "psql")
        if os.path.isfile(psql_path):
            return psql_path
        # Also check parent dir
        psql_path2 = os.path.join(os.path.dirname(bin_dir), "psql")
        if os.path.isfile(psql_path2):
            return psql_path2

    # Broad search for psql
    result2 = subprocess.run(
        ["find", "/home/paperclip/.npm", "-name", "psql", "-type", "f"],
        capture_output=True, text=True, timeout=10
    )
    for line in result2.stdout.strip().splitlines():
        line = line.strip()
        if line and os.path.isfile(line):
            return line

    return None


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


def try_create_via_api(company_id, workspace_id, base_url, repo_root):
    """Try creating agents via REST API using companyId."""
    print(f"Attempting API agent creation for company {company_id}...\n")

    # List existing agents via shortname endpoint pattern
    status, body = api_request(f"/api/agents?companyId={company_id}", base_url=base_url)
    print(f"  GET /api/agents?companyId: HTTP {status}: {body[:200]}\n")

    success = 0
    fail_names = []

    for agent in AGENTS:
        instructions = read_instructions(agent, repo_root)
        slug = agent["name"].lower().replace(" ", "-").replace("&", "and")

        base_payload = {
            "companyId": company_id,
            "name": agent["name"],
            "shortname": slug,
            "description": agent["description"],
            "model": agent["model"],
            "color": agent["color"],
            "emoji": agent["emoji"],
            "systemPrompt": instructions,
            "command": agent.get("command", "claude"),
        }

        # Try every plausible create endpoint
        attempts = [
            # POST /api/agents (most likely for REST create)
            ("POST", "/api/agents", base_payload),
            # POST /api/agents with workspaceId too
            ("POST", "/api/agents", {**base_payload, "workspaceId": workspace_id}),
            # Company-scoped
            ("POST", f"/api/companies/{company_id}/agents", {k: v for k, v in base_payload.items() if k != "companyId"}),
            # PUT (upsert pattern)
            ("PUT", f"/api/agents/{slug}?companyId={company_id}", {k: v for k, v in base_payload.items() if k not in ("companyId", "shortname")}),
            # PATCH to existing shortname
            ("PATCH", f"/api/agents/{slug}?companyId={company_id}", base_payload),
        ]

        created = False
        last_status = None
        last_body = None
        for method, path, payload in attempts:
            status, body = api_request(path, method=method, data=payload, base_url=base_url)
            last_status = status
            last_body = body
            if status and 200 <= status < 300:
                print(f"  ✓ {agent['emoji']} {agent['name']} via {method} {path}")
                created = True
                success += 1
                break
            elif status == 404:
                continue
            elif status in (400, 409, 422):
                snippet = (body or "")[:250].replace("\n", " ")
                print(f"  ? {agent['name']} {method} {path} → HTTP {status}: {snippet}")
                break  # We found the right endpoint, payload is wrong

        if not created:
            fail_names.append(agent['name'])
            snippet = (last_body or "")[:150].replace("\n", " ")
            print(f"  ✗ {agent['emoji']} {agent['name']}: last HTTP {last_status}: {snippet}")

    print(f"\nAPI Results: {success} created, {len(fail_names)} failed")
    return fail_names


def create_agents_via_node(company_id, workspace_id, repo_root, base_url):
    """Create agents by running a Node.js script that uses the postgres npm package."""
    # Find node binary
    node_bin = None
    for p in ["/usr/bin/node", "/usr/local/bin/node"]:
        if os.path.exists(p):
            node_bin = p
            break
    if not node_bin:
        result = subprocess.run(["which", "node"], capture_output=True, text=True)
        if result.returncode == 0:
            node_bin = result.stdout.strip()

    if not node_bin:
        print("node binary not found")
        return False

    # Find the postgres npm module bundled with paperclip
    result = subprocess.run(
        ["find", "/home/paperclip/.npm", "-name", "index.js", "-path", "*/postgres/src/*"],
        capture_output=True, text=True, timeout=10
    )
    pg_module = None
    for line in result.stdout.strip().splitlines():
        if "/postgres/" in line and "node_modules" in line:
            # Get the module root
            idx = line.find("/postgres/")
            pg_module = line[:idx + len("/postgres")]
            break

    if not pg_module:
        # Try generic path
        result2 = subprocess.run(
            ["find", "/home/paperclip/.npm", "-maxdepth", "6", "-name", "package.json", "-path", "*/postgres/package.json"],
            capture_output=True, text=True, timeout=10
        )
        for line in result2.stdout.strip().splitlines():
            pg_module = os.path.dirname(line)
            break

    print(f"node: {node_bin}, postgres module: {pg_module}")

    # Build the JS that connects to postgres and creates agents
    agents_data = []
    for agent in AGENTS:
        instructions = read_instructions(agent, repo_root)
        slug = agent["name"].lower().replace(" ", "-").replace("&", "and")
        agents_data.append({
            "name": agent["name"],
            "shortname": slug,
            "description": agent["description"],
            "model": agent["model"],
            "color": agent["color"],
            "emoji": agent["emoji"],
            "instructions": instructions,
        })

    js_script = f"""
const http = require('http');

const AGENTS = {json.dumps(agents_data)};
const COMPANY_ID = {json.dumps(company_id)};
const WORKSPACE_ID = {json.dumps(workspace_id)};
const BASE = 'http://127.0.0.1:3100';

function apiPost(path, data) {{
  return new Promise((resolve, reject) => {{
    const body = JSON.stringify(data);
    const opts = {{
      hostname: '127.0.0.1', port: 3100,
      path, method: 'POST',
      headers: {{'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body)}}
    }};
    const req = http.request(opts, res => {{
      let d = '';
      res.on('data', c => d += c);
      res.on('end', () => resolve({{status: res.statusCode, body: d}}));
    }});
    req.on('error', reject);
    req.write(body);
    req.end();
  }});
}}

async function main() {{
  let ok = 0;
  for (const agent of AGENTS) {{
    const payload = {{ companyId: COMPANY_ID, workspaceId: WORKSPACE_ID, ...agent, systemPrompt: agent.instructions, command: agent.command || 'claude' }};

    // Try multiple endpoints
    const attempts = [
      ['/api/agents', payload],
      [`/api/companies/${{COMPANY_ID}}/agents`, {{...agent, systemPrompt: agent.instructions}}],
    ];

    let created = false;
    for (const [path, data] of attempts) {{
      try {{
        const r = await apiPost(path, data);
        if (r.status >= 200 && r.status < 300) {{
          console.log('✓', agent.emoji, agent.name, 'via POST', path);
          ok++;
          created = true;
          break;
        }} else if (r.status !== 404) {{
          console.log('?', agent.name, path, r.status, r.body.slice(0, 200));
          break;
        }}
      }} catch(e) {{ console.log('ERR', agent.name, e.message); }}
    }}
    if (!created) console.log('✗', agent.emoji, agent.name);
  }}
  console.log(`\\nNode results: ${{ok}}/${{AGENTS.length}} created`);
}}

main().catch(console.error);
"""

    js_file = "/tmp/create_agents.js"
    with open(js_file, "w") as f:
        f.write(js_script)

    print(f"Running Node.js agent creator...")
    result = subprocess.run([node_bin, js_file], capture_output=True, text=True, timeout=60)
    print(result.stdout)
    if result.stderr:
        print("stderr:", result.stderr[:300])
    return "created" in result.stdout and "0/" not in result.stdout


def create_agents_via_postgres(workspace_id, repo_root):
    """Create agents via direct PostgreSQL connection using psycopg2 or psql."""
    pg_port = 54329
    pg_host = "127.0.0.1"
    pg_user = "paperclip"
    pg_db = "paperclip"

    try:
        import psycopg2
        print(f"Using psycopg2 to connect to PostgreSQL on port {pg_port}...")
        return _create_via_psycopg2(pg_host, pg_port, pg_user, pg_db, workspace_id, repo_root)
    except ImportError:
        pass

    psql_bin = find_psql_binary()
    if psql_bin:
        print(f"Using psql binary: {psql_bin}")
        return _create_via_psql(psql_bin, pg_host, pg_port, pg_user, pg_db, workspace_id, repo_root)

    print("Neither psycopg2 nor psql found.")
    print("  As root run: apt-get install -y postgresql-client")
    print("  Then retry: python3 /root/Phi-nance/scripts/create_paperclip_agents.py --db")
    return False


def _inspect_schema(conn):
    """Inspect agent-related tables to understand the schema."""
    cur = conn.cursor()
    # Find tables with 'agent' in name
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    tables = [r[0] for r in cur.fetchall()]
    print(f"  Tables: {tables}")

    agent_tables = [t for t in tables if 'agent' in t.lower()]
    schema = {}
    for tbl in agent_tables:
        cur.execute(f"""
            SELECT column_name, data_type, column_default, is_nullable
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position;
        """, (tbl,))
        cols = cur.fetchall()
        schema[tbl] = cols
        print(f"\n  Table '{tbl}' columns:")
        for col in cols:
            print(f"    {col[0]}: {col[1]} (default={col[2]}, nullable={col[3]})")

    cur.close()
    return schema


def _create_via_psycopg2(host, port, user, db, workspace_id, repo_root):
    """Create agents using psycopg2."""
    import psycopg2
    import uuid

    try:
        conn = psycopg2.connect(host=host, port=port, user=user, dbname=db)
        print("Connected to PostgreSQL successfully!")

        # Inspect schema first
        schema = _inspect_schema(conn)

        agent_table = None
        for tbl in schema:
            if 'agent' in tbl.lower():
                agent_table = tbl
                break

        if not agent_table:
            print("No agent table found! Listing all tables with row counts...")
            cur = conn.cursor()
            cur.execute("""
                SELECT relname, n_live_tup FROM pg_stat_user_tables ORDER BY relname;
            """)
            for row in cur.fetchall():
                print(f"  {row[0]}: {row[1]} rows")
            cur.close()
            conn.close()
            return False

        # Get column names for the agent table
        col_names = [c[0] for c in schema[agent_table]]
        print(f"\nInserting into table '{agent_table}' with columns: {col_names}")

        cur = conn.cursor()
        success = 0

        for agent in AGENTS:
            instructions = read_instructions(agent, repo_root)
            agent_id = str(uuid.uuid4())

            # Build INSERT based on available columns
            row = {"id": agent_id, "workspace_id": workspace_id}

            # Map our fields to whatever columns exist
            field_map = {
                "name": agent["name"],
                "description": agent["description"],
                "model": agent["model"],
                "color": agent["color"],
                "emoji": agent["emoji"],
                "system_prompt": instructions,
                "systemPrompt": instructions,
                "instructions": instructions,
                "prompt": instructions,
            }
            for col, val in field_map.items():
                if col in col_names:
                    row[col] = val

            cols_str = ", ".join(row.keys())
            placeholders = ", ".join(["%s"] * len(row))
            values = list(row.values())

            try:
                cur.execute(
                    f"INSERT INTO {agent_table} ({cols_str}) VALUES ({placeholders})",
                    values
                )
                conn.commit()
                print(f"  ✓ {agent['emoji']} {agent['name']}")
                success += 1
            except Exception as e:
                conn.rollback()
                print(f"  ✗ {agent['name']}: {e}")

        cur.close()
        conn.close()
        print(f"\nDB Results: {success}/{len(AGENTS)} inserted")
        return success == len(AGENTS)

    except Exception as e:
        print(f"PostgreSQL connection failed: {e}")
        return False


def _run_psql(psql_bin, host, port, user, db, sql):
    """Run a SQL command via psql."""
    cmd = [psql_bin, f"-h{host}", f"-p{port}", f"-U{user}", f"-d{db}", "-c", sql]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    return result.returncode, result.stdout, result.stderr


def _create_via_psql(psql_bin, host, port, user, db, workspace_id, repo_root):
    """Create agents using psql binary."""
    import uuid

    # First inspect schema
    rc, out, err = _run_psql(psql_bin, host, port, user, db,
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;")
    if rc != 0:
        print(f"psql connection failed: {err}")
        return False

    print(f"Tables:\n{out}")

    success = 0
    for agent in AGENTS:
        instructions = read_instructions(agent, repo_root).replace("'", "''")
        name = agent['name'].replace("'", "''")
        desc = agent['description'].replace("'", "''")
        agent_id = str(uuid.uuid4())

        sql = f"""
INSERT INTO agents (id, workspace_id, name, description, model, color, emoji, system_prompt, created_at, updated_at)
VALUES ('{agent_id}', '{workspace_id}', '{name}', '{desc}', '{agent['model']}',
        '{agent['color']}', '{agent['emoji']}', '{instructions}', NOW(), NOW())
ON CONFLICT DO NOTHING;
"""
        rc, out, err = _run_psql(psql_bin, host, port, user, db, sql.strip())
        if rc == 0:
            print(f"  ✓ {agent['emoji']} {agent['name']}")
            success += 1
        else:
            print(f"  ✗ {agent['name']}: {err[:150]}")

    print(f"\nDB Results: {success}/{len(AGENTS)} inserted")
    return success == len(AGENTS)


def install_psycopg2():
    """Try to install psycopg2-binary."""
    print("Attempting to install psycopg2-binary...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "psycopg2-binary", "-q"],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode == 0:
        print("psycopg2-binary installed successfully!")
        return True
    else:
        print(f"pip install failed: {result.stderr[:200]}")
        # Try apt
        result2 = subprocess.run(
            ["apt-get", "install", "-y", "-q", "python3-psycopg2"],
            capture_output=True, text=True, timeout=60
        )
        if result2.returncode == 0:
            print("python3-psycopg2 installed via apt!")
            return True
        print(f"apt install also failed: {result2.stderr[:100]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Create Phi Capital agents in Paperclip AI")
    parser.add_argument("--probe", action="store_true", help="Probe API endpoints")
    parser.add_argument("--workspace", help="Workspace ID override")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--dry-run", action="store_true", help="Print agents without creating")
    parser.add_argument("--db", action="store_true", help="Skip API, use PostgreSQL directly")
    # Default repo root: parent of the scripts/ directory (where this file lives)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_repo_root = os.path.dirname(_script_dir) if os.path.basename(_script_dir) == "scripts" else "/root/Phi-nance"
    parser.add_argument("--repo-root", default=_default_repo_root)
    parser.add_argument("--schema", action="store_true", help="Just inspect the DB schema")
    args = parser.parse_args()

    if args.probe:
        _, cid = get_company_id(args.base_url), None
        cid, _ = get_company_id(args.base_url)
        probe_api(args.base_url, company_id=cid)
        return

    workspace_id = args.workspace or get_workspace_id(base_url=args.base_url)
    if not workspace_id:
        print(f"ERROR: Could not auto-detect workspace ID from filesystem or API ({args.base_url})")
        print("Pass --workspace YOUR_WORKSPACE_ID")
        print("  You can find it by running: python3 scripts/create_paperclip_agents.py --probe")
        sys.exit(1)

    print("Phi Capital Agent Creator")
    print("=" * 40)
    print(f"Workspace: {workspace_id}")
    print(f"Repo root: {args.repo_root}")
    print(f"Paperclip: {args.base_url}")
    print()

    if args.dry_run:
        for a in AGENTS:
            print(f"  {a['emoji']} {a['name']} ({a['model']})")
        return

    if args.schema:
        # Just inspect DB schema
        try:
            import psycopg2
        except ImportError:
            if not install_psycopg2():
                sys.exit(1)
            import psycopg2
        conn = psycopg2.connect(host="127.0.0.1", port=54329, user="paperclip", dbname="paperclip")
        _inspect_schema(conn)
        conn.close()
        return

    # Auto-detect company ID
    company_id, company_name = get_company_id(args.base_url)
    if company_id:
        print(f"Company:   {company_name} ({company_id})")
    else:
        print("WARNING: Could not auto-detect company ID from /api/companies")
    print()

    if args.db:
        create_agents_via_postgres(workspace_id, args.repo_root)
        return

    # Default: try API first
    print("Step 1: Probing API endpoints...")
    probe_api(args.base_url, company_id=company_id)

    if company_id:
        print("Step 2: Attempting agent creation via REST API (companyId)...")
        failed = try_create_via_api(company_id, workspace_id, args.base_url, args.repo_root)
    else:
        print("Step 2: Skipped (no company ID).")
        failed = [a["name"] for a in AGENTS]

    if failed:
        print(f"\nStep 3: Trying Node.js approach...")
        ok = create_agents_via_node(company_id, workspace_id, args.repo_root, args.base_url)
        if not ok:
            print(f"\nStep 4: Trying PostgreSQL direct insertion...")
            create_agents_via_postgres(workspace_id, args.repo_root)


if __name__ == "__main__":
    main()
