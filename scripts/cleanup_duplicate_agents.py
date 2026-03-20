#!/usr/bin/env python3
"""
Delete duplicate Paperclip agents (those created by running create_paperclip_agents.py twice).

Removes agents whose names end with " 2" (e.g. "Advisor 2", "Chief Trader 2", etc.)
and optionally removes test agents ("test", "test 2", "CEO").

Usage:
    python3 scripts/cleanup_duplicate_agents.py [options]

Options:
    --also-test          Also delete agents named "test", "test 2"
    --also-ceo           Also delete agents named "CEO"
    --dry-run            Print what would be deleted without deleting
    --list               Just list all agents
    --base-url URL       Paperclip base URL (default: http://localhost:3100)
    --db                 Use direct PostgreSQL deletion instead of API
"""

import json
import sys
import argparse
import subprocess
import urllib.request
import urllib.error


BASE_URL = "http://localhost:3100"
PG_HOST = "127.0.0.1"
PG_PORT = 54329
PG_USER = "paperclip"
PG_DB = "paperclip"


# Names to always treat as duplicates (end with " 2")
# Plus optional extras controlled by flags
ALWAYS_DELETE_SUFFIX = " 2"


def api_request(path, method="GET", data=None, base_url=BASE_URL):
    url = base_url + path
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return None, str(e)


def get_company_id(base_url):
    status, body = api_request("/api/companies", base_url=base_url)
    if status == 200:
        try:
            companies = json.loads(body)
            if companies and isinstance(companies, list):
                return companies[0]["id"], companies[0].get("name", "")
        except Exception:
            pass
    return None, None


def list_agents_api(company_id, base_url):
    for path in [
        f"/api/agents?companyId={company_id}",
        f"/api/companies/{company_id}/agents",
        "/api/agents",
    ]:
        status, body = api_request(path, base_url=base_url)
        if status == 200:
            try:
                data = json.loads(body)
                agents = data if isinstance(data, list) else data.get("agents", data.get("data", []))
                if isinstance(agents, list):
                    return agents
            except Exception:
                pass
    return []


def delete_agent_api(agent, company_id, base_url):
    aid = agent.get("id") or agent.get("agentId")
    slug = agent.get("shortname") or agent.get("slug")

    endpoints = []
    if aid:
        endpoints += [
            f"/api/agents/{aid}",
            f"/api/agents/{aid}?companyId={company_id}",
            f"/api/companies/{company_id}/agents/{aid}",
        ]
    if slug:
        endpoints += [
            f"/api/agents/{slug}?companyId={company_id}",
            f"/api/companies/{company_id}/agents/{slug}",
        ]

    for path in endpoints:
        status, body = api_request(path, method="DELETE", base_url=base_url)
        if status and 200 <= status < 300:
            return True, path
        if status and status not in (404, 405):
            return False, f"HTTP {status}: {body[:120]}"

    return False, "no working DELETE endpoint found"


# ── PostgreSQL path ────────────────────────────────────────────────────────────

def _run_psql(sql):
    """Run SQL via psql and return (returncode, stdout, stderr)."""
    psql = _find_psql()
    if not psql:
        return 1, "", "psql binary not found"
    cmd = [psql, f"-h{PG_HOST}", f"-p{PG_PORT}", f"-U{PG_USER}", f"-d{PG_DB}",
           "-t", "-A", "-c", sql]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    return r.returncode, r.stdout, r.stderr


def _find_psql():
    for p in ["/usr/bin/psql", "/usr/local/bin/psql"]:
        import os
        if os.path.exists(p):
            return p
    r = subprocess.run(["which", "psql"], capture_output=True, text=True)
    if r.returncode == 0:
        return r.stdout.strip()
    return None


def _find_agent_table():
    rc, out, err = _run_psql(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name ILIKE '%agent%' ORDER BY table_name;"
    )
    if rc != 0:
        return None, err
    tables = [t.strip() for t in out.strip().splitlines() if t.strip()]
    return (tables[0] if tables else None), None


def cleanup_via_db(names_to_delete, dry_run):
    """Delete agents by name directly from PostgreSQL."""
    table, err = _find_agent_table()
    if not table:
        print(f"ERROR: Could not find agent table in DB: {err}")
        return False

    print(f"Using DB table: {table}")

    # List matching agents first
    quoted = ", ".join(f"'{n.replace(chr(39), chr(39)*2)}'" for n in names_to_delete)
    rc, out, err = _run_psql(f"SELECT id, name FROM {table} WHERE name IN ({quoted});")
    if rc != 0:
        print(f"ERROR querying agents: {err}")
        return False

    rows = [line.split("|") for line in out.strip().splitlines() if "|" in line]
    if not rows:
        print("No matching agents found in DB.")
        return True

    print(f"\nFound {len(rows)} agent(s) to delete:")
    for row in rows:
        print(f"  [{row[0][:8]}] {row[1]}")

    if dry_run:
        print("\n[DRY RUN] No changes made.")
        return True

    rc, out, err = _run_psql(f"DELETE FROM {table} WHERE name IN ({quoted});")
    if rc == 0:
        print(f"\n✓ Deleted {len(rows)} agent(s) from DB.")
        return True
    else:
        print(f"\nERROR deleting: {err}")
        return False


# ── main ───────────────────────────────────────────────────────────────────────

def should_delete(name, also_test, also_ceo):
    if name.endswith(ALWAYS_DELETE_SUFFIX):
        return True
    if also_test and name.lower() in ("test", "test 2"):
        return True
    if also_ceo and name.upper() == "CEO":
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Clean up duplicate Paperclip agents")
    parser.add_argument("--also-test", action="store_true", help='Also delete "test" and "test 2" agents')
    parser.add_argument("--also-ceo", action="store_true", help='Also delete "CEO" agent')
    parser.add_argument("--dry-run", action="store_true", help="Print without deleting")
    parser.add_argument("--list", action="store_true", help="List all agents and exit")
    parser.add_argument("--base-url", default=BASE_URL, help="Paperclip URL (default: http://localhost:3100)")
    parser.add_argument("--db", action="store_true", help="Use PostgreSQL directly instead of API")
    args = parser.parse_args()

    # ── DB path ──
    if args.db:
        # Build the list of names to delete without needing the API
        # We delete by pattern: anything ending in " 2", plus optionals
        rc, out, err = _run_psql("SELECT DISTINCT name FROM agents ORDER BY name;")
        if rc != 0:
            # try to find table name first
            table, terr = _find_agent_table()
            if table:
                rc, out, err = _run_psql(f"SELECT DISTINCT name FROM {table} ORDER BY name;")

        all_names = [line.strip() for line in out.strip().splitlines() if line.strip()]
        if not all_names:
            print(f"ERROR: Could not list agents from DB.\n{err}")
            sys.exit(1)

        print(f"All agents in DB ({len(all_names)}):")
        for n in all_names:
            print(f"  {n}")

        names_to_delete = [n for n in all_names if should_delete(n, args.also_test, args.also_ceo)]
        if not names_to_delete:
            print("\nNothing to delete.")
            return

        cleanup_via_db(names_to_delete, args.dry_run)
        return

    # ── API path ──
    company_id, company_name = get_company_id(args.base_url)
    if not company_id:
        print(f"ERROR: Cannot reach Paperclip at {args.base_url}")
        print("Try: python3 scripts/cleanup_duplicate_agents.py --db")
        sys.exit(1)

    print(f"Company: {company_name} ({company_id})")

    agents = list_agents_api(company_id, args.base_url)
    if not agents:
        print("No agents returned. Try --db for direct database deletion.")
        sys.exit(1)

    print(f"\nAll agents ({len(agents)}):")
    for a in agents:
        print(f"  [{a.get('id','?')[:8]}] {a.get('name','?')}")

    if args.list:
        return

    to_delete = [a for a in agents if should_delete(a.get("name", ""), args.also_test, args.also_ceo)]

    if not to_delete:
        print("\nNothing to delete.")
        return

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Deleting {len(to_delete)} agent(s):")
    deleted = 0
    for a in to_delete:
        name = a.get("name", "?")
        if args.dry_run:
            print(f"  would delete: {name}")
            continue
        ok, info = delete_agent_api(a, company_id, args.base_url)
        if ok:
            print(f"  ✓ {name}")
            deleted += 1
        else:
            print(f"  ✗ {name} — {info}")

    if not args.dry_run:
        print(f"\nDeleted {deleted}/{len(to_delete)}.")


if __name__ == "__main__":
    main()
