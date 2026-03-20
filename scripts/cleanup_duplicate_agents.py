#!/usr/bin/env python3
"""
Delete duplicate Paperclip agents (those created by running create_paperclip_agents.py twice).

Removes agents whose names end with " 2" (e.g. "Advisor 2", "Chief Trader 2", etc.)
and optionally removes test agents ("test", "test 2", "CEO").

Usage:
    python3 scripts/cleanup_duplicate_agents.py [options]

Options:
    --base-url URL       Paperclip base URL (default: http://localhost:3100)
    --also-test          Also delete agents named "test", "test 2"
    --also-ceo           Also delete agents named "CEO"
    --dry-run            Print what would be deleted without deleting
    --list               Just list all agents
"""

import json
import sys
import argparse
import urllib.request
import urllib.error


BASE_URL = "http://localhost:3100"


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


def list_agents(company_id, base_url):
    """Fetch all agents for the company."""
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


def delete_agent(agent, company_id, base_url):
    """Try several DELETE endpoint patterns."""
    aid = agent.get("id") or agent.get("agentId")
    slug = agent.get("shortname") or agent.get("slug")
    name = agent.get("name", "?")

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

    return False, "no working endpoint found"


def main():
    parser = argparse.ArgumentParser(description="Clean up duplicate Paperclip agents")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--also-test", action="store_true", help="Also delete test/test 2 agents")
    parser.add_argument("--also-ceo", action="store_true", help="Also delete CEO agent")
    parser.add_argument("--dry-run", action="store_true", help="Print without deleting")
    parser.add_argument("--list", action="store_true", help="List all agents and exit")
    args = parser.parse_args()

    company_id, company_name = get_company_id(args.base_url)
    if not company_id:
        print(f"ERROR: Cannot reach Paperclip at {args.base_url} or no company found.")
        print("Make sure --base-url points to your Paperclip instance.")
        sys.exit(1)

    print(f"Company: {company_name} ({company_id})")

    agents = list_agents(company_id, args.base_url)
    if not agents:
        print("No agents found (or could not list them).")
        sys.exit(1)

    print(f"\nFound {len(agents)} agent(s):")
    for a in agents:
        print(f"  [{a.get('id','?')[:8]}] {a.get('name','?')}")

    if args.list:
        return

    # Determine which agents to delete
    to_delete = []
    for a in agents:
        name = a.get("name", "")
        if name.endswith(" 2"):
            to_delete.append(a)
        elif args.also_test and name.lower() in ("test", "test 2"):
            to_delete.append(a)
        elif args.also_ceo and name.upper() == "CEO":
            to_delete.append(a)

    if not to_delete:
        print("\nNothing to delete.")
        return

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Deleting {len(to_delete)} duplicate agent(s):")
    deleted = 0
    for a in to_delete:
        name = a.get("name", "?")
        if args.dry_run:
            print(f"  would delete: {name}")
            continue
        ok, info = delete_agent(a, company_id, args.base_url)
        if ok:
            print(f"  ✓ deleted: {name}")
            deleted += 1
        else:
            print(f"  ✗ failed: {name} — {info}")

    if not args.dry_run:
        print(f"\nDeleted {deleted}/{len(to_delete)} duplicates.")


if __name__ == "__main__":
    main()
