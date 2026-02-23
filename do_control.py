#!/usr/bin/env python3
"""
Control the Phi-nance VPS droplet via the DigitalOcean API.

Uses DIGITALOCEAN_TOKEN from the environment (e.g. from .env).
Target droplet: 165.245.142.100 or first droplet whose name contains "Phi".

Usage:
    python do_control.py status    # show droplet info and state
    python do_control.py reboot    # graceful reboot
    python do_control.py power_off  # power off
    python do_control.py power_on  # power on
"""

import os
import sys

# Optional: load .env so DIGITALOCEAN_TOKEN is set without exporting
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

DO_BASE = "https://api.digitalocean.com/v2"
PHI_IP = "165.245.142.100"


def get_token():
    token = os.environ.get("DIGITALOCEAN_TOKEN", "").strip()
    if not token:
        print("Error: DIGITALOCEAN_TOKEN not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)
    return token


def headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def list_droplets(token):
    r = requests.get(f"{DO_BASE}/droplets", headers=headers(token), timeout=30)
    r.raise_for_status()
    return r.json().get("droplets", [])


def find_phi_droplet(droplets):
    """Return droplet for Phi VPS: by IP 165.245.142.100 or name containing 'Phi'."""
    for d in droplets:
        for v in d.get("networks", {}).get("v4", []):
            if v.get("ip_address") == PHI_IP:
                return d
    for d in droplets:
        if "phi" in (d.get("name") or "").lower():
            return d
    return None


def droplet_public_ip(droplet):
    for v in droplet.get("networks", {}).get("v4", []):
        if v.get("type") == "public":
            return v.get("ip_address")
    return None


def run_action(token, droplet_id, action_type):
    r = requests.post(
        f"{DO_BASE}/droplets/{droplet_id}/actions",
        headers=headers(token),
        json={"type": action_type},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("action", {})


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("status", "reboot", "power_off", "power_on"):
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]
    token = get_token()
    droplets = list_droplets(token)
    droplet = find_phi_droplet(droplets)

    if not droplet:
        print("Error: No droplet found for Phi VPS (IP 165.245.142.100 or name containing 'Phi').", file=sys.stderr)
        sys.exit(1)

    d_id = droplet["id"]
    d_name = droplet["name"]
    d_status = droplet["status"]
    d_ip = droplet_public_ip(droplet)

    if cmd == "status":
        print(f"Droplet: {d_name} (id={d_id})")
        print(f"Status:  {d_status}")
        print(f"IP:      {d_ip}")
        print(f"URL:     http://{d_ip}:8501")
        return

    action_map = {"reboot": "reboot", "power_off": "power_off", "power_on": "power_on"}
    action_type = action_map[cmd]
    action = run_action(token, d_id, action_type)
    aid = action.get("id")
    astatus = action.get("status", "unknown")
    print(f"Sent '{action_type}' to {d_name} (droplet {d_id}). Action id={aid}, status={astatus}.")


if __name__ == "__main__":
    main()
