# Beta deploy — Phi-nance GUI

Use this checklist to deploy the Streamlit dashboard for beta access.

## Pre-deploy

- [ ] **VPS** — Ubuntu 24.04 (LTS) x64 at your chosen host (e.g. `165.245.142.100`).
- [ ] **Repo** — Code pushed to `MAIN` (or the branch you deploy from).
- [ ] **Secrets** — You have `AV_API_KEY` (and optionally `POLYGON_API_KEY` for L2) ready for `.env` on the server.

## Deploy on the VPS

1. **SSH in**

   ```bash
   ssh root@165.245.142.100
   ```

2. **Clone and enter project**

   ```bash
   git clone https://github.com/DGator86/Phi-nance.git
   cd Phi-nance
   ```

3. **Run setup script**

   ```bash
   chmod +x deploy_vps.sh
   ./deploy_vps.sh
   ```

4. **Create `.env`**

   ```bash
   cp .env.example .env
   nano .env
   ```

   Fill in `AV_API_KEY` (and others if needed). Save (Ctrl+O, Enter, Ctrl+X).

5. **Start the dashboard in `screen`**

   ```bash
   screen -S phi-nance
   source venv/bin/activate
   streamlit run dashboard.py
   ```

   Detach: **Ctrl+A** then **D**.

## Post-deploy

- **URL:** `http://165.245.142.100:8501`
- **Reattach to session:** `ssh` in, then `screen -r phi-nance`
- **Smoke test:** Open the URL, confirm all 6 tabs load (ML Model Status, Fetch Data, MFT Blender, Phi-Bot, Backtests, System Status).

## Control VPS from your machine (DigitalOcean API)

With `DIGITALOCEAN_TOKEN` in your local `.env`, you can manage the Phi droplet from your laptop:

```bash
python do_control.py status    # droplet name, status, IP, dashboard URL
python do_control.py reboot    # graceful reboot
python do_control.py power_off  # power off
python do_control.py power_on  # power on
```

The script finds the droplet by IP `165.245.142.100` or by name containing "Phi". No SSH needed for power/reboot.

## Optional (later)

- Put **nginx** in front for HTTPS and a friendly hostname.
- Use **systemd** or **supervisor** instead of `screen` for restarts and logging.

For more detail, see **VPS_SETUP.md**.
