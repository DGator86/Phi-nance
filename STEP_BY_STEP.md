# Phi-nance — Everything You Need to Do (Step by Step)

Do these in order. Your VPS IP is **165.245.142.100** (replace with yours if different).

---

## Part A — On your Windows PC (one-time setup)

### Step 1 — Open the project

- Open the `Phi-nance` folder in Cursor (or your editor).
- Open PowerShell or Terminal in that folder.

### Step 2 — Push latest code to GitHub (if you have changes)

```powershell
git add -A
git commit -m "Beta deploy ready"
git push origin MAIN
```

*(If you don’t use git yet, skip this. You’ll clone from GitHub on the VPS; make sure the code you want is at <https://github.com/DGator86/Phi-nance>.)*

### Step 3 — Create your local `.env` file

- In the `Phi-nance` folder, copy the example env file:
  - **PowerShell:** `Copy-Item .env.example .env`
  - **Or:** Create a new file named `.env` (no name before the dot).
- Open `.env` and set these (use your real values):

```
AV_API_KEY=PLN25H3ESMM1IRBN
AV_MCP_TOKEN=G7jhPGMv69WcDYerwJ6VZ2Tw6upJ
DIGITALOCEAN_TOKEN=Dop_v1_b380c96df515cc38acdd5e4fb7bad2b3e1e1fd14d894a7d1b05a7a0ed7e4b2fa
```

- Save and close. **Do not commit `.env`** — it’s gitignored.

### Step 4 — (Optional) Test DigitalOcean control from your PC

```powershell
pip install -r requirements.txt
python do_control.py status
```

- You should see your droplet name, status, and IP. If you get “DIGITALOCEAN_TOKEN not set”, fix `.env` and run again.

---

## Part B — On the VPS (first-time deploy)

*You need SSH access: root password or SSH key for `root@165.245.140.115`.*

### Step 5 — Connect to the VPS

```powershell
ssh root@165.245.142.100
```

- Enter the root password when prompted. You should see a Linux prompt.

### Step 6 — Clone the repo and go into it

```bash
git clone https://github.com/DGator86/Phi-nance.git
cd Phi-nance
```

### Step 7 — Run the setup script

```bash
chmod +x deploy_vps.sh
./deploy_vps.sh
```

- Wait until it says “Setup complete!” (installs Python, venv, packages, configures firewall, creates `data/cache`, `logs`, `models`).

### Step 8 — Create `.env` on the VPS

```bash
cp .env.example .env
nano .env
```

- Paste at least:
  - `AV_API_KEY=...` (your Alpha Vantage key)
  - `AV_MCP_TOKEN=...` (if you use it)
- **Do not put your DigitalOcean token on the VPS** unless you plan to run `do_control.py` there (you’ll run it from your PC).
- Save: **Ctrl+O**, **Enter**, then exit: **Ctrl+X**.

### Step 9 — Start the dashboard in a `screen` session

```bash
screen -S phi-nance
source venv/bin/activate
streamlit run dashboard.py
```

- You should see “You can now view your Streamlit app in your browser” and a URL.
- **Detach from screen** (app keeps running): press **Ctrl+A**, then **D**.

### Step 10 — Leave the VPS

```bash
exit
```

---

## Part C — After deploy (from your PC)

### Step 11 — Open the dashboard in a browser

- Go to: **<http://165.245.142.100:8501>**
- Check that all 6 tabs load: ML Model Status, Fetch Data, MFT Blender, Phi-Bot, Backtests, System Status.

### Step 12 — When you need to get back into the dashboard session on the VPS

```powershell
ssh root@165.245.142.100
screen -r phi-nance
```

- To detach again: **Ctrl+A**, then **D**, then `exit`.

### Step 13 — When you want to reboot or power the VPS from your PC (no SSH)

From the `Phi-nance` folder on your PC:

```powershell
python do_control.py status    # see droplet name, status, IP, dashboard URL
python do_control.py reboot    # reboot the VPS
python do_control.py power_off # power off
python do_control.py power_on  # power on
```

---

## Quick reference

| What you want | What to do |
|---------------|------------|
| Open dashboard | Browser: `http://165.245.142.100:8501` |
| Reattach to dashboard process | `ssh root@165.245.142.100` → `screen -r phi-nance` |
| Reboot VPS from PC | `python do_control.py reboot` |
| Check VPS status from PC | `python do_control.py status` |
| Update app on VPS | SSH in → `cd Phi-nance` → `git pull` → reattach screen, restart Streamlit (Ctrl+C, then `streamlit run dashboard.py`) |

---

**If your VPS IP is not 165.245.142.100**, replace it everywhere above. Get the IP from DigitalOcean’s control panel or by running `python do_control.py status` from your PC.
