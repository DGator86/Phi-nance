# Deploy Phi-nance to VPS (ATL1)

**VPS:** Phinance — 8 GB RAM / 2 vCPUs / 160 GB Disk  
**OS:** Ubuntu 24.04 (LTS) x64  
**IPv4:** 165.245.142.100  

---

## 1. SSH into the VPS

From your Windows machine (PowerShell or terminal):

```bash
ssh root@165.245.142.100
```

(Or use your non-root user if you have one, e.g. `ssh ubuntu@165.245.142.100`.)

---

## 2. Clone the repo and go to project dir

```bash
git clone https://github.com/DGator86/Phi-nance.git ~/Phi-nance
cd ~/Phi-nance
```

---

## 3. Run the deploy script

```bash
chmod +x deploy/deploy_vps.sh
./deploy/deploy_vps.sh
```

This will:

- Install Python 3.12, venv, pip, git, screen, ufw
- Allow SSH and port 8501 in the firewall
- Create a virtualenv and install `requirements.txt`
- Create `data/cache`, `logs`, `models`

---

## 4. Configure environment

```bash
cp .env.example .env
nano .env
```

Set at least:

- `AV_API_KEY` — your Alpha Vantage API key (required for market data)

Save (Ctrl+O, Enter) and exit (Ctrl+X).

---

## 5. Start the app (screen)

```bash
screen -S phi-nance
source venv/bin/activate
./start.sh
```

Detach from screen: **Ctrl+A**, then **D**. The app keeps running.

Reattach later:

```bash
screen -r phi-nance
```

---

## 6. Open in browser

- **Direct Streamlit:** http://165.245.142.100:8501  
- If you set up Nginx (optional): http://165.245.142.100 (port 80)

---

## Optional: Nginx reverse proxy (port 80)

To serve on port 80 and proxy to Streamlit:

```bash
sudo apt install -y nginx
sudo cp ~/Phi-nance/streamlit_nginx.conf /etc/nginx/sites-available/phinance
sudo ln -sf /etc/nginx/sites-available/phinance /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
sudo ufw allow 80/tcp
sudo ufw reload
```

Then open: http://165.245.142.100

---

## Later: pull and restart

```bash
cd ~/Phi-nance
git pull origin MAIN
source venv/bin/activate
./start.sh
```

If the app is already running in a screen session, attach with `screen -r phi-nance`, stop it (Ctrl+C), then run `./start.sh` again.
