# VPS Setup Guide (Phi-Bot)

Follow these steps to get Phi-nance running on your new server at **165.245.142.100**.

For a short beta-deploy checklist, see **BETA_DEPLOY.md**.

## 1. Connect to your VPS

Open your terminal (PowerShell or Bash) and run:

```bash
ssh root@165.245.142.100
```

*(Enter your password when prompted)*

## 2. Clone the Repository

```bash
git clone https://github.com/DGator86/Phi-nance.git
cd Phi-nance
```

## 3. Run the Setup Script

Make the script executable and run it:

```bash
chmod +x deploy_vps.sh
./deploy_vps.sh
```

## 4. Configure your API Keys

Create a `.env` file and paste your credentials:

```bash
nano .env
```

*(Paste your AV_API_KEY etc., then press Ctrl+O, Enter, Ctrl+X to save)*

## 5. Launch the Dashboard

We'll use `screen` so the dashboard stays alive after you close your terminal.

```bash
screen -S phi-nance
source venv/bin/activate
streamlit run dashboard.py
```

**To disconnect (keeping it running):**
Press `Ctrl+A` then `D`.

**To reconnect later:**

```bash
screen -r phi-nance
```

---
**Access your dashboard at:**
`http://165.245.142.100:8501`
