#!/bin/bash

# Phi-nance VPS Deployment Script
# Targets Ubuntu 24.04 (LTS) x64

echo "🚀 Starting Phi-nance Setup..."

# 1. Update system and install dependencies
echo "📦 Installing system dependencies..."
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip libomp-dev git screen ufw
sudo ufw allow ssh
sudo ufw allow 8501/tcp
sudo ufw --force enable

# 2. Create virtual environment
echo "🐍 Creating virtual environment..."
python3.12 -m venv venv
source venv/bin/activate

# 3. Install Python requirements
echo "📥 Installing Python packages (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Create local data directories
mkdir -p data/cache logs models

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy env: 'cp .env.example .env' then 'nano .env' (set AV_API_KEY, etc.)"
echo "2. Launch in screen: 'screen -S phi-nance' then run: ./start.sh"
echo "   Or: source venv/bin/activate && python -m streamlit run app_streamlit/app.py --server.port 8501 --server.address 0.0.0.0"
echo ""
echo "Press Ctrl+A then D to detach from the screen session."
echo ""
echo "🔍 Troubleshooting Debug Info:"
echo "---------------------------"
echo "Public IP: \$(curl -s https://api.ipify.org)"
echo "Internal Port Status:"
ss -tulnp | grep 8501
echo "Firewall Status:"
sudo ufw status | grep 8501
echo "---------------------------"

