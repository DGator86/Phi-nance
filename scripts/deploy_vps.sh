#!/bin/bash

# Phi-nance VPS Deployment Script
# Targets Ubuntu 24.04 (LTS) x64

echo "🚀 Starting Phi-nance Setup..."

# 1. Update system and install dependencies
echo "📦 Installing system dependencies..."
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip libomp-dev git screen ufw
sudo ufw allow ssh
sudo ufw allow 8080/tcp
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
echo "1. Create your .env file: 'nano .env'"
echo "2. Launch the export API in a screen session:"
echo "   screen -S phi-nance"
echo "   source venv/bin/activate"
echo "   ./start.sh"
echo ""
echo "Press Ctrl+A then D to detach from the screen session."
echo ""
echo "🔍 Troubleshooting Debug Info:"
echo "---------------------------"
echo "Public IP: \$(curl -s https://api.ipify.org)"
echo "Internal Port Status:"
ss -tulnp | grep 8080
echo "Firewall Status:"
sudo ufw status | grep 8501
echo "---------------------------"

