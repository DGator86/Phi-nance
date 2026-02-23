#!/bin/bash
# Phi-nance deploy on Oracle Linux 9 (dnf, opc user)
# Run as: ./deploy_oracle.sh

set -e
echo "🚀 Phi-nance setup (Oracle Linux 9)..."

# 1. Install dependencies (use sudo; default python3 is 3.9 on OL9)
echo "📦 Installing system packages..."
sudo dnf install -y python3 python3-pip python3-devel git screen gcc gcc-c++ make

# 2. Create venv and install Python packages
echo "🐍 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "📥 Installing Python packages (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Directories
mkdir -p data/cache logs models

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Create .env:  nano .env   (AV_API_KEY, etc.)"
echo "2. Start dashboard in screen:"
echo "   screen -S phi-nance"
echo "   source venv/bin/activate"
echo "   streamlit run dashboard.py"
echo "3. Detach: Ctrl+A then D"
echo "4. Open: http://<this-server-public-ip>:8501"
echo "   (Ensure VCN Security List allows ingress TCP 8501)"
