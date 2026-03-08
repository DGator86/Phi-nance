# Installation

## Requirements

- Python 3.10+
- `pip`
- (Optional) `python3.12-venv` system package for creating virtual environments

## Local setup

```bash
git clone https://github.com/DGator86/Phi-nance.git
cd Phi-nance
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env
```

## Verify installation

```bash
python engine_health.py
pytest
```

## Run app

```bash
streamlit run app_streamlit/main.py --server.headless true
```
