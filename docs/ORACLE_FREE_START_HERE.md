# Oracle Cloud Free — Start Here (No Experience Needed)

One path, start to finish: create a **free** server with **4 CPU cores and 24 GB RAM**, then run Phi-nance on it. Do the steps in order.

---

## Part 1 — Get an Oracle Cloud account (if you don’t have one)

1. Go to **https://www.oracle.com/cloud/free**
2. Click **Start for free**
3. Sign up (email, country, name). You’ll need a **credit card**; Oracle only checks it, they don’t charge if you use **Always Free** resources.
4. Choose your **region** (e.g. US East — Ashburn). Remember it.
5. Verify email and log in to the **Oracle Cloud Console**.

---

## Part 2 — Create your free server (one big free VM)

### Step 1 — Open “Create Instance”

1. In the Oracle Cloud Console, click the **☰** (hamburger) top-left.
2. Go to **Compute** → **Instances**.
3. Click **Create instance**.

### Step 2 — Name and compartment

- **Name:** type `phinance` (or any name).
- **Compartment:** leave default (your root compartment, e.g. `djvogeli (root)`).

### Step 3 — Placement

- Leave **Availability domain** as “Let Oracle choose” or pick the first one.
- **Fault domain:** “Let Oracle choose” is fine.

### Step 4 — Image (operating system)

- Click **Edit** next to “Image and shape”.
- Under **Image**, pick **Ubuntu 22.04** (or the latest Ubuntu 22).  
  If you don’t see it: **Change image** → **Canonical** → **Ubuntu 22.04** → **Select image**.
- Click **Select image** to confirm.

### Step 5 — Shape (this is where you get 4 CPU + 24 GB free)

- Still in “Edit image and shape”.
- Under **Shape**, click **Change shape**.
- In the left filter, choose **Ampere** (ARM).
- Select **VM.Standard.A1.Flex**.
- Click **Select shape**.
- Back on the shape configuration:
  - **OCPU count:** set to **4**
  - **Memory (GB):** set to **24**
- Click **Save** (or **Done**).  
  This stays in the **Always Free** limit (3,000 OCPU-hours and 18,000 GB-hours per month = one 4 OCPU / 24 GB VM 24/7).

### Step 6 — Networking (so the VM gets a public IP)

- Under **Networking**, leave “Create new virtual cloud network” (or use existing if you already have one).
- Make sure **Assign a public IPv4 address** is **checked**.  
  If you don’t see it, expand the VCN/subnet section; the option is often under “Subnet” or “Public subnet”.
- If you’re creating a new VCN, the default usually creates a **public subnet** and assigns a public IP. Just leave defaults.

### Step 7 — SSH keys (so you can log in from your PC)

You need **one** of these:

**Option A — Generate a new key (easiest on Windows)**

1. On your Windows PC, open **PowerShell**.
2. Run (replace with your email if you like):
   ```powershell
   ssh-keygen -t ed25519 -C "you@email.com" -f "$env:USERPROFILE\.ssh\oci_phinance"
   ```
3. When it asks for a passphrase, press **Enter** twice (no passphrase) or type one and remember it.
4. Two files are created:
   - **Private key:** `C:\Users\YourName\.ssh\oci_phinance` (never share this).
   - **Public key:** `C:\Users\YourName\.ssh\oci_phinance.pub`.

**Option B — Use an existing key**

- If you already have a key (e.g. `id_ed25519.pub` or `id_rsa.pub`), you can use that. You’ll SSH with the matching private key later.

**Paste the public key into Oracle**

1. In the Create Instance form, find **Add SSH keys**.
2. Choose **Paste public keys**.
3. Open your **public** key file in Notepad (e.g. `C:\Users\YourName\.ssh\oci_phinance.pub`). It’s one long line starting with `ssh-ed25519` or `ssh-rsa`.
4. Copy the **entire line** and paste it into the “SSH key” box. You can add a second key if you want; one is enough.
5. Click **Create instance** at the bottom.

Wait 1–2 minutes until the instance state is **Running**. Then go to the instance details page.

### Step 8 — Get the public IP and username

1. On the instance details page, find **Public IP address**. It might show “-” for a minute; refresh until a number appears (e.g. `129.146.xxx.xxx`). Copy it.
2. **Username** for Ubuntu is **ubuntu** (not root, not opc). You’ll use: `ubuntu@<that-IP>`.

---

## Part 3 — Open the firewall (ports 22 and 8501)

Oracle blocks everything by default. You need to allow SSH and the Phi-nance dashboard.

1. On the instance page, under **Primary VNIC**, click the **Subnet** link (e.g. “Public subnet-…”) or the **VNIC** name.
2. You’ll see the subnet; click the **Security List** (e.g. “Default Security List for …”).
3. Click **Add ingress rules**.
4. Add **two** rules (you can do them one by one):

**Rule 1 — SSH**

- **Source CIDR:** `0.0.0.0/0`
- **IP Protocol:** TCP
- **Destination port range:** `22`
- **Description:** SSH (optional)
- Click **Add ingress rules** (or **Save**).

**Rule 2 — Phi-nance dashboard**

- **Add ingress rules** again.
- **Source CIDR:** `0.0.0.0/0`
- **IP Protocol:** TCP
- **Destination port range:** `8501`
- **Description:** Streamlit (optional)
- **Add ingress rules** (or **Save**).

---

## Part 4 — Log in and install Phi-nance

### Step 1 — SSH from your PC

In **PowerShell** (replace with your instance’s public IP and key path):

```powershell
ssh -i "$env:USERPROFILE\.ssh\oci_phinance" ubuntu@129.146.xxx.xxx
```

(If you used a different key file, use that path instead of `oci_phinance`.)

- First time you may see “Are you sure you want to continue connecting?” — type **yes** and Enter.
- You should get a prompt like `ubuntu@instance-name:~$`.

### Step 2 — Install and run the deploy script

Copy and paste these blocks one at a time:

```bash
sudo apt update
sudo apt install -y git
git clone https://github.com/DGator86/Phi-nance.git
cd Phi-nance
chmod +x deploy_vps.sh
./deploy_vps.sh
```

Wait until it says **“Setup complete!”**.

### Step 3 — Create the .env file

```bash
cp .env.example .env
nano .env
```

- Paste your **AV_API_KEY** (and any other keys you use). One line: `AV_API_KEY=your_key_here`
- Save: **Ctrl+O**, **Enter**, then exit: **Ctrl+X**.

### Step 4 — Start the dashboard in the background

```bash
screen -S phi-nance
source venv/bin/activate
streamlit run dashboard.py
```

- When you see “You can now view your Streamlit app”, detach so it keeps running: press **Ctrl+A**, then **D**.

### Step 5 — Open the dashboard in your browser

On your PC, open a browser and go to:

**http://YOUR_PUBLIC_IP:8501**

(Use the same public IP you used for SSH.) You should see the Phi-nance dashboard.

---

## Quick reference

| What | How |
|------|-----|
| **Log in to the server** | `ssh -i ~/.ssh/oci_phinance ubuntu@YOUR_PUBLIC_IP` |
| **Dashboard URL** | http://YOUR_PUBLIC_IP:8501 |
| **Reattach to the running app** | `ssh` in, then `screen -r phi-nance` |
| **Stop the app** | In that screen: **Ctrl+C**, then `exit` |

---

## If something goes wrong

- **“Permission denied (publickey)”** — You’re using the wrong key or wrong user. Use `-i` with the **private** key path and user **ubuntu**.
- **“Connection refused” on port 8501** — Add the **8501** ingress rule in the Security List (Part 3, Rule 2).
- **No public IP** — On the instance page, under Primary VNIC, click the VNIC → Edit and check “Assign a public IPv4 address”.
- **Instance is “Stopped”** — On the instance page, click **Start**.

---

## You already have the small “Phinance” (1 CPU, 1 GB) instance?

You can still use it for a quick test:

1. On that instance’s page, give it a **public IP** (Primary VNIC → Edit → Assign public IPv4).
2. Open **port 22** and **8501** in its **Security List** (same as Part 3).
3. SSH: `ssh -i your_private_key opc@ITS_PUBLIC_IP` (user is **opc** for Oracle Linux).
4. Then run **deploy_oracle.sh** instead of deploy_vps.sh (see [ORACLE_DEPLOY.md](ORACLE_DEPLOY.md)).

That small box will be slow for ML/backtests; the 4 OCPU / 24 GB Ampere instance above is the one that’s “free and really powerful.”
