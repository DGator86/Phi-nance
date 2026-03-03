# Deploy Phi-nance on Oracle Cloud (OCI)

Your instance **Phinance** is on Oracle Cloud. Here’s how to get it reachable and what to do about its size.

---

## Your current instance (E2.1.Micro)

| Item | Value |
|------|--------|
| **Shape** | VM.Standard.E2.1.Micro |
| **OCPU** | 1 |
| **Memory** | 1 GB |
| **OS** | Oracle Linux 9 |
| **Username** | `opc` (not root) |
| **Region** | iad (e.g. Ashburn) |

**Problem:** 1 OCPU and 1 GB RAM are **below** the minimum we recommend for Phi-nance (2 vCPU / 4 GB). The dashboard may start, but ML training and backtests will be slow or run out of memory.

**Options:**
1. **Use this instance for a quick test** — Assign a public IP, deploy, and see how it behaves (expect slowness/OOM on heavy tabs).
2. **Switch to the free Ampere (ARM) shape** — Create a new instance with **VM.Standard.A1.Flex**: **4 OCPU, 24 GB RAM**, same Always Free tier. Then deploy there and (optionally) terminate the Micro.

---

## 1. Assign a public IP (required for SSH)

Your instance needs a **public IP** so you can connect from the internet.

1. In OCI Console go to **Compute** → **Instances** → **Phinance**.
2. Under **Instance access** (or **Primary VNIC**), click **Edit** (or open the subnet/VNIC).
3. Ensure the instance is in a **public subnet** and that the VNIC has a **public IPv4**:
   - Either check **“Assign a public IPv4 address”** (ephemeral) and save,  
   - Or create a **Reserved Public IP** under **Networking** → **IP Management** → **Reserved Public IPs**, then attach it to this instance’s private IP.
4. Open **Ingress** for SSH: **Networking** → **Virtual Cloud Networks** → **Phinance** → **Security Lists** → default list → **Add Ingress Rule**:
   - Source: `0.0.0.0/0`
   - IP Protocol: TCP
   - Destination port: 22
   - (And for the dashboard later) TCP port **8501** (Streamlit).

After saving, note the **Public IP** shown on the instance page.

---

## 2. Connect via SSH

From your laptop (PowerShell or Bash):

```bash
ssh opc@<PUBLIC_IP>
```

Use the SSH key you chose when creating the instance (or the one you added to the instance).  
You are **opc**, not root; use `sudo` when you need to install packages.

---

## 3. Deploy on this instance (Oracle Linux 9)

The main `deploy_vps.sh` is for **Ubuntu** (apt). On **Oracle Linux 9** use the Oracle-specific script:

```bash
# On the instance, as opc:
curl -sL https://github.com/DGator86/Phi-nance/raw/MAIN/deploy_oracle.sh -o deploy_oracle.sh
chmod +x deploy_oracle.sh
./deploy_oracle.sh
```

Or clone the repo and run it:

```bash
sudo dnf install -y git
git clone https://github.com/DGator86/Phi-nance.git
cd Phi-nance
chmod +x deploy_oracle.sh
./deploy_oracle.sh
```

Then create `.env`, start the dashboard in `screen` (same as in [STEP_BY_STEP.md](STEP_BY_STEP.md)), and open `http://<PUBLIC_IP>:8501`.  
Ensure the VCN **security list** allows **ingress TCP 8501** from `0.0.0.0/0` (or your IP).

---

## 4. (Recommended) Use a bigger free instance — Ampere A1

For a **free** and **much more capable** box:

1. **Compute** → **Instances** → **Create Instance**.
2. **Name:** e.g. `Phinance-A1`.
3. **Placement:** same region (e.g. iad), any AD.
4. **Image:** **Ubuntu 22.04** or **Oracle Linux** (if you prefer).
5. **Shape:** **Ampere** → **VM.Standard.A1.Flex**.
6. **OCPU:** **4**, **Memory:** **24 GB** (within Always Free: 3,000 OCPU-hours and 18,000 GB-hours per month).
7. **Networking:** create or select a **public subnet**, **Assign a public IPv4 address**.
8. **SSH keys:** upload your public key.
9. Create the instance.

Then:

- In **Security List**, add **Ingress** for **TCP 22** and **TCP 8501** from `0.0.0.0/0`.
- SSH: `ssh ubuntu@<new-public-ip>` (if Ubuntu) or `ssh opc@<new-public-ip>` (if Oracle Linux).
- On **Ubuntu** use **deploy_vps.sh**; on **Oracle Linux** use **deploy_oracle.sh**.

That gives you **4 OCPU and 24 GB RAM** at no cost, which is enough for Phi-nance.

---

## 5. Summary

| Step | Action |
|------|--------|
| **Right now** | Assign a **public IP** to **Phinance** and open **TCP 22** (and **8501**) in the VCN security list. |
| **Connect** | `ssh opc@<PUBLIC_IP>`. |
| **Deploy on current (Micro)** | Use **deploy_oracle.sh** (Oracle Linux 9). Expect limited performance. |
| **Better experience** | Create an **A1.Flex** instance (4 OCPU, 24 GB), use **deploy_vps.sh** (Ubuntu) or **deploy_oracle.sh** (OL), then use that for Phi-nance. |
