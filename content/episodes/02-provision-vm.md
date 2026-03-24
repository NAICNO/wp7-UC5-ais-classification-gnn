# Provisioning a VM

```{objectives}
- Provision a GPU-enabled virtual machine on the NAIC Orchestrator
- Connect to the VM via SSH with proper key configuration
- Verify GPU availability and driver installation
- Understand the NAIC Orchestrator VM environment
```

## NAIC Orchestrator

The NAIC Orchestrator at [orchestrator.naic.no](https://orchestrator.naic.no) provides virtual machines pre-configured for AI workloads. These VMs come with NVIDIA GPU drivers, CUDA toolkit, and standard ML libraries pre-installed.

### Step 1: Request a VM

1. Log in to the NAIC Orchestrator portal with your Feide credentials
2. Select **"Create VM"** from the dashboard
3. Choose a GPU-enabled flavor:
   - **1x NVIDIA T4** (16 GB VRAM) -- recommended for this demonstrator
   - **1x NVIDIA A100** (40/80 GB VRAM) -- for large-scale experiments or multiple UCs
4. Select **Ubuntu 22.04** as the operating system
5. Upload your SSH public key (or use one already registered)
6. Note the assigned IP address once the VM is ready

```{admonition} VM Startup Time
:class: tip

VM provisioning typically takes 2-5 minutes. The portal status will change from "Building" to "Active" when the VM is ready to accept SSH connections. If the VM stays in "Building" for more than 10 minutes, try deleting and recreating it.
```

### Step 2: Connect via SSH

```bash
ssh -i ~/.ssh/naic-vm.pem ubuntu@<YOUR_VM_IP>
```

Replace `<YOUR_VM_IP>` with your actual VM IP address.

#### SSH Troubleshooting

If you cannot connect, check these common issues:

| Problem | Solution |
|---------|----------|
| `Permission denied (publickey)` | Ensure your key file has correct permissions: `chmod 600 ~/.ssh/naic-vm.pem` |
| `Connection timed out` | Verify the VM is in "Active" state in the portal; check that your network allows outbound SSH (port 22) |
| `Host key verification failed` | Remove the old entry: `ssh-keygen -R <YOUR_VM_IP>` and reconnect |
| `Connection refused` | The SSH daemon may not have started yet; wait 1-2 minutes after VM creation |

#### SSH Config (Optional)

For convenience, add the VM to your SSH config file (`~/.ssh/config`):

```bash
Host naic-vm
    HostName <YOUR_VM_IP>
    User ubuntu
    IdentityFile ~/.ssh/naic-vm.pem
    StrictHostKeyChecking no
```

Then connect with just `ssh naic-vm`.

### Step 3: Initialize the VM

```bash
curl -O https://raw.githubusercontent.com/NAICNO/wp7-UC5-ais-classification-gnn/main/vm-init.sh
chmod +x vm-init.sh
./vm-init.sh
```

This installs system packages, checks for GPU drivers, and configures CUDA.

### Step 4: Verify GPU Availability

After initialization, confirm that the GPU is recognized:

```bash
# Check NVIDIA driver and GPU
nvidia-smi
```

You should see output showing your GPU model, driver version, and CUDA version. For example:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
+-------------------------------+----------------------+----------------------+
```

If `nvidia-smi` is not found or shows an error, the GPU drivers may need to be installed:

```bash
# Install NVIDIA drivers (if not pre-installed)
sudo apt-get update
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

After rebooting, reconnect via SSH and verify with `nvidia-smi` again.

```{admonition} CUDA Toolkit Version
:class: tip

The CUDA version shown by `nvidia-smi` is the *maximum supported* CUDA version for the driver. PyTorch and DGL ship their own CUDA runtime libraries, so the PyTorch CUDA version does not need to exactly match the driver version -- it just needs to be equal to or lower than the driver version.
```

