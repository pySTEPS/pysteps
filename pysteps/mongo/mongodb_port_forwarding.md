# Port Forwarding to Remote MongoDB on Localhost

This guide explains how to open port **27018** on your local machine and forward it to a **remote MongoDB** instance running on port **27017**, either temporarily or permanently using `systemd` and `autossh`.

---

## 🔁 SSH Port Forwarding (Temporary)

To forward local port `27018` to the remote MongoDB server on port `27017`, run:

```bash
ssh -L 27018:localhost:27017 your_user@remote_host
```

Once the tunnel is active, connect to MongoDB using:

```bash
mongosh --port 27018
```

Or with a URI:

```bash
mongodb://localhost:27018
```

---

## 🔄 Making Port Forwarding Persistent with systemd and autossh

To create a self-healing SSH tunnel that auto-reconnects on failure, use `autossh` with a `systemd` user service.

### 1. Install autossh

On Fedora:

```bash
sudo dnf install autossh
```

On Debian/Ubuntu:

```bash
sudo apt install autossh
```

---

### 2. Set up SSH keys

```bash
ssh-keygen
ssh-copy-id radar@remote_host
```

Make sure `ssh radar@remote_host` works without a password.

---

### 3. Create the systemd user service

Create the file:  
`~/.config/systemd/user/mongodb-tunnel.service`

```ini
[Unit]
Description=Persistent SSH tunnel to radar MongoDB
After=network.target

[Service]
Environment=AUTOSSH_GATETIME=0
ExecStart=/usr/bin/autossh -M 0 -N -L 27018:10.8.0.41:27017 radar
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

> - Replace `10.8.0.41` with the IP of the remote MongoDB server (not necessarily `localhost` on the remote if it’s bound to a specific interface).
> - `radar` is your SSH alias or username. Make sure it’s configured in `~/.ssh/config` if using an alias.

---

### 4. Enable and start the tunnel

```bash
systemctl --user daemon-reexec
systemctl --user daemon-reload
systemctl --user enable mongodb-tunnel
systemctl --user start mongodb-tunnel
```

To check the status:

```bash
systemctl --user status mongodb-tunnel
```

---

### 5. Optional: Ensure ssh-agent is running

Add to your shell startup script:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
```

Or use your desktop’s SSH key manager.

---

## ✅ Verifying the Tunnel

Once active, test it with:

```bash
mongosh --port 27018
```

---

## 🔐 Security Note

- Keep your SSH key safe with a passphrase.
- Use `ufw`, `firewalld`, or similar to restrict access if needed.
