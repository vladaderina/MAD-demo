#cloud-config
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    groups: sudo
    shell: /bin/bash
    ssh_authorized_keys:
      - ${file("~/.ssh/id_rsa.pub")}

# Настройка сети с фиксированным IP внутри подсети
write_files:
  - path: /etc/netplan/50-cloud-init.yaml
    content: |
      network:
        version: 2
        ethernets:
          ens5:
            dhcp4: no
            addresses:
              - 10.0.0.10/24
            gateway4: 10.0.0.1
            nameservers:
              addresses:
                - 8.8.8.8
                - 1.1.1.1
runcmd:
  - netplan apply
