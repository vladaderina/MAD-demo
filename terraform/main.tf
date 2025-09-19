terraform {
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.98.0"
    }
  }
  required_version = ">= 1.6.0"
}

provider "yandex" {
  folder_id = var.yc_folder_id
  cloud_id  = var.yc_cloud_id
  zone      = var.yc_zone
}

# Сеть
resource "yandex_vpc_network" "this" {
  name = "${var.project_name}-network"
}

resource "yandex_vpc_subnet" "this" {
  name           = "${var.project_name}-subnet"
  zone           = var.yc_zone
  network_id     = yandex_vpc_network.this.id
  v4_cidr_blocks = ["10.0.0.0/24"]
}

# Firewall
resource "yandex_vpc_security_group" "ssh" {
  name       = "${var.project_name}-sg"
  network_id = yandex_vpc_network.this.id

  ingress {
    description    = "Allow SSH"
    protocol       = "TCP"
    port           = 22
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    protocol       = "ANY"
    description    = "Allow all egress"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
}

# Статический IP
resource "yandex_vpc_address" "static_ip" {
  name = "${var.project_name}-ip"
}

# ВМ
resource "yandex_compute_instance" "vm" {
  name        = "${var.project_name}-vm"
  platform_id = "standard-v3"
  zone        = var.yc_zone

  resources {
    cores  = var.vm_cores
    memory = var.vm_memory
    core_fraction = 100
  }

  boot_disk {
    initialize_params {
      image_id = var.yc_image_id
      size     = var.vm_disk_size
    }
  }

  network_interface {
    subnet_id          = yandex_vpc_subnet.this.id
    nat                = true
    nat_ip_address     = yandex_vpc_address.static_ip.external_ipv4_address.0.address
    security_group_ids = [yandex_vpc_security_group.ssh.id]
  }

  metadata = {
    ssh-keys  = "ubuntu:${file(var.yc_ssh_key_path)}"
    user-data = file("${path.module}/cloud-init.yaml")
  }
}
