output "vm_public_ip" {
  description = "Публичный IP-адрес ВМ"
  value       = yandex_vpc_address.static_ip.external_ipv4_address[0].address
}

output "vm_private_ip" {
  description = "Приватный IP-адрес ВМ"
  value       = yandex_compute_instance.vm.network_interface[0].ip_address
}
