# Создание сервера

На этой странице описан вариант развертывания **всех компонентов MAD на одном сервере**:

- база данных (PostgreSQL),
- система мониторинга (Grafana, VictoriaMetrics, Loki),
- кластер Kubernetes (Minikube или k3s),
- сервис поиска аномалий (MAD).

> Это **MVP** подход. Для продакшн сред рекомендуется разделение компонентов на отдельные VM.

---

## Terraform

Развёртывание автоматизируется через **Terraform-модуль**, который создаёт виртуальную машину с заданными ресурсами и конфигурацией.

Terraform создаёт следующие ресурсы в Yandex Cloud:

- VPC-сеть и подсеть (10.0.0.0/24)

- Security Group с правилом для SSH (порт 22)

- Статический IP-адрес для сервера

- Виртуальную машину с указанными параметрами (CPU, RAM, диск)

- cloud-init конфигурацию, которая назначает статический приватный IP и разворачивает пользователя ubuntu с SSH-ключом

---

### План действий по запуску Terraform

1. Установить Terraform
```bash
sudo apt install terraform   # Ubuntu/Debian
```
1. Склонировать репозиторий с инфраструктурой и перейти в папку `terraform`.
2. Указать параметры в файле `terraform.tfvars`:
```tf
yc_folder_id     = "ваш-folder-id"
yc_cloud_id      = "ваш-cloud-id"
yc_zone          = "ru-central1-a"
yc_ssh_key_path  = "~/.ssh/id_rsa.pub
```
1. Подготовить окружение:
```bash
terraform init
terraform workspace new dev
terraform workspace select dev
```
1. Настроить переменные `terraform/variables.tf`:
```tf
folder_id: каталог Yandex Cloud

zone: зона размещения VM

env: окружение (dev)

ssh_key: публичный ключ для доступа

instance_type: размер VM

elastic_ip: true/false
```
5. Посмотреть план развертывания:
```bash
terraform plan
```
6. Применить конфигурацию:
```bash
terraform apply
```
7. При изменении ресурсов:
```bash
terraform plan
terraform apply
```
8. Команда для удаления окружения:
```bash
terraform destroy
```