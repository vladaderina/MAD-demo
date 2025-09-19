#### Установка Helm

[Официальная документация по установке Helm](https://helm.sh/docs/)

```bash
# Скачивание скрипта установки
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3

# Даем права на выполнение
chmod 700 get_helm.sh

# Запускаем установку
./get_helm.sh

# Проверяем установку
helm version
```
#### Установка Terraform

[Официальная документация по установке Terraform](https://developer.hashicorp.com/terraform/install)

```bash
# Скачивание Terraform через прокси
curl -LOx socks5://192.168.75.190:1080 https://releases.hashicorp.com/terraform/1.8.4/terraform_1.8.4_linux_amd64.zip

# Распаковка архива
unzip terraform_1.8.4_linux_amd64.zip

# Перемещение в директорию исполняемых файлов
sudo mv terraform /usr/local/bin/

# Проверяем установку
terraform version
```
#### Установка K3S

[Официальная документация по установке K3S]()

```bash
# Установка K3S
curl -sfL https://get.k3s.io | sh -

# Проверка статуса службы
sudo systemctl status k3s

# Получение конфигурации для kubectl
sudo cat /etc/rancher/k3s/k3s.yaml

# Проверка работы кластера
sudo k3s kubectl get nodes
```
#### Установка Ansible

[Официальная документация по установке Ansible](https://docs.ansible.com/ansible/latest/installation_guide/index.html)

```bash
# Обновление пакетов
sudo apt update

# Установка Ansible
sudo apt install -y ansible

# Проверка установки
ansible --version

# Проверка доступных хостов
ansible all -m ping -i localhost,
```