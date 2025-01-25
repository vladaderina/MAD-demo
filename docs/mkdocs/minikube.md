На этой странице приведена инструкция, по которой был развернут minikube

#### 1 шаг. Установка minikube

Свойства системы:
```
ОС: Ubuntu server

CPU: 12

MEM: 100Gb

/dev/kvm exists
```
[Официальная документация по установке minikube](https://kubernetes.io/ru/docs/tasks/tools/install-minikube/)

[Установка и настройка KVM на Ubuntu](https://help.ubuntu.com/community/KVM/Installation)

После того, как minikube и KVM были установлены и настроены, была выполнена команда по запуску minikube:

```
minikube start --driver=kvm2 --cpus=11 --memory=10000
```

Статус созданной виртуальной машины с minikube можно проверить с помощью команды:

```
virsh list --all
```

Получить информацию о виртуальной машине:

```
virsh dominfo minikube
```
Посмотреть IP адрес виртуальной машины с minikube:
```
minikube ip
```
Подключиться к виртуальной машине по ssh:
```
minikube ssh
```

#### 2 шаг. Создание системного сервиса для minikube

Создаем файл конфигурации
```
sudo nano /etc/systemd/system/minikube.service
```
со следующим содержанием:
```bash
[Unit]
Description=Minikube Kubernetes Cluster
After=network.target
Requires=network.target

[Service]
User=vderina
ExecStart=/usr/local/bin/minikube start --driver=kvm2 --extra-config=scheduler.bind-address=<node-ip> --extra-config=controller-manager.bind-address=<node-ip>
ExecStop=/usr/local/bin/minikube stop
Type=simple
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```
Перезагружаем, включаем и запускаем системный демон:

```bash
sudo systemctl daemon-reload && systemctl start minikube && systemctl enable minikube
```
Проверяем состояние сервиса:
```bash
sudo systemctl status minikube.service
```
!!! Note
    В среде WSL в resolv.conf среди DNS серверов не было 8.8.8.8, из-за чего не было доступа до доменов *.github.io
    
    Решение: 
    ```bash
    sudo dnf install resolvconf
    ```
    ```bash
    wsl --shutdown
    ```