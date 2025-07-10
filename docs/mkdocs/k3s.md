На этой странице приведена инструкция, по которой был развернут k3s

#### 1 шаг. Установка minikube

[Официальная документация по установке k3s](https://docs.k3s.io/installation/configuration)

```bash
curl -Lo /usr/local/bin/k3s https://github.com/k3s-io/k3s/releases/download/v1.26.5+k3s1/k3s; chmod a+x /usr/local/bin/k3s
```

#### 2 шаг. Создание системного сервиса для k3s

Создаем файл конфигурации
```
sudo nano /etc/systemd/system/k3s.service
```
со следующим содержанием:
```bash
[Unit]
Description=Lightweight Kubernetes
Documentation=https://k3s.io
After=network.target

[Service]
Type=notify
Environment="K3S_KUBECONFIG_MODE=644"
ExecStart=/usr/local/bin/k3s server
Restart=always

[Install]
WantedBy=multi-user.target

```
Перезагружаем и запускаем системный демон:

```bash
sudo systemctl daemon-reload && systemctl enable --now k3s
```
Проверяем состояние сервиса:
```bash
sudo sudo systemctl status k3s
```