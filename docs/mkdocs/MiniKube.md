## Тестовая среда MiniKube для сбора метрик

На этой странице приведена инструкция по развертыванию и конфигурированию кластера для тестового сбора метрик с компонент:
- kube-api
- controller-manager
- scheduler
- etcd

1 шаг. Установка minikube
[Официальная документация](https://kubernetes.io/ru/docs/tasks/tools/install-minikube/)

2 шаг. Создание системного сервиса для minikube

# Файл конфигурации
sudo nano /etc/systemd/system/minikube.service

[Unit]
Description=Minikube Kubernetes Cluster
After=network.target

[Service]
Type=simple
User=vderina
ExecStart=/usr/local/bin/minikube start --extra-config=scheduler.bind-address=192.168.49.2 --extra-config=controller-manager.bind-address=192.168.49.2
ExecStop=/usr/local/bin/minikube stop
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target

# Перезагрузите системный демон
sudo systemctl daemon-reload

# Включите сервис
sudo systemctl enable minikube.service

# Запустите сервис
sudo systemctl start minikube.service

# Проверьте состояние сервиса
sudo systemctl status minikube.service

3 шаг. Установка и конфигурирование Victoria Metrics

helm repo add victoria-metrics https://victoriametrics.github.io/helm-charts/
helm repo update
helm install victoria-metrics victoria-metrics/victoria-metrics-cluster

export POD_NAME=$(kubectl get pods --namespace default -l "app=vmselect" -o jsonpath="{.items[0].metadata.name}")
kubectl --namespace default port-forward $POD_NAME 8481 &>/dev/null &

Доступ к UI: http://localhost:8481/select/0/vmui/

4 шаг. Установка и конфигурирование Grafana

helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
helm install grafana grafana/grafana

export POD_NAME=$(kubectl get pods --namespace default -l "app.kubernetes.io/name=grafana,app.kubernetes.io/instance=grafana" -o jsonpath="{.items[0].metadata.name}")
kubectl --namespace default port-forward $POD_NAME 3000 &>/dev/null &

Доступ к UI: http://localhost:3000/select/0/vmui/

Логин: admin
Пароль хранится внутри кластера:
kubectl get secret --namespace default grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo

5 шаг. Установка и конфигурирование Prometheus

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/prometheus

export POD_NAME=$(kubectl get pods --namespace default -l "app.kubernetes.io/name=prometheus,app.kubernetes.io/instance=prometheus" -o jsonpath="{.items[0].metadata.name}")
kubectl --namespace default port-forward $POD_NAME 9090 &>/dev/null &

Доступ к UI: http://localhost:9090/targets


В среде WSL в resolv.conf среди DNS серверов не было 8.8.8.8 из-за чего не было доступа до домена grafana.github.io
Решение: 
sudo dnf install resolvconf
wsl --shutdown

helm upgrade prometheus prometheus-community/prometheus


lsof -ti :9090 | xargs kill

helm repo update

edit configmap prometheus-server

- bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    job_name: kubernetes-apiservers
    scrape_interval: 10s
    scheme: https
    metrics_path: /metrics
    static_configs:
    - targets:
        - 192.168.49.2:8443
    tls_config:
    ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    insecure_skip_verify: true


kubectl rollout restart deployment prometheus-server

https://github.com/prometheus-community/helm-charts/issues/204#issuecomment-765155883

minikube ssh "cat /var/lib/minikube/certs/etcd/ca.crt" > ca.crt
minikube ssh "cat /var/lib/minikube/certs/etcd/healthcheck-client.crt" > healthcheck-client.crt
minikube ssh "cat /var/lib/minikube/certs/etcd/healthcheck-client.key" > healthcheck-client.key

kubectl create secret generic etcd-client-cert \
        --from-file=ca.crt \
        --from-file=healthcheck-client.crt \
        --from-file=healthcheck-client.key