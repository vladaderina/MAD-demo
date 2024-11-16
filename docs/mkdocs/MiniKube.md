## Тестовая среда MiniKube для сбора метрик

На этой странице приведена инструкция по развертыванию и конфигурированию кластера для тестового сбора метрик с компонент:

- kube-api
- controller-manager
- scheduler
- etcd

#### 1 шаг. Установка minikube

[Официальная документация](https://kubernetes.io/ru/docs/tasks/tools/install-minikube/)

#### 2 шаг. Создание системного сервиса для minikube

Создаем файл конфигурации
```
sudo nano /etc/systemd/system/minikube.service
```
со следующим содержанием:
```bash
[Unit]
Description=Minikube Kubernetes Cluster
After=network.target docker.service
Requires=docker.service

[Service]
User=vderina
ExecStart=/usr/local/bin/minikube start --extra-config=scheduler.bind-address=192.168.49.2 --extra-config=controller-manager.bind-address=192.168.49.2
ExecStop=/usr/local/bin/minikube stop
Type=simple
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```
Перезагружаем, включаем и запускаем системный демон:

```bash
systemctl daemon-reload && systemctl start minikube && systemctl enable minikube
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


#### 3 шаг. Установка и конфигурирование Victoria Metrics

Установка кластерной версии Victoria Metrics с помощью менеджера пакетов [Helm](https://helm.sh/ru/docs/intro/install/):

```bash
helm repo add victoria-metrics https://victoriametrics.github.io/helm-charts/ \
     repo update \
     install victoria-metrics victoria-metrics/victoria-metrics-cluster
```

Настроим проброс портов для доступа к UI:
```bash
export POD_NAME=$(kubectl get pods --namespace default -l "app=vmselect" -o jsonpath="{.items[0].metadata.name}")
kubectl --namespace default port-forward $POD_NAME 8481 &>/dev/null &
```
**Адрес UI:** http://localhost:8481/select/0/vmui/

#### 4 шаг. Установка и конфигурирование Grafana
Создаем файл конфигурации
```bash
sudo nano ./grafana-values.yaml
```
со следующим содержанием:
```yaml
datasources:
  datasources.yaml:
    apiVersion: 1
    datasources:
      - name: VictoriaMetrics
        type: prometheus
        url: http://victoria-metrics-victoria-metrics-cluster-vmselect.default.svc.cluster.local.:8481/select/0/prometheus/
        access: proxy
        isDefault: true
```
Установка Grafana с помощью менеджера пакетов [Helm](https://helm.sh/ru/docs/intro/install/):
```bash
helm repo add grafana https://grafana.github.io/helm-charts \
     repo update \
     install grafana grafana/grafana -f grafana-values.yaml
```

Настроим проброс портов для доступа к UI:
```bash
export POD_NAME=$(kubectl get pods --namespace default -l "app.kubernetes.io/name=grafana,app.kubernetes.io/instance=grafana" -o jsonpath="{.items[0].metadata.name}")
kubectl --namespace default port-forward $POD_NAME 3000 &>/dev/null &
```
**Адрес UI:** http://localhost:3000

**Логин:** admin

**Пароль** (хранится внутри кластера):
```bash 
kubectl get secret --namespace default grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
```

#### 5 шаг. Установка и конфигурирование Prometheus

Чтобы [Prometheus смог обращаться к etcd](
https://github.com/prometheus-community/helm-charts/issues/204#issuecomment-765155883), загрузим в кластер сертификаты и ключ:
```bash
minikube ssh "cat /var/lib/minikube/certs/etcd/ca.crt" > ca.crt
minikube ssh "cat /var/lib/minikube/certs/etcd/healthcheck-client.crt" > healthcheck-client.crt
minikube ssh "cat /var/lib/minikube/certs/etcd/healthcheck-client.key" > healthcheck-client.key
```

```bash
kubectl create secret generic etcd-client-cert \
        --from-file=ca.crt \
        --from-file=healthcheck-client.crt \
        --from-file=healthcheck-client.key
```
Смонтируем созданный секрет в под prometheus-server:
```yaml
apiVersion: apps/v1
kind: Deployment
  name: prometheus-server
  ...
  template:
    spec:
      containers:
        - volumeMounts:
            - mountPath: /etc/prometheus/secrets/etcd-client-cert
            name: etcd-secret
      volumes:
        - name: etcd-secret
            secret:
            defaultMode: 420
            secretName: etcd-client-cert
```
Создаем файл конфигурации
```bash
sudo nano ./prometheus.yaml
```
со следующим содержанием:
```yaml
server:
  remoteWrite:
    - url: "http://victoria-metrics-victoria-metrics-cluster-vminsert.default.svc.cluster.local.:8480/insert/0/prometheus/"
scrape_configs:
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
  - bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    job_name: kubernetes-etcd
    scrape_interval: 10s
    scheme: https
    metrics_path: /metrics
    static_configs:
      - targets:
        - 192.168.49.2:2379
    tls_config:
    ca_file: /etc/prometheus/secrets/etcd-client-cert/ca.crt
    cert_file: /etc/prometheus/secrets/etcd-client-cert/healthcheck-client.crt
    key_file: /etc/prometheus/secrets/etcd-client-cert/healthcheck-client.key
    insecure_skip_verify: true
  - bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    job_name: kubernetes-scheduler
    scrape_interval: 10s
    scheme: https
    metrics_path: /metrics
    static_configs:
      - targets:
        - 192.168.49.2:10259
    tls_config:
    ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    insecure_skip_verify: true
  - bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    job_name: kubernetes-controller
    scrape_interval: 10s
    scheme: https
    metrics_path: /metrics
    static_configs:
      - targets:
        - 192.168.49.2:10257
    tls_config:
    ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    insecure_skip_verify: true
```

Установка Prometheus с помощью менеджера пакетов [Helm](https://helm.sh/ru/docs/intro/install/):
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
     repo update
     install prometheus prometheus-community/prometheus -f prometheus.yaml
```

Настроим проброс портов для доступа к UI:
```bash
export POD_NAME=$(kubectl get pods --namespace default -l "app.kubernetes.io/name=prometheus,app.kubernetes.io/instance=prometheus" -o jsonpath="{.items[0].metadata.name}")
kubectl --namespace default port-forward $POD_NAME 9090 &>/dev/null &
```

**Адрес UI:** http://localhost:9090/targets

### Полезные команды
```bash
lsof -ti :9090 | xargs kill
```
```bash
kubectl rollout restart deployment prometheus-server
```