За основу был взят чарт [victoria-metrics-k8s-stack](https://github.com/VictoriaMetrics/helm-charts/tree/master/charts/victoria-metrics-k8s-stack)

### Установка библиотеки с чартом

Для скачивания чарта нужен доступ к публичному DNS серверу. На сервере в файл /etc/resolv.conf были добавлены строки:

```
nameserver 8.8.8.8
nameserver 8.8.4.4
```

И, чтобы весь трафик шел на маршрутизатор, была выполнена команда:

```
sudo ip route add default via 192.168.75.1 dev enp2s0
```
![alt text](image-2.png)

Команда по добавлению библиотеки с чартом:

```
helm repo add vm https://victoriametrics.github.io/helm-charts/
helm repo update
```

Список всех чартов в бибилитеке:

```
helm search repo vm/
```

Команда для импорта values из чарта:

```
helm show values vm/victoria-metrics-cluster > values.yaml
```

### Конфигурация чарта

Для проекта KubeAnomaly в чарте репозитория были выставлены 'enabled: false' для следующих values:

- defaultRules

- kube-state-metrics

- Компоненты скрейпинга: kubelet, kubeApiServer, kubeControllerManager, kubeDns, kubeEtcd, kubeScheduler, kubeProxy

Устанавливаем чарт в кластер:

```
helm dependency build
```

```
helm install monitoring .
```


