За основу был взят чарт [victoria-metrics-k8s-stack](https://github.com/VictoriaMetrics/helm-charts/tree/master/charts/victoria-metrics-k8s-stack)

### Установка библиотеки с чартом

Для скачивания чарта нужен доступ к публичному DNS серверу. На сервере в файл /etc/resolv.conf были сначала добавлены строки:

```
nameserver 8.8.8.8
nameserver 8.8.4.4
```

И, чтобы весь трафик шел на маршрутизатор, была выполнена команда:

```
sudo ip route add default via 192.168.75.1 dev enp2s0
```

![alt text](image-2.png)

Но так как данные перетирались, то была установлена утилита `resolvconf` для персистентности данных с DNS сервером:

```
apt install resolvconf
```

```
echo "nameserver 8.8.8.8" | sudo resolvconf -a enp2s0
```

А для сохранения настроек маршрута был отредактирован файл:

```
sudo nano /etc/netplan/50-cloud-init.yaml
```

```
network:
  version: 2
  ethernets:
    enp2s0:
      dhcp4: no
      addresses:
        - 192.168.75.11/24
      gateway4: 192.168.75.1
      nameservers:
        addresses:
          - 8.8.8.8
          - 1.1.1.1
```

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
Загружаем зависимости чарта:
```
helm dependency build
```
### Конфигурация чарта

Для проекта MAD в чарте репозитория в файле values.yaml были выставлены 'enabled: false' для следующих values:

- defaultRules

- kube-state-metrics

- Компоненты скрейпинга: kubelet, kubeApiServer, kubeControllerManager, kubeDns, kubeEtcd, kubeScheduler, kubeProxy

А так же добавлен раздел с grafana.ini для обеспечения персистентности с дефолтной базы SQLite, которая находится уже внутри бинарника с Grafana, на базу PostgreSQL ([см. конфигурацию grafana.ini](https://grafana.com/docs/grafana/latest/setup-grafana/configure-grafana/#database)).

Устанавливаем чарт в кластер:

```
helm upgrade monitoring . --install --create-namespace -n monitoring
```
Проверяем доступность Grafana:
```
curl localhost:30300
```
!!! Note
    Дефолтные логин и пароль в Grafana: `admin`