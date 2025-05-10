## Установка Isio Operator

1. Установка IstioCtl

На сервере установим IstioCtl:

```
curl -L https://istio.io/downloadIstio | sh -
cd istio-*
sudo mv ./bin/istioctl /usr/bin/
```

Проверим версию:

![alt text](images/image-3-1.png)

2. Создание манифеста IstioOperator

Создаем файл `istio-operator.yaml`:

```
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: istio-control-plane
  namespace: istio-system
spec:
  profile: minimal
  meshConfig:
    defaultConfig:
      proxyMetadata:
        ISTIO_META_DNS_CAPTURE: "true"
        ISTIO_META_DNS_AUTO_ALLOCATE: "true"
  values:
    telemetry:
      enabled: true
```

3. Установка Istio в кластер

```
istioctl install -f istio-operator.yaml --set profile=minimal -y
```

!!! note "Пояснение к команде istioctl install"
    1. Определяет текущий контекст `kubectl config current-context`
    2. Проверяет совместимость версии Istio с кластером
    3. Проверяет, установлен ли уже Istio (kubectl get namespace istio-system)
    4. Если CRD Istio еще нет в кластере, istioctl install создает их: `kubectl apply -f https://github.com/istio/istio/releases/latest/download/samples/operator/crd.yaml`
    5. Если его istio-system namespace нет - создает его
    6. На основе профиля (default) и манифеста istio-operator.yaml разворачивает компоненты в istio-system


!!! note "Как istioctl понимает, что устанавливать?"

    istioctl install использует комбинацию параметров:

    1. Профиль установки (--set profile=default):

    Определяет, какие компоненты Istio включить (например, Ingress, Egress, Telemetry).
    Можно задать minimal, default, demo или кастомный профиль. [Официальная документация](https://istio.io/latest/docs/setup/additional-setup/config-profiles/). 

    2. Файл конфигурации (-f istio-operator.yaml):

    Позволяет переопределить стандартные параметры установки.
    Включает такие настройки, как proxyMetadata, telemetry.enabled, meshConfig.

4. Проверка установки Istio

```
kubectl get crds | grep istio
```

![alt text](image-5.png)

```
kubectl get pods -n istio-system
```

![alt text](image-6.png)

```
kubectl get svc -n istio-system
```

![alt text](image-7.png)

## Настройка сбора метрик с Isio

1. Включение auto-injection

Auto-injection включается аннотацией istio-injection=enabled в нужном namespace:

```
kubectl label namespace product istio-injection=enabled
```

Теперь при деплое новых подов сайдкар будет автоматически добавляться. 

2. Обновление существующих подов

```
kubectl rollout restart deployment -n product
```

После обновления у всех подов появились sidecar контейнеры istio:

![alt text](image-8.png)

3. Проверка, что Istio отдает метрики

```
kubectl port-forward -n istio-system svc/istiod 15014:15014
curl http://localhost:15014/metrics
```

![alt text](image-9.png)

## Настройка scraping для VictoriaMetrics

Добавим конфигурацию для VMServiceScrape:

```
apiVersion: operator.victoriametrics.com/v1beta1
kind: VMScrapeConfig
metadata:
  name: istio-metrics
  namespace: monitoring
spec:
  kubernetesSDConfigs:
  - role: endpoints
    namespaces:
      names:
        - istio-system
  relabelConfigs:
    - source_labels: [__meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name, __meta_kubernetes_namespace]
      action: keep
      regex: istiod;http-monitoring;istio-system
---
apiVersion: operator.victoriametrics.com/v1beta1
kind: VMScrapeConfig
metadata:
  name: istio-sidecar-metrics
  namespace: monitoring
spec:
  path: /stats/prometheus
  kubernetesSDConfigs:
  - role: pod
    namespaces:
      names:
        - product
  relabelConfigs:
    - source_labels: [__meta_kubernetes_pod_container_port_name]
      action: keep
      regex: '.*-envoy-prom'
```

Теперь у VM появились таргеты:

![alt text](image-10.png)

!!! note "Полезные команды"
  1. Проверка DNS-резолвинга из другого namespace:
  kubectl run -n monitoring -it --rm --image=busybox dns-test -- nslookup istiod.istio-system.svc.cluster.local
  2. Проверка доступности через curl из другого namespace:
  kubectl run -n monitoring -it --rm --image=curlimages/curl curl-test -- curl http://istiod.istio-system.svc.cluster.local:15014/metrics

## Импорт Grafana dashboards

Список использованных шаблонов с [Grafana Labs](https://grafana.com/orgs/istio/dashboards):
- Istio Mesh Dashboard by istio
- Istio Workload Dashboard by istio
