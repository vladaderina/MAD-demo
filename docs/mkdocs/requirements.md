#### Минимальные системные требования

```
ОС: Ubuntu server

CPU: 12

MEM: 100Gb

/dev/kvm exists
```

#### Необходимое ПО для системы поиска аномалий

1. [Minikube](minikube.md), или [K3S](k3s.md), или K8S
2. [Helm](others_pp.md)
3. База данных с хранением метрик:
    - Prometheus + Thanos
    - VictoriaMetrics
    - TimescaleDB
4. База данных для хранения моделей:
    - Redis
    - PostgreSQL

#### Для тестового стенда

1. [Terraform](others_pp.md)