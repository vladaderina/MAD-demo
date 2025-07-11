В репозитории на GitHub хранится [исходный код сервисов](https://github.com/vladaderina/MAD/tree/main/src).

В директории с каждым сервисом хранятся Dockerfile, на основе которых собираются образы контейнеров.

Также в репозитории хранится [Helm Chart](https://github.com/vladaderina/MAD/tree/main/helm) с шаблонами.

С помощью механизмов GitHub Actions стоят задачи:

 1. Автоматизировать сборку контейнеров и доставку их до Docker Hub.

 2. Автоматизировать обновление релиза helm на сервере.

#### 1 шаг. Настройка Docker Hub

Создаем токен доступа, который будет использоваться GitHub Actions и во вспомогательных скриптах.

Для каждого сервиса был создан репозиторий на [Docker Hub](https://hub.docker.com/) с использованием скрипта [`create-repo.py`](https://github.com/vladaderina/MAD/blob/main/scripts/docker-hub/create-repo.py).

Ссылки на созданные репозитории (через скрипт [`get-link.py`](https://github.com/vladaderina/MAD/blob/main/scripts/docker-hub/get-link.py)): 

[adservice](https://hub.docker.com/r/vladaderina/adservice)

[cartservice](https://hub.docker.com/r/vladaderina/cartservice)

[emailservice](https://hub.docker.com/r/vladaderina/emailservice)

[currencyservice](https://hub.docker.com/r/vladaderina/currencyservice)

[loadgenerator](https://hub.docker.com/r/vladaderina/loadgenerator)

[shippingservice](https://hub.docker.com/r/vladaderina/shippingservice)

[frontend](https://hub.docker.com/r/vladaderina/frontend)

[paymentservice](https://hub.docker.com/r/vladaderina/paymentservice)

[shoppingassistantservice](https://hub.docker.com/r/vladaderina/shoppingassistantservice)

[checkoutservice](https://hub.docker.com/r/vladaderina/checkoutservice)

[recommendationservice](https://hub.docker.com/r/vladaderina/recommendationservice)

[productcatalogservice](https://hub.docker.com/r/vladaderina/productcatalogservice)

#### 2 шаг. Настройка self-hosted GitHub runner

В GitHub был получен токен репозитория

На сервер 192.168.75.11 были установлены:

 - GitHub self-host runner ([офиц. документация](https://github.com/vladaderina/MAD/settings/actions/runners/new?arch=x64&os=linux))

 - Docker Engine ([офиц. документация](https://docs.docker.com/engine/install/ubuntu/))

 - Утилита `yq` с использованием snap:

    ```bash
    sudo apt install snapd
    sudo snap install yq
    ```

Был создан шаблон с ServiceAccount, Role и RoleBinding для github-runner, после чего обновлен релиз:
```
helm upgrade my-microservices .
```

Создан токен аккаунта на 1 год:

```
kubectl -n default create token github-runner --duration=8760h
```

Этот токен нужно указать в [файле конфигурации](https://github.com/vladaderina/MAD/tree/main/meta/kubeconfig-github) для kubectl.


#### 3 шаг. Написание Workflow

[Текст Workflow](https://github.com/vladaderina/MAD/blob/main/.github/workflows/ci-cd.yaml)

Пайплайн запускается после пуша изменения в репозиторий и влючает в себя стадии:

- Клонирование репозитория

- Определение списка папок, в которых произошли изменения

- Сборка новых образов для сервисов, в папках которых произошли изменения, с тегом, равным хешу коммита

- Перезапись тегов по необходимости в values.yaml

- Установка нового релиза helm