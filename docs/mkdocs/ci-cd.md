В репозитории на GitHub хранится [исходный код сервисов](https://github.com/vladaderina/KubeAnomaly/tree/main/src).

В директории с каждым сервисом хранятся Dockerfile, на основе которых собираются образы контейнеров.

Также в репозитории хранится [Helm Chart](https://github.com/vladaderina/KubeAnomaly/tree/main/helm) с шаблонами.

С помощью механизмов GitHub Actions стоят задачи:

 1. Автоматизировать сборку контейнеров и доставку их до Docker Hub.

 2. Автоматизировать обновление релиза helm на сервере.

#### 1 шаг. Настройка Docker Hub

Создаем токен доступа, который будет использоваться GitHub Actions и во вспомогательных скриптах.

Для каждого сервиса был создан репозиторий на [Docker Hub](https://hub.docker.com/) с использованием скрипта `create-repo.py`.

Ссылки на созданные репозитории (через скрипт `get-link.py`): 

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

#### 2 шаг. Настройка GitHub Actions

#### 3 шаг. Написание Workflow