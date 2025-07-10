База данных в проекте нужна для:

 - хранения конфигурации Grafana (дашборды, пользователи, настройки)

 - хранения построенных LSTM моделей и истории аномалий

Для развертывания PostgreSQL был использован [helm-чарт от Bitnami](https://github.com/bitnami/charts/tree/main/bitnami/postgresql).

Дополнительно был добавлен сервис типа NodePort для дальнейшей конфигурации PostgreSQL через Terraform.

```bash
git clone https://github.com/vladaderina/KubeAnomaly.git

cd ./helm/storage

helm dependency build

helm upgrade storage . --install --create-namespace -n storage

```
Terraform был использован для заведения пользователя и базы данных.

### Провайдер для работы с Postgres

Провойдер был загружен из [облака](https://tf.org.ru/).

В домашнем каталоге пользователя был создан .terraformrc с содержанием:

```
provider_installation {
  network_mirror {
    url     = "https://nm.tf.org.ru/"
    include = ["registry.terraform.io/*/*"]
  }
  direct {
    exclude = ["registry.terraform.io/*/*"]
  }
}
```

### Запуск Terraform

Перед запуском нужно установить переменные окружения

```bash
export TF_VAR_postgres_password=$(kubectl get secret storage-postgresql -n storage -o jsonpath="{.data.postgres-password}" | base64 --decode)
export TF_VAR_grafana_db_password="grafana"
```

Далее выполнить команды

```
terraform init
terraform plan
terraform apply
```