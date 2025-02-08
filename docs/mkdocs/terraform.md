В проекте Terraform используется для создания базы данных и пользователя Grafana.

### Установка Terraform

```
curl -LOx socks5://192.168.75.190:1080 https://releases.hashicorp.com/terraform/1.8.4/terraform_1.8.4_linux_amd64.zip

unzip terraform_1.8.4_linux_amd64.zip

sudo mv terraform /usr/local/bin/
```
### Провайдер для работы с Postgres

Провойдер был собран из [исходного кода](https://github.com/cyrilgdn/terraform-provider-postgresql).

Бинарный файл был размещен в проекте по пути `/terraform/.terraform.d/plugins/registry.terraform.io/local/postgresql/1.0.0/linux_amd64/terraform-provider-postgresql`.


В файле main.tf было указано его расположение:

```
terraform {
  required_providers {
    postgresql = {
      source  = "local/postgresql"
      version = "1.0.0"
    }
  }
}
```