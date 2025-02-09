# Настройка PostgreSQL для Grafana с помощью Terraform

С помощью Terraform в проекте создается база данных PostgreSQL и пользователь для Grafana, а также предоставлются необходимые привилегии и доступы для пользователя Grafana.

## Требования

- Terraform v1.8.4
- PostgreSQL v17.2

## Провайдеры

Этот конфигурационный файл использует провайдер `local/postgresql` для управления ресурсами PostgreSQL. Перед применением конфигурации необходимо убедиться, что провайдер установлен. 

Репозиторий с исходными файлами провайдера: [cyrilgdn/terraform-provider-postgresql](https://github.com/cyrilgdn/terraform-provider-postgresql).

Расположение бинарного файла с провайдером в проекте: `/terraform/.terraform.d/plugins/registry.terraform.io/local/postgresql/1.0.0/linux_amd64/terraform-provider-postgresql`.

## Описание ресурсов

### 1. **PostgreSQL база данных (`grafana`)**

   Создает новую базу данных PostgreSQL с именем `grafana`.

   ```hcl
   resource "postgresql_database" "grafana" {
     name = "grafana"
   }
   ```

### 2. **PostgreSQL роль (`grafana`)**

   Создает нового пользователя PostgreSQL для Grafana с заданным паролем.

   ```hcl
    resource "postgresql_role" "grafana_user" {
    name     = "grafana"
    password = var.grafana_db_password
    login    = true
    superuser = false
    }
```

### 3. **PostgreSQL привилегии по умолчанию для таблиц**

Предоставляет привилегии SELECT, INSERT, UPDATE, DELETE для всех таблиц в схеме public базы данных grafana пользователю grafana_user.

   ```hcl
    resource "postgresql_default_privileges" "grant_all_tables" {
    owner      = "pg_database_owner"
    role       = postgresql_role.grafana_user.name
    database   = postgresql_database.grafana.name
    schema     = "public"
    object_type = "table"
    privileges  = ["SELECT", "INSERT", "UPDATE", "DELETE"]
    }
```
### 4. **PostgreSQL предоставление прав на создание объектов в схеме**

Предоставляет пользователю grafana право создавать объекты (таблицы, представления и т. д.) в схеме public базы данных grafana.

```hcl
    resource "postgresql_grant" "grant_create_on_schema" {
    role = "grafana"
    database  = "grafana"
    schema    = "public"
    object_type = "schema"
    privileges = ["CREATE"]
    }
   ```

## Переменные

 - postgres_host: Хост PostgreSQL.
  
 - postgres_port: Порт для подключения к PostgreSQL (по умолчанию 5432).
  
 - postgres_user: Имя пользователя для подключения к PostgreSQL.
  
 - postgres_password: Пароль пользователя PostgreSQL.

 - grafana_db_password: Пароль для пользователя Grafana в базе данных grafana.