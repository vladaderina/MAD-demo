PostgreSQL развернут через [helm-чарт от Bitnami](https://github.com/bitnami/charts/tree/main/bitnami/postgresql)

Для заведения пользователя и базы данных был использован Terraform. Подробнее на [странице о Terraform](terraform.md).

В values добавлен раздел с grafana.ini для обеспечения персистентности с дефолтной базы SQLite, которая находится уже внутри бинарника с Grafana, на созданную базу PostgreSQL ([см. конфигурацию grafana.ini](https://grafana.com/docs/grafana/latest/setup-grafana/configure-grafana/#database)).