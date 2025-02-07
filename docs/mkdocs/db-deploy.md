PostgreSQL развернут через [helm-чарт от Bitnami](https://github.com/bitnami/charts/tree/main/bitnami/postgresql)

Для заведения пользователей и баз данных был использован Terraform.

В values добавлен раздел с grafana.ini для обеспечения персистентности с дефолтной базы SQLite, которая находится уже внутри бинарника с Grafana, на созданную базу PostgreSQL ([см. конфигурацию grafana.ini](https://grafana.com/docs/grafana/latest/setup-grafana/configure-grafana/#database)).