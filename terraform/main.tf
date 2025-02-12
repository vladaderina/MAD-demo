terraform {
  required_providers {
    postgresql = {
      source  = "local/postgresql"
      version = "1.0.0"
    }
  }
}

provider "postgresql" {
  host     = var.postgres_host
  port     = var.postgres_port                        
  username = var.postgres_user
  password = var.postgres_password
  database = "postgres"
  sslmode  = "disable"
}

resource "postgresql_database" "grafana" {
  name = "grafana"
}

resource "postgresql_role" "grafana_user" {
  name     = "grafana"
  password = var.grafana_db_password
  login    = true
  superuser = false
}

resource "postgresql_grant" "grafana_grant" {
  database    = postgresql_database.grafana.name
  role        = postgresql_role.grafana_user.name
  object_type = "database"
  privileges  = ["ALL"]
}

resource "postgresql_default_privileges" "grant_all_tables" {
  owner      = "pg_database_owner"
  role       = postgresql_role.grafana_user.name
  database   = postgresql_database.grafana.name
  schema     = "public"
  object_type = "table"
  privileges  = ["SELECT", "INSERT", "UPDATE", "DELETE"]
}

resource "postgresql_grant" "grant_create_on_schema" {
  role = "grafana"
  database  = "grafana"
  schema    = "public"
  object_type = "schema"
  privileges = ["CREATE"]

  depends_on = [postgresql_database.grafana]
}