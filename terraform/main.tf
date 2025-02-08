terraform {
  required_providers {
    postgresql = {
      source  = "local/postgresql"
      version = "1.0.0"
    }
  }
}

provider "postgresql" {
  host     = "192.168.39.24"
  port     = 30000                        
  username = "postgres"
  password = "postgres"
  database = "postgres"
  sslmode  = "disable"
}

resource "postgresql_database" "grafana" {
  name = "grafana"
}

resource "postgresql_role" "grafana_user" {
  name     = "grafana"
  password = "grafana"
  login    = true
  superuser = false
}

resource "postgresql_grant" "grafana_grant" {
  database    = postgresql_database.grafana.name
  role        = postgresql_role.grafana_user.name
  object_type = "database"
  privileges  = ["ALL"]
}