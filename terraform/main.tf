terraform {
  required_providers {
    postgresql = {
      source  = "cyrilgdn/postgresql"
#      version = "~> 1.7.1"
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

resource "postgresql_database" "ml_models" {
  name = "ml_models"
}

resource "postgresql_schema" "models_schema" {
  name     = "modeling"
  database = postgresql_database.ml_models.name
}

resource "postgresql_extension" "uuid" {
  name     = "uuid-ossp"
  database = postgresql_database.ml_models.name
}

resource "postgresql_table" "models_table" {
  name     = "models"
  schema   = postgresql_schema.models_schema.name
  database = postgresql_database.ml_models.name

  owner = "mad"

  depends_on = [postgresql_schema.models_schema]

  columns = [
    {
      name = "id"
      type = "uuid"
      default = "uuid_generate_v4()"
      null = false
    },
    {
      name = "name"
      type = "text"
      null = false
    },
    {
      name = "labels"
      type = "jsonb"
      null = false
    },
    {
      name = "config"
      type = "jsonb"
      null = false
    },
    {
      name = "history"
      type = "jsonb"
      null = false
    },
    {
      name = "created_at"
      type = "timestamp"
      default = "now()"
      null = false
    }
  ]

  primary_key = ["id"]
}