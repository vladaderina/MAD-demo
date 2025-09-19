variable "postgres_host" {
  description = "PostgreSQL host"
  type        = string
}

variable "postgres_port" {
  description = "PostgreSQL port"
  type        = number
}

variable "postgres_user" {
  description = "PostgreSQL admin username"
  type        = string
}

variable "postgres_password" {
  description = "PostgreSQL admin password"
  type        = string
  sensitive   = true
}

variable "grafana_db_password" {
  description = "Password for Grafana DB user"
  type        = string
  sensitive   = true
}

variable "mad_db_password" {
  description = "Password for MAD database user"
  type        = string
  sensitive   = true
}
