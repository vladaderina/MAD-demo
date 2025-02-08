variable "postgres_host" {
  description = "PostgreSQL host"
  type        = string
  default     = "192.168.39.24"
}

variable "postgres_port" {
  description = "PostgreSQL port"
  type        = number
  default     = 30000
}

variable "postgres_user" {
  description = "PostgreSQL admin username"
  type        = string
  default     = "postgres"
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
