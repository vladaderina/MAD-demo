variable "postgres_port" {
  description = "Port PostgreSQL"
  type        = number
  default     = 30000
}

variable "postgres_host" {
  description = "Host PostgreSQL"
  type        = string
  default     = "192.168.39.24"
}