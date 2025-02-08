variable "postgres_port" {
  description = "Порт подключения к PostgreSQL"
  type        = number
  default     = 30000  # Укажите порт по умолчанию, если нужно
}

variable "postgres_host" {
  description = "Хост PostgreSQL"
  type        = string
  default     = "192.168.39.24"
}
