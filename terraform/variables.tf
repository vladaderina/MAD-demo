variable "project_name" {
  description = "Название проекта"
  type        = string
  default     = "mad-single"
}

variable "yc_folder_id" {
  description = "Yandex Cloud Folder ID"
  type        = string
}

variable "yc_cloud_id" {
  description = "Yandex Cloud ID"
  type        = string
}

variable "yc_zone" {
  description = "Зона доступности"
  type        = string
  default     = "ru-central1-a"
}

variable "yc_ssh_key_path" {
  description = "Путь до публичного SSH-ключа"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "yc_image_id" {
  description = "ID образа (например Ubuntu 22.04)"
  type        = string
  default     = "fd8n7sjqk8k2********" # замените на актуальный ID
}

variable "vm_cores" {
  description = "Количество vCPU"
  type        = number
  default     = 4
}

variable "vm_memory" {
  description = "Память (в ГБ)"
  type        = number
  default     = 8
}

variable "vm_disk_size" {
  description = "Размер диска (в ГБ)"
  type        = number
  default     = 50
}
