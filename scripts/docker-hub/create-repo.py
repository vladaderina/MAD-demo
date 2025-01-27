import os
import requests

DOCKER_HUB_USERNAME = os.getenv("DOCKER_HUB_USERNAME")  # Логин на Docker Hub
DOCKER_HUB_TOKEN = os.getenv("DOCKER_HUB_TOKEN")   # Токен доступа Docker Hub
SERVICES_DIR = "src"  # Директория с сервисами

# API Docker Hub
DOCKER_HUB_API_URL = "https://hub.docker.com/v2/repositories/"
HEADERS = {
    "Authorization": f"Bearer {DOCKER_HUB_TOKEN}",
    "Content-Type": "application/json"
}

def create_dockerhub_repository(repo_name):
    """
    Создает репозиторий в Docker Hub.
    """
    url = f"{DOCKER_HUB_API_URL}"
    data = {
        "namespace": DOCKER_HUB_USERNAME,
        "name": repo_name,
        "is_private": False  # Сделать репозиторий публичным
    }
    response = requests.post(url, json=data, headers=HEADERS)
    
    if response.status_code == 201:
        print(f"Репозиторий '{repo_name}' успешно создан.")
    else:
        print(f"Ошибка при создании репозитория '{repo_name}': {response.status_code} - {response.text}")

def main():
    """
    Основная функция: сканирует директорию и создает репозитории.
    """
    if not os.path.exists(SERVICES_DIR):
        print(f"Директория '{SERVICES_DIR}' не найдена.")
        return

    # Получаем список папок в директории
    services = [name for name in os.listdir(SERVICES_DIR) if os.path.isdir(os.path.join(SERVICES_DIR, name))]

    if not services:
        print(f"В директории '{SERVICES_DIR}' нет поддиректорий.")
        return

    # Создаем репозитории для каждой папки
    for service in services:
        create_dockerhub_repository(service)

if __name__ == "__main__":
    main()