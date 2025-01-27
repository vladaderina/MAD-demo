import os
import requests

# Получаем данные из переменных окружения
DOCKER_HUB_USERNAME = os.getenv("DOCKER_HUB_USERNAME")  # Логин Docker Hub
DOCKER_HUB_TOKEN = os.getenv("DOCKER_HUB_TOKEN")  # Токен Docker Hub

# Проверка наличия переменных окружения
if not DOCKER_HUB_USERNAME or not DOCKER_HUB_TOKEN:
    print("Ошибка: Укажите DOCKER_HUB_USERNAME и DOCKER_HUB_TOKEN в файле .env.")
    exit(1)

# API Docker Hub
DOCKER_HUB_API_URL = f"https://hub.docker.com/v2/repositories/{DOCKER_HUB_USERNAME}/"
HEADERS = {
    "Authorization": f"Bearer {DOCKER_HUB_TOKEN}",
    "Content-Type": "application/json"
}

def get_repositories():
    """
    Получает список репозиториев из Docker Hub.
    """
    repositories = []
    url = DOCKER_HUB_API_URL

    while url:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Ошибка при получении репозиториев: {response.status_code} - {response.text}")
            return []

        data = response.json()
        repositories.extend(data["results"])
        url = data.get("next")  # Переход на следующую страницу (если есть)

    return repositories

def main():
    """
    Основная функция: получает список репозиториев и выводит их в формате `название [репозитория](ссылка)`.
    """
    repositories = get_repositories()
    if not repositories:
        print("Репозитории не найдены.")
        return

    for repo in repositories:
        repo_name = repo["name"]
        repo_url = f"https://hub.docker.com/r/{DOCKER_HUB_USERNAME}/{repo_name}"
        print(f"[{repo_name}]({repo_url})\n")

if __name__ == "__main__":
    main()