#!/bin/bash

# Указываем имя пода и образ, который будет использоваться
POD_NAME_PREFIX="my-pod"
IMAGE="nginx"

# Бесконечный цикл для создания подов
while true; do
  # Генерируем уникальное имя для пода
  POD_NAME="${POD_NAME_PREFIX}-$(date +%s%N)"
  
  # Создаем под
  kubectl run $POD_NAME --image=$IMAGE
  
  # Ждем 2 секунды перед созданием следующего пода
  sleep 2
done
