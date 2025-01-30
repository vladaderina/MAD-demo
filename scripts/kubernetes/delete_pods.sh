#!/bin/bash

# Указываем префикс имени пода
POD_NAME_PREFIX="my-pod"

# Получаем список всех подов с указанным префиксом имени
PODS=$(kubectl get pods -o jsonpath="{.items[*].metadata.name}" | tr ' ' '\n' | grep "^${POD_NAME_PREFIX}")

# Удаляем все найденные поды
kubectl delete pods $PODS
