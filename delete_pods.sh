#!/bin/bash

# Указываем префикс имени пода
POD_NAME_PREFIX="my-pod"

# Получаем список всех подов с указанным префиксом имени
PODS=$(kubectl get pods -o jsonpath="{.items[*].metadata.name}" | tr ' ' '\n' | grep "^${POD_NAME_PREFIX}")

# Удаляем все найденные поды
for POD in $PODS; do
  kubectl delete pod $POD
done
