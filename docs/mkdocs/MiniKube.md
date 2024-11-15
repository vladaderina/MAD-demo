## Тестовая среда MiniKube для сбора метрик

Запуск minikube:
minikube start --extra-config=scheduler.bind-address=192.168.49.2 --extra-config=controller-manager.bind-address=192.168.49.2

В среде WSL в resolv.conf среди DNS серверов не было 8.8.8.8 из-за чего не было доступа до домена grafana.github.io
Решение: 
sudo dnf install resolvconf
wsl --shutdown

helm repo add grafana https://grafana.github.io/helm-charts
helm repo add victoria-metrics https://victoriametrics.github.io/helm-charts/
helm upgrade prometheus prometheus-community/prometheus
kubectl get secret --namespace default grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
export POD_NAME=$(kubectl get pods --namespace default -l "app.kubernetes.io/name=grafana,app.kubernetes.io/instance=grafana" -o jsonpath="{.items[0].metadata.name}")
kubectl --namespace default port-forward $POD_NAME 3000 &>/dev/null &
export POD_NAME=$(kubectl get pods --namespace default -l "app.kubernetes.io/name=prometheus,app.kubernetes.io/instance=prometheus" -o jsonpath="{.items[0].metadata.name}")
kubectl --namespace default port-forward $POD_NAME 9090
export POD_NAME=$(kubectl get pods --namespace default -l "app=vmselect" -o jsonpath="{.items[0].metadata.name}")
kubectl --namespace default port-forward $POD_NAME 8481 &>/dev/null &

lsof -ti :9090 | xargs kill

helm repo update
helm cache clean