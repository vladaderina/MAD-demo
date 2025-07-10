#### Установка Helm

[Официальная документация по установке k3s](https://helm.sh/docs/)

```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
```
#### Установка Terraform

[Официальная документация по установке Terraform](https://developer.hashicorp.com/terraform/install)

```
curl -LOx socks5://192.168.75.190:1080 https://releases.hashicorp.com/terraform/1.8.4/terraform_1.8.4_linux_amd64.zip

unzip terraform_1.8.4_linux_amd64.zip

sudo mv terraform /usr/local/bin/
```