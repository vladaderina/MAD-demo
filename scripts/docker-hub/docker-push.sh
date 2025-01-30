#!/bin/sh

docker build -t vladaderina/checkoutservice:latest ./src/checkoutservice/
docker push vladaderina/checkoutservice:latest

docker build -t vladaderina/currencyservice:latest ./src/currencyservice/
docker push vladaderina/currencyservice:latest

docker build -t vladaderina/emailservice:latest ./src/emailservice/
docker push vladaderina/emailservice:latest

docker build -t vladaderina/frontend:latest ./src/frontend/
docker push vladaderina/frontend:latest

docker build -t vladaderina/loadgenerator:latest ./src/loadgenerator/
docker push vladaderina/loadgenerator:latest

docker build -t vladaderina/paymentservice:latest ./src/paymentservice/
docker push vladaderina/paymentservice:latest

docker build -t vladaderina/productcatalogservice:latest ./src/productcatalogservice/
docker push vladaderina/productcatalogservice:latest

docker build -t vladaderina/recommendationservice:latest ./src/recommendationservice/
docker push vladaderina/recommendationservice:latest

docker build -t vladaderina/shippingservice:latest ./src/shippingservice/
docker push vladaderina/shippingservice:latest

docker build -t vladaderina/shoppingassistantservice:latest ./src/shoppingassistantservice/
docker push vladaderina/shoppingassistantservice:latest