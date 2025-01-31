#!/bin/sh
docker pull vladaderina/adservice:latest
docker pull vladaderina/checkoutservice:latest
docker pull vladaderina/currencyservice:latest
docker pull vladaderina/emailservice:latest
docker pull vladaderina/frontend:latest
docker pull vladaderina/loadgenerator:latest
docker pull vladaderina/paymentservice:latest
docker pull vladaderina/productcatalogservice:latest
docker pull vladaderina/recommendationservice:latest
docker pull vladaderina/shippingservice:latest
docker push vladaderina/shoppingassistantservice:latest