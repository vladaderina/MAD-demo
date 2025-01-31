#!/bin/sh

docker tag vladaderina/adservice:latest vladaderina/adService:latest
docker rmi vladaderina/adservice:latest
docker push vladaderina/adService:latest

docker tag vladaderina/cartservice:latest vladaderina/cartService:latest
docker rmi vladaderina/cartservice:latest
docker push vladaderina/cartService:latest

docker tag vladaderina/checkoutservice:latest vladaderina/checkoutService:latest
docker rmi vladaderina/checkoutservice:latest
docker push vladaderina/checkoutservice:latest

docker tag vladaderina/currencyservice:latest vladaderina/currencyService:latest 
docker rmi vladaderina/currencyservice:latest
docker push vladaderina/currencyService:latest 

docker tag vladaderina/emailservice:latest vladaderina/emailService:latest
docker rmi vladaderina/emailservice:latest
docker push vladaderina/emailService:latest

docker tag vladaderina/frontend:latest vladaderina/frontend:latest
docker rmi vladaderina/frontend:latest
docker push vladaderina/frontend:latest

docker tag vladaderina/loadgenerator:latest vladaderina/loadGenerator:latest
docker rmi vladaderina/loadgenerator:latest
docker push vladaderina/loadGenerator:latest

docker tag vladaderina/paymentservice:latest vladaderina/paymentService:latest
docker rmi vladaderina/paymentservice:latest
docker push vladaderina/paymentService:latest

docker tag vladaderina/productcatalogservice:latest vladaderina/productCatalogService:latest
docker rmi vladaderina/productcatalogservice:latest
docker push vladaderina/productCatalogService:latest

docker tag vladaderina/recommendationservice:latest vladaderina/recommendationService:latest
docker rmi vladaderina/recommendationservice:latest
docker push vladaderina/recommendationService:latest

docker tag vladaderina/shippingservice:latest vladaderina/shippingService:latest
docker rmi vladaderina/shippingservice:latest
docker push vladaderina/shippingService:latest

docker tag vladaderina/shoppingassistantservice:latest vladaderina/shoppingAssistantService:latest
docker rmi vladaderina/shoppingassistantservice:latest
docker push vladaderina/shoppingAssistantService:latest