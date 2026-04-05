#!/bin/bash
mkdir -p data/
curl -L -o amazon_products.zip \
  https://www.kaggle.com/api/v1/datasets/download/asaniczka/amazon-products-dataset-2023-1-4m-products
unzip amazon_products.zip -d data/
rm amazon_products.zip