#!/bin/bash -ex
#
# Get the Stanford Online Product dataset
#

DATA_DIR=/media/brian/00686ed3-5895-442b-a66e-15fbe9951a91/stanford_products

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist, will create one.";
  mkdir -p ${DATA_DIR}
fi

wget -P ${DATA_DIR} ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
cd ${DATA_DIR}; unzip Stanford_Online_Products.zip