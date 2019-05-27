#!/bin/bash

CLUSTER_NAME=randomNAS
DISK_SIZE=500GB
DISK_TYPE=pd-standard
PHYSICAL_BLOCK_SIZE=4096
ZONE=us-east1-b

gcloud beta compute disks create ${CLUSTER_NAME}-disk \
    --physical-block-size ${PHYSICAL_BLOCK_SIZE} \
    --size ${DISK_SIZE} \
    --type ${DISK_TYPE} \
    --zone ${ZONE}


