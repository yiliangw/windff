#!/bin/bash

PROJECT_DIR=$(cd $(dirname $0); cd ..; pwd)
INFLUXDB_CACHE_DIR=${PROJECT_DIR}/.influxdb

docker run \
 --name influxdb2 \
 --publish 8086:8086 \
 --mount type=bind,source=${INFLUXDB_CACHE_DIR}/data,target=/var/lib/influxdb2 \
 --mount type=bind,source=${INFLUXDB_CACHE_DIR}/config,target=/etc/influxdb2 \
 --env DOCKER_INFLUXDB_INIT_MODE=setup \
 --env DOCKER_INFLUXDB_INIT_USERNAME=windff \
 --env DOCKER_INFLUXDB_INIT_PASSWORD=windffpasswd \
 --env DOCKER_INFLUXDB_INIT_ORG=windff \
 --env DOCKER_INFLUXDB_INIT_BUCKET=windff \
 --rm influxdb:2
