#!/bin/bash
docker build -t liamcli/randomnas:latest -f randnas.dockerfile .
docker run -it --rm --mount type=bind,source=/home/ubuntu/data,target=/data liamcli/randomnas:latest
docker push liamcli/randomnas:latest
