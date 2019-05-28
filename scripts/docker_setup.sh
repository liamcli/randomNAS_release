docker build -f randomnas.dockerfile .
docker tag SHA liamcli/randomnas:latest
docker run -it --rm --mount type=bind,source=/home/ubuntu/data,target=/data liamcli/randomnas:latest
