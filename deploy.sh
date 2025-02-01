currentDirectory="$(pwd)"
imageName=student

echo "Searching for Docker image ..."
DOCKER_IMAGE_ID=$(docker images -q $imageName | head -n 1)
echo "Found and using ${DOCKER_IMAGE_ID}"

#xhost + $(hostname)
#host.docker.internal:0
container_id=$(docker run -d --network host --user docker:docker \
 -e DISPLAY=unix$DISPLAY \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -v `pwd`/Work:/home/docker/Work \
 ${DOCKER_IMAGE_ID} sleep infinity) 

 echo "Container ID: $container_id"

 docker exec -it --user docker:docker \
 -e DISPLAY=unix$DISPLAY \
 -e TERM=xterm-256color \
 $container_id python3 /home/docker/Work/main.py

#host.docker.internal:0
docker exec -it --user docker:docker \
 -e DISPLAY=unix$DISPLAY \
 -e TERM=xterm-256color \
 $container_id /bin/bash

docker commit $container_id student
docker stop $container_id
docker rm $container_id

#xhost - $(hostname)
