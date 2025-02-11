#!/bin/bash
##############################################################################
##                            Build the image                               ##
##############################################################################
RENDER=nvidia

uid=$(eval "id -u")
gid=$(eval "id -g")
docker build \
  --build-arg RENDER="$RENDER" \
  --build-arg UID="$uid" \
  --build-arg GID="$gid" \
  -f Dockerfile \
  -t stacking/bullet .
