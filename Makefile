# Variables
DOCKER_USER ?= jesusgf23 #Your user name on Docker Hub
IMAGE_NAME ?= $(DOCKER_USER)/hip_hip_hippie
TAG ?= latest
CONTAINER_NAME ?= hip_hip_hippie

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME):$(TAG) .

# Run the Docker container
run:
	docker run --rm -it --name $(CONTAINER_NAME) $(IMAGE_NAME):$(TAG)


# Start container in detached mode
start:
	docker run -d --name $(CONTAINER_NAME) $(IMAGE_NAME):$(TAG)

# Stop the container
stop:
	docker stop $(CONTAINER_NAME)

# Remove all stopped containers and unused images
prune:
	docker system prune -f

# Push image to Docker Hub
push:
	docker push $(IMAGE_NAME):$(TAG)

# Tag and push with latest
tag-latest:
	docker tag $(IMAGE_NAME):$(TAG) $(IMAGE_NAME):latest
	docker push $(IMAGE_NAME):latest

go:
	make build && make tag-latest && make push