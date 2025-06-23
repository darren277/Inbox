include .env

# Docker
auth:
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(DOCKER_REGISTRY)

create-repos:
	aws ecr create-repository --repository-name $(PROCESS_IMAGE) --region us-east-1 || true
	aws ecr create-repository --repository-name $(DASHBOARD_IMAGE) --region us-east-1 || true
	aws ecr create-repository --repository-name $(INGEST_COMMUNICATIONS_IMAGE) --region us-east-1 || true
	aws ecr create-repository --repository-name $(INGEST_APICRONJOB_IMAGE) --region us-east-1 || true
	aws ecr create-repository --repository-name $(INGEST_WEBHOOK_IMAGE) --region us-east-1 || true
	aws ecr create-repository --repository-name $(INGEST_RSS_IMAGE) --region us-east-1 || true

docker-process:
	cd lib/process && docker build --build-arg PORT=$(FLASK_PORT) -t $(DOCKER_REGISTRY)/$(PROCESS_IMAGE):$(PROCESS_VERSION) -f Dockerfile .
	docker push $(DOCKER_REGISTRY)/$(PROCESS_IMAGE):$(PROCESS_VERSION)

docker-dashboard:
	cd lib/dashboard && docker build --build-arg PORT=$(FLASK_PORT) -t $(DOCKER_REGISTRY)/$(DASHBOARD_IMAGE):$(DASHBOARD_VERSION) -f Dockerfile .
	docker push $(DOCKER_REGISTRY)/$(DASHBOARD_IMAGE):$(DASHBOARD_VERSION)

docker-ingest-communications:
	cd lib/ingest/communications && docker build --build-arg PORT=$(FLASK_PORT) -t $(DOCKER_REGISTRY)/$(INGEST_COMMUNICATIONS_IMAGE):$(INGEST_COMMUNICATIONS_VERSION) -f Dockerfile .
	docker push $(DOCKER_REGISTRY)/$(INGEST_COMMUNICATIONS_IMAGE):$(INGEST_COMMUNICATIONS_VERSION)

docker-ingest-apicronjob:
	cd lib/ingest/apicronjob && docker build --build-arg PORT=$(FLASK_PORT) -t $(DOCKER_REGISTRY)/$(INGEST_APICRONJOB_IMAGE):$(INGEST_APICRONJOB_VERSION) -f Dockerfile .
	docker push $(DOCKER_REGISTRY)/$(INGEST_APICRONJOB_IMAGE):$(INGEST_APICRONJOB_VERSION)

docker-ingest-webhook:
	cd lib/ingest/webhook && docker build --build-arg PORT=$(FLASK_PORT) -t $(DOCKER_REGISTRY)/$(INGEST_WEBHOOK_IMAGE):$(INGEST_WEBHOOK_VERSION) -f Dockerfile .
	docker push $(DOCKER_REGISTRY)/$(INGEST_WEBHOOK_IMAGE):$(INGEST_WEBHOOK_VERSION)

docker-ingest-rss:
	cd lib/ingest/rss && docker build --build-arg PORT=$(FLASK_PORT) -t $(DOCKER_REGISTRY)/$(INGEST_RSS_IMAGE):$(INGEST_RSS_VERSION) -f Dockerfile .
	docker push $(DOCKER_REGISTRY)/$(INGEST_RSS_IMAGE):$(INGEST_RSS_VERSION)

docker-all: docker-process docker-dashboard docker-ingest-communications docker-ingest-apicronjob docker-ingest-webhook docker-ingest-rss


# Kubernetes and Helm
k8s-init:
	kubectl create namespace $(NAMESPACE) || true
	helm repo update
	helm repo add strimzi https://strimzi.io/charts/ || true
	helm upgrade --install strimzi-operator strimzi/strimzi-kafka-operator -n $(NAMESPACE) --version 0.46.0

k8s-auth:
	kubectl create secret docker-registry ecr-secret --docker-server=$(DOCKER_REGISTRY) --docker-username=AWS --docker-password=$(DOCKER_PASSWORD) --namespace=$(NAMESPACE)

SECRETS=--set surrealdb.secret.user=$(SURREALDB_USER) --set surrealdb.secret.pass=$(SURREALDB_PASS)
k8s-deploy:
	kubectl create namespace $(NAMESPACE) || true
	helm upgrade --install $(NAMESPACE) ./k8s --namespace $(NAMESPACE) $(SECRETS) -f ./k8s/values.yaml

k8s-debug:
	kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	helm template $(NAMESPACE) ./k8s -f ./k8s/values.yaml | kubectl apply --namespace $(NAMESPACE) -f - --dry-run=server



# `my-app` can be any random string.
# helm template my-app ./k8s --namespace relayops

# helm history inbox -n inbox
