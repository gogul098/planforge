#!/usr/bin/env bash
# One-time setup. Run in Azure Cloud Shell (Bash): portal.azure.com -> the >_ icon at the top.
#   bash azure_setup.sh <github-user> <HF_TOKEN>
# Creates a Container App (2 CPU / 4 GB, sleeps when idle) and prints the AZURE_CREDENTIALS
# JSON to paste into GitHub -> repo Settings -> Secrets and variables -> Actions.
set -euo pipefail
GH_USER=${1:?github user}; HF_TOKEN=${2:?HF token}
RG=planforge-rg; ENV=planforge-env; APP=planforge; LOC=centralindia

az extension add --name containerapp --upgrade -y --only-show-errors
az provider register -n Microsoft.App --wait
az provider register -n Microsoft.OperationalInsights --wait
az group create -n $RG -l $LOC -o none
az containerapp env create -n $ENV -g $RG -l $LOC --logs-destination none -o none
az containerapp create -n $APP -g $RG --environment $ENV \
  --image mcr.microsoft.com/k8se/quickstart:latest --target-port 8000 --ingress external \
  --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 1 \
  --secrets hf-token="$HF_TOKEN" --env-vars HF_TOKEN=secretref:hf-token -o none
# public image on ghcr.io after the first GitHub build; the workflow switches the image
SUB=$(az account show --query id -o tsv)
echo; echo "===== AZURE_CREDENTIALS (copy everything between the lines) ====="
az ad sp create-for-rbac --name planforge-deploy --role contributor \
  --scopes /subscriptions/$SUB/resourceGroups/$RG --json-auth
echo "=================================================================="
echo "App URL: https://$(az containerapp show -n $APP -g $RG --query properties.configuration.ingress.fqdn -o tsv)"
