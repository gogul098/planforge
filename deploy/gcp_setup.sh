#!/usr/bin/env bash
# One-time Google Cloud setup. Run in Cloud Shell (console.cloud.google.com -> the >_ icon, top right):
#   bash gcp_setup.sh <PROJECT_ID> <HF_TOKEN>
# Creates: image registry (keeps only the newest images), HF_TOKEN in Secret Manager, a deploy
# service account, and a 100 INR budget alert. Prints the GCP_SA_KEY JSON for GitHub secrets.
set -euo pipefail
PROJECT=${1:?project id}; HF_TOKEN=${2:?HF token}
REGION=asia-south1; REPO=planforge; SA=planforge-deploy

gcloud config set project "$PROJECT" >/dev/null
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com \
  billingbudgets.googleapis.com iam.googleapis.com cloudresourcemanager.googleapis.com

# image registry; cleanup policy keeps the 2 newest images (storage stays ~free)
gcloud artifacts repositories describe $REPO --location=$REGION >/dev/null 2>&1 || \
  gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION
cat > /tmp/cleanup.json <<'J'
[{"name":"keep-newest","action":{"type":"Keep"},"mostRecentVersions":{"keepCount":2}},
 {"name":"delete-rest","action":{"type":"Delete"},"condition":{"tagState":"ANY"}}]
J
gcloud artifacts repositories set-cleanup-policies $REPO --location=$REGION --policy=/tmp/cleanup.json --no-dry-run

# HF token as a secret
if gcloud secrets describe hf-token >/dev/null 2>&1; then
  printf '%s' "$HF_TOKEN" | gcloud secrets versions add hf-token --data-file=-
else
  printf '%s' "$HF_TOKEN" | gcloud secrets create hf-token --data-file=-
fi
PNUM=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding hf-token --role=roles/secretmanager.secretAccessor \
  --member="serviceAccount:${PNUM}-compute@developer.gserviceaccount.com" >/dev/null

# deploy account for GitHub Actions
gcloud iam service-accounts describe $SA@$PROJECT.iam.gserviceaccount.com >/dev/null 2>&1 || \
  gcloud iam service-accounts create $SA --display-name="PlanForge GitHub deploy"
for ROLE in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA@$PROJECT.iam.gserviceaccount.com" \
    --role=$ROLE --condition=None >/dev/null
done

# budget alert at 100 INR (emails you at 50%, 90%, 100%)
BILL=$(gcloud billing projects describe "$PROJECT" --format='value(billingAccountName)' | sed 's#billingAccounts/##')
gcloud billing budgets create --billing-account="$BILL" --display-name="planforge-alert" \
  --budget-amount=100INR --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0 \
  >/dev/null 2>&1 || echo "(budget: create it by hand in Billing -> Budgets & alerts if this failed)"

gcloud iam service-accounts keys create /tmp/key.json --iam-account=$SA@$PROJECT.iam.gserviceaccount.com
echo; echo "===== GCP_SA_KEY (copy everything between the lines) ====="; cat /tmp/key.json; echo
echo "==========================================================="; rm /tmp/key.json
echo "Also add GitHub secret GCP_PROJECT = $PROJECT"
