#!/usr/bin/env bash
set -euo pipefail

# Stop the public AWS demo and remove its billable runtime infrastructure.
REGION="eu-north-1"
ACCOUNT_ID="108464427575"
SERVICE_NAME="tamil-fingerspelling-prod"
SERVICE_ARN="arn:aws:ecs:${REGION}:${ACCOUNT_ID}:service/default/${SERVICE_NAME}"

existing_state=$(aws ecs describe-express-gateway-service \
  --service-arn "${SERVICE_ARN}" \
  --region "${REGION}" \
  --query 'service.status.statusCode' \
  --output text 2>/dev/null || true)

if [ -z "${existing_state}" ] || [ "${existing_state}" = "None" ]; then
  echo "AWS service already stopped."
  exit 0
fi

if [ "${existing_state}" = "DRAINING" ]; then
  echo "AWS service is already stopping."
  exit 0
fi

echo "Stopping ${SERVICE_NAME}..."
aws ecs delete-express-gateway-service \
  --service-arn "${SERVICE_ARN}" \
  --region "${REGION}"

echo "Stop requested. Fargate and the load balancer will be removed."
