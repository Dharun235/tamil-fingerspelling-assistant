#!/usr/bin/env bash
set -euo pipefail

# Start the public AWS demo in Stockholm.
REGION="eu-north-1"
ACCOUNT_ID="108464427575"
SERVICE_NAME="tamil-fingerspelling-prod"
IMAGE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/tamil-fingerspelling:v5"
EXECUTION_ROLE="arn:aws:iam::${ACCOUNT_ID}:role/ecsTaskExecutionRole"
INFRASTRUCTURE_ROLE="arn:aws:iam::${ACCOUNT_ID}:role/ecsInfrastructureRoleForExpressServices"

existing_state=$(aws ecs describe-express-gateway-service \
  --service-arn "arn:aws:ecs:${REGION}:${ACCOUNT_ID}:service/default/${SERVICE_NAME}" \
  --region "${REGION}" \
  --query 'service.status.statusCode' \
  --output text 2>/dev/null || true)

if [ "${existing_state}" = "ACTIVE" ]; then
  echo "AWS service already exists: ${SERVICE_NAME}"
  aws ecs describe-express-gateway-service \
    --service-arn "arn:aws:ecs:${REGION}:${ACCOUNT_ID}:service/default/${SERVICE_NAME}" \
    --region "${REGION}" \
    --query 'service.activeConfigurations[0].ingressPaths[0].endpoint' \
    --output text
  exit 0
fi

if [ "${existing_state}" = "DRAINING" ]; then
  echo "AWS service is still being deleted; waiting before recreating it..."
  for attempt in $(seq 1 60); do
    sleep 10
    existing_state=$(aws ecs describe-express-gateway-service \
      --service-arn "arn:aws:ecs:${REGION}:${ACCOUNT_ID}:service/default/${SERVICE_NAME}" \
      --region "${REGION}" \
      --query 'service.status.statusCode' \
      --output text 2>/dev/null || true)
    if [ -z "${existing_state}" ] || [ "${existing_state}" = "None" ]; then
      break
    fi
    echo "Still deleting (${attempt}/60): ${existing_state}"
  done
  if [ -n "${existing_state}" ] && [ "${existing_state}" != "None" ]; then
    echo "Timed out waiting for AWS service deletion. Try again later."
    exit 1
  fi
fi

echo "Starting ${SERVICE_NAME} in ${REGION}..."
aws ecs create-express-gateway-service \
  --service-name "${SERVICE_NAME}" \
  --infrastructure-role-arn "${INFRASTRUCTURE_ROLE}" \
  --execution-role-arn "${EXECUTION_ROLE}" \
  --primary-container "image=${IMAGE},containerPort=8000,awsLogsConfiguration={logGroup=/aws/ecs/default/${SERVICE_NAME},logStreamPrefix=ecs}" \
  --health-check-path "/health" \
  --cpu 1024 \
  --memory 2048 \
  --cpu-architecture X86_64 \
  --scaling-target "minTaskCount=1,maxTaskCount=20,autoScalingMetric=AVERAGE_CPU,autoScalingTargetValue=60" \
  --region "${REGION}"

echo "AWS service creation requested. Wait until status becomes ACTIVE."
