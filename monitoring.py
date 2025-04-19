Set Up

# Import Libraries
# Build Model
import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import time
import sagemaker
from sagemaker import get_execution_role

# Monitoring
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
import json

# Monitoring Scheduling
from sagemaker import Session
from sagemaker.model_monitor import DefaultModelMonitor, DataCaptureConfig

role = get_execution_role()
# Load Data from S3
s3 = boto3.client('s3')
bucket = 'dmg-iris-hakkoda'
key = 'iris.csv'
obj = s3.get_object(Bucket=bucket, Key=key)
df = pd.read_csv(obj['Body'])
# Prep Data
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
Build and Evaluate Model

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)
RandomForestClassifier()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")
Model accuracy: 0.9666666666666667
Save Model to S3

# Save Model
joblib.dump(model, 'dmg-iris-model.joblib')
['dmg-iris-model.joblib']
# Upload Model to S3
s3.upload_file('dmg-iris-model.joblib', bucket, 'models/dmg-iris-model.joblib')
Save Model to Model Registry

# Create SageMaker Model Package Group (this is a container for versioned model packages)
# This step creates a Model Package Group in SageMaker, allowing multiple model versions to be stored and managed.

sagemaker_client = boto3.client("sagemaker")

model_package_group_name = "dmg-iris-package-group"  # Change this to your desired name

# Create the model package group
response = sagemaker_client.create_model_package_group(
    ModelPackageGroupName=model_package_group_name,
    ModelPackageGroupDescription="Model group for versioned SageMaker models"
)

print(f"Model Package Group ARN: {response['ModelPackageGroupArn']}")
Model Package Group ARN: arn:aws:sagemaker:us-east-1:878763418883:model-package-group/dmg-iris-package-group
# Create a Model Package
# Register it in the SageMaker Model Registry

# Explanation of Parameters:
# ModelPackageGroupName → The group to store your versioned models.
# ModelApprovalStatus → Can be "PendingManualApproval" or "Approved".
# InferenceSpecification → Defines how the model will be used for inference.

model_artifact_s3 = f"s3://{bucket}/models/dmg-iris-model.joblib"
inference_container_image = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1"  # Scikit-Learn container

response = sagemaker_client.create_model_package(
    ModelPackageGroupName=model_package_group_name,
    ModelPackageDescription="Registered locally trained model",
    ModelApprovalStatus="PendingManualApproval",
    InferenceSpecification={
        "Containers": [
            {
                "Image": inference_container_image,
                "ModelDataUrl": model_artifact_s3,
            }
        ],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"],
    },
)

print(f"Model Package ARN: {response['ModelPackageArn']}")
Model Package ARN: arn:aws:sagemaker:us-east-1:878763418883:model-package/dmg-iris-package-group/1
# Check Model Status
response = sagemaker_client.list_model_packages(ModelPackageGroupName=model_package_group_name)

for model in response["ModelPackageSummaryList"]:
    print(f"Model Package Name: {model['ModelPackageArn']}")
    print(f"Status: {model['ModelApprovalStatus']}")
    print("----")
Model Package Name: arn:aws:sagemaker:us-east-1:878763418883:model-package/dmg-iris-package-group/1
Status: PendingManualApproval
----
# If needs manual approval:
model_package_arn = "arn:aws:sagemaker:us-east-1:878763418883:model-package/dmg-iris-package-group/1"

sagemaker_client.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus="Approved",
    ApprovalDescription="Model validated and approved for deployment."
)

print("Model successfully approved!")
Model successfully approved!
Set up Model Monitoring

# Set up Baseline
# Analyze a baseline dataset to understand the data schema, statistics, and constraints.

# Initialize SageMaker session and role
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Define variables
baseline_dataset_uri = 's3://dmg-iris-hakkoda/iris.csv'
baseline_output_uri = 's3://dmg-iris-hakkoda'

# Create a default model monitor instance
model_monitor = DefaultModelMonitor(
    role=role,
    sagemaker_session=sagemaker_session
)

# Create a baseline job
model_monitor.suggest_baseline(
    baseline_dataset=baseline_dataset_uri,
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=baseline_output_uri,
    wait=True
) 

# Verify that the expected files were created in baseline job
# You can also do this by looking in your s3 bucket for the expected files

# List objects in the baseline output prefix
response = s3.list_objects_v2(Bucket=bucket)#, Prefix=prefix)

# Print the files in the output directory
print("Files in baseline output directory:")
for obj in response.get('Contents', []):
    print(obj['Key'])

# Download and print the content of statistics.json
statistics_key = "statistics.json"

statistics_obj = s3.get_object(Bucket=bucket, Key=statistics_key)
statistics_content = statistics_obj['Body'].read().decode('utf-8')
statistics_data = json.loads(statistics_content)
print("Statistics Data:", json.dumps(statistics_data, indent=4))

# Download and print the content of constraints.json
constraints_key = "constraints.json"
constraints_obj = s3.get_object(Bucket=bucket, Key=constraints_key)
constraints_content = constraints_obj['Body'].read().decode('utf-8')
constraints_data = json.loads(constraints_content)
print("Constraints Data:", json.dumps(constraints_data, indent=4))

# Set up Monitoring Schedule
# Set up a monitoring schedule to periodically analyze the data being sent to your model and detect any deviations from the baseline.
# Once the baseline job is complete and the baseline constraints are available, you can set up a monitoring schedule.
# periodically analyze the data being sent to your model endpoint and detect any deviations from the baseline.
# 2 steps:
# Define the Monitoring Schedule: This step involves setting up the monitoring schedule to periodically analyze the data being sent to your model endpoint.
# Configure Data Capture: Ensure that data capture is enabled for your endpoint to collect incoming requests and responses.

# Initialize SageMaker session and role
sagemaker_session = Session()
role = get_execution_role()

# Define variables
endpoint_name = 'dmg-iris-endpoint'
endpoint_config_name = 'dmg-iris-endpoint-config'
monitoring_schedule_name = 'dmg-iris-monitoring-schedule'
monitoring_output_uri = 's3://dmg-iris-hakkoda/monitoring-output'
baseline_dataset_uri = 's3://dmg-iris-hakkoda/iris.csv'
baseline_output_uri = 's3://dmg-iris-hakkoda'

# Create a default model monitor instance
model_monitor = DefaultModelMonitor(
    role=role,
    sagemaker_session=sagemaker_session
)

# Set up data capture configuration
data_capture_config = DataCaptureConfig(
    destination_s3_uri=monitoring_output_uri,
    enable_capture=True,
    sampling_percentage=100,
    capture_options=["REQUEST", "RESPONSE"]
)

# Create a new endpoint configuration with data capture enabled
sagemaker_client = boto3.client('sagemaker')
response = sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            'VariantName': 'AllTraffic',
            'ModelName': 'dmg-iris-model',
            'InstanceType': 'ml.m5.large',
            'InitialInstanceCount': 1,
        },
    ],
    DataCaptureConfig={
        'EnableCapture': True,
        'InitialSamplingPercentage': 100,
        'DestinationS3Uri': monitoring_output_uri,
        'CaptureOptions': [
            {'CaptureMode': 'Input'},
            {'CaptureMode': 'Output'},
        ],
    }
)

# Update the endpoint to use the new endpoint configuration
sagemaker_client.update_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

# Create monitoring schedule
model_monitor.create_monitoring_schedule(
    endpoint_input=endpoint_name,
    output=monitoring_output_uri,
    # schedule_cron_expression="cron(0 0 * * ? *)",  # This cron expression represents daily monitoring
    schedule_cron_expression="cron(0 0 1 */3 ? *)",  # This cron expression represents every 3 months
    monitoring_schedule_name=monitoring_schedule_name,
    ground_truth_input=baseline_dataset_uri
)

