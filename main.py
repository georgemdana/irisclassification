
Set Up

# Import Libraries
import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import time
from sagemaker import get_execution_role
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
