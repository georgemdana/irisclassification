### Iris Species Classification with AWS SageMaker
#### Overview
This Python script trains a Random Forest Classifier on the Iris dataset, evaluates its performance, and integrates it with AWS services for storage and model management. The script loads data from an S3 bucket, trains the model, saves it to S3, and registers it in the SageMaker Model Registry for versioned model management.

#### Purpose
Load and preprocess the Iris dataset from an S3 bucket.
Train and evaluate a Random Forest Classifier.
Save the trained model to S3.
Register the model in SageMaker Model Registry for deployment and versioning.

#### Features
Fetches the Iris dataset (iris.csv) from an S3 bucket.
Trains a Random Forest Classifier using scikit-learn.
Evaluates model accuracy on a test set.
Stores the trained model in S3.
Registers the model in SageMaker Model Registry with version control and approval workflow.

#### Dependencies
Python 3.x
boto3 (AWS SDK for Python)
pandas (data manipulation)
numpy (numerical operations)
scikit-learn (machine learning)
joblib (model serialization)
sagemaker (AWS SageMaker SDK)

Install dependencies using:
pip install boto3 pandas numpy scikit-learn joblib sagemaker

#### Prerequisites
AWS account with access to S3 and SageMaker.
An S3 bucket named dmg-iris-hakkoda containing iris.csv.
Proper AWS credentials configured (e.g., via AWS CLI or environment variables).
SageMaker execution role with permissions for S3 and SageMaker operations.

#### Setup
Ensure the Iris dataset (iris.csv) is uploaded to the S3 bucket dmg-iris-hakkoda with the key iris.csv.
Configure AWS credentials and region (default: us-east-1).
Verify the SageMaker execution role has necessary permissions.

#### Usage
Run the script in a Jupyter notebook or Python environment:
python iris_classification.py

#### The script will:
Load iris.csv from S3.
Split data into training and test sets (80/20 split).
Train a Random Forest Classifier.
Print model accuracy (e.g., Model accuracy: 0.9666666666666667).
Save the model as dmg-iris-model.joblib and upload it to S3 (s3://dmg-iris-hakkoda/models/dmg-iris-model.joblib).
Register the model in SageMaker Model Registry under the group dmg-iris-package-group.
Approve the model package for deployment.

#### Script Details
1. Load Data from S3
Uses boto3 to fetch iris.csv from the specified S3 bucket and key.
Loads the data into a pandas DataFrame.

2. Prep Data
Drops the species column to create feature set X.
Uses species as the target variable y.
Splits data into training (80%) and test (20%) sets using train_test_split.

3. Build and Evaluate Model
Trains a RandomForestClassifier on the training data.
Predicts on the test set and calculates accuracy using accuracy_score.
Outputs the model accuracy.

4. Save Model to S3
Serializes the model to dmg-iris-model.joblib using joblib.
Uploads the model file to S3 at s3://dmg-iris-hakkoda/models/dmg-iris-model.joblib.

5. Save Model to Model Registry
Creates a SageMaker Model Package Group (dmg-iris-package-group) to store versioned models.
Registers the model package with:
S3 model artifact location.
Scikit-learn inference container (sagemaker-scikit-learn:1.2-1).
Support for JSON input/output.
Sets initial approval status to PendingManualApproval.
Approves the model package programmatically for deployment.

Example Output
Model accuracy: 0.9666666666666667
Model Package Group ARN: arn:aws:sagemaker:us-east-1:878763418883:model-package-group/dmg-iris-package-group
Model Package ARN: arn:aws:sagemaker:us-east-1:878763418883:model-package/dmg-iris-package-group/1
Model Package Name: arn:aws:sagemaker:us-east-1:878763418883:model-package/dmg-iris-package-group/1
Status: PendingManualApproval

#### Notes
The script assumes the S3 bucket dmg-iris-hakkoda and the SageMaker execution role are pre-configured.
The model is registered with PendingManualApproval status and then programmatically approved. You can modify the approval process as needed.
The inference container is specific to scikit-learn 1.2-1; ensure compatibility with your model.
The script is designed for the us-east-1 region; update the region and container image URI if using a different region.

#### Author
Dana M. George
