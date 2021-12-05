# ML Ops
ML Ops is carried out in Vertex AI, a GCP based automation platform
## Pre-requisites
### A service account has to be created to link Google Cloud Storage, Vertex AI, Cloud Compute and Cloud Run
### Google Cloud SDK Command tools should have been installed
### Create buckets to store the training data and generated models
	> BUCKET_NAME="gs://<<bucket-name>>"
	> gsutil mb -l us-central1 $BUCKET_NAME
## Two modules exist for each of the model - train and pred. In both the cases, it is important to build the docker containers using the docker files in the respective folders. 

##Move the core/ folder to the appropriate folder 
	> mv -r core/ pred/
	
##Build docker instance on Jupyter Lab
PROJECT_ID="nice-abbey-328722"
	> IMAGE_URI="gcr.io/$PROJECT_ID/zerodce:v1"
	> docker build ./ -t $IMAGE_URI

##Run docker instance for testing
	> docker run $IMAGE_URI

##Deploy to Container Registry
	> docker push $IMAGE_URI

##For Training
##Run Training Pipeline
Once the docker has been tested, the docker can be deployed using the command, a sample request.json is available under base/

curl -X POST \
-H "Authorization: Bearer "$(gcloud auth application-default print-access-token) \
-H "Content-Type: application/json; charset=utf-8" \
-d @request.json \
"https://LOCATION_ID-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/LOCATION_ID/trainingPipelines"

##For Prediction
##Deploy to Cloud Run
The prediction containers have to be deployed to Cloud Run using the command:
	> gcloud run deploy SERVICE --image IMAGE_URL
where SERVICE is the Cloud Run service name and IMAGE_URI is the docker image that has been pushed to the registry