#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd, csv
import boto3, io, os, json, pickle, gzip
import re, pyarrow
import time
from time import gmtime, strftime
import sagemaker
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.estimator import Estimator 
from sagemaker.transformer import Transformer
# from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter
from sagemaker.amazon.amazon_estimator import get_image_uri


# # load a small sample dataset to local (optional)

# In[3]:


"""
def load_dpc_data():
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('innovation-dev-data-ingress-bucket')
    prefix_objs = bucket.objects.filter(Prefix='EAC_D39_DPC/')
    df_dpc = []
    for obj in prefix_objs:
        if obj.key.endswith('.parquet'):
            body = obj.get()['Body'].read()
            temp = pd.read_parquet(io.BytesIO(body))
            df_dpc.append(temp.iloc[::-1])
    df_dpc_total = pd.concat(df_dpc)
    df_dpc_total.reset_index(drop=True, inplace=True)
    df_dpc_total.drop(columns=['daily_profile_coefficient_per_second','rn', 'PARTITION_YYYYMM'])
    return df_dpc_total
df_dpc_total=load_dpc_data()
"""


# In[ ]:


"""
#dask version
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
ProgressBar().register()
df_meter_read_2021=dd.read_parquet('s3://innovation-dev-data-ingress-bucket/METER_READINGS/2021/**/*.parquet', columns=['mpan', 'read_datetime', 'reading_type', 'register_reading', 'valid_read', 'flow_name'])
df_meter_read_2022=dd.read_parquet('s3://innovation-dev-data-ingress-bucket/METER_READINGS/2022/**/*.parquet', columns=['mpan', 'read_datetime', 'reading_type', 'register_reading', 'valid_read', 'flow_name'])
df_meter_read_2023=dd.read_parquet('s3://innovation-dev-data-ingress-bucket/METER_READINGS/2023/**/*.parquet', columns=['mpan', 'read_datetime', 'reading_type', 'register_reading', 'valid_read', 'flow_name'])
"""


# In[186]:


""" the following requires pythena to be installed
s3 = boto3.client('s3')
bucket = 'innovation-dev-athena-results/tables/b922b8ff-ea60-40f6-b89b-557373790d65/'
output_location = 's3://innovation-dev-athena-results/athena-ml/'
conn = pyathena.connect(s3_staging_dir=output_location, region_name='eu-west-2',work_group='V2EngineWorkGroup')
"""


# In[22]:


# load training data locally
from urllib.parse import urlparse
def load_eac_table(s3_data_folder_path):  
    parsed_path = urlparse(s3_data_folder_path)
    bucket = parsed_path.netloc
    prefix = parsed_path.path.lstrip('/')
    prefix_objs = boto3.resource('s3').Bucket(bucket).objects.filter(Prefix=prefix)
    df = []
    for obj in prefix_objs:
        body = obj.get()['Body'].read()
        temp = pd.read_parquet(io.BytesIO(body))
        df.append(temp)
    df_total = pd.concat(df)
    df_total.reset_index(drop=True, inplace=True)
    return df_total


# In[24]:


import psutil
print(psutil.virtual_memory())


# In[31]:


#df_sample_train = load_eac_table('s3://innovation-dev-data-emi-ingress-bucket/sample-data/training-samples/') 
df_sample_test = load_eac_table('s3://innovation-dev-data-emi-ingress-bucket/sample-data/testing/')


# In[32]:


df_sample_test.shape


# In[33]:


df_sample_test.head()


# In[34]:


df_sample_test.dtypes


# In[35]:


print(df_sample_test.isna().sum(axis=0))


# # set up cloud environment 

# In[2]:


account = boto3.client('sts').get_caller_identity()["Account"]
role = sagemaker.get_execution_role()
region = boto3.Session().region_name
sess = sagemaker.Session()
xgboost_container = sagemaker.image_uris.retrieve("xgboost", region, "1.7-1")
#xgboost_container = get_image_uri(region, 'xgboost')

#model_file_name = "xgboost"
model_file_name = "sagemaker-xgboost"


# # if local model: save, upload and deploy the trained model to endpoint for inference

# In[85]:


# if using xgboost api/local model then need to ensure version is the same as the aws xgboost container
#pip install xgboost=1.7.5
#import xgboost


# In[3]:


#pickle.dump(xgb_model, open('model.pkl', 'wb'))
xgboost_model = pickle.load(open('xgboost_175.pkl', 'rb'))


# In[81]:


xgboost_model.save_model(model_file_name)


# In[68]:


get_ipython().system('tar czvf xgboost_model.tar.gz $model_file_name')


# In[ ]:


bucket = "innovation-dev-models" 
prefix = "models"
fObj = open("xgboost_model.tar.gz", "rb")
key = os.path.join(prefix, model_file_name, "xgboost_model.tar.gz")
boto3.Session().resource("s3").Bucket(bucket).Object(key).upload_fileobj(fObj)


# # Sagemaker built-in model

# In[3]:


# built-in model

train_data_path = f's3://innovation-dev-data-emi-ingress-bucket/sample-data/training/'
val_data_path = f's3://innovation-dev-data-emi-ingress-bucket/sample-data/validation/'
output_path = f's3://innovation-dev-models/sagemaker-xgb'

xgboost_estimator = Estimator(image_uri=xgboost_container, 
                    #entry_point='innovation-dev-data-insight.ipynb',
                    #framework_version="1.7-1", 
                    #hyperparameters=hyperparameters,
                    role=role,
                    instance_count=1, 
                    instance_type='ml.m5.4xlarge', 
                    #volume_size=100, # GB 
                    base_job_name = 'xgboost-training-job',
                    output_path=output_path,
                    sagemaker_session=sess)


# In[4]:


# use preset hyperparameters
xgboost_estimator.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:squarederror',
                        num_round=400
                                     )


# In[ ]:


# hyperparamter tuning 

""" 
hyperparameter_ranges = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         'eta': [0.01, 0.02, 0.05, 0.1, 0.2],
                         'gamma': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         'subsample': [0.8, 0.9, 1],
                         'num_round': [300, 400, 500, 600, 700, 800, 900]}
"""

""" 
hyperparameter_ranges = {'max_depth': IntegerParameter(3,10),
                         'eta': ContinuousParameter(0.01, 0.20),
                         'gamma': ContinuousParameter(0,10),
                         'min_child_weight': ContinuousParameter(1,10),
                         'subsample': ContinuousParameter(0.8,1),
                         'num_round': IntegerParameter(400, 900)}


tuner = HyperparameterTuner(estimator = xgboost_estimator,
                            objective_metric_name = 'validation:mse',
                            objective_type='Minimize',
                            hyperparameter_ranges=hyperparameter_ranges,
                             max_jobs=30,
                             max_parallel_jobs=5
                               )

train = sagemaker.inputs.TrainingInput(s3_data=train_data_path, content_type='text/csv')
val = sagemaker.inputs.TrainingInput(s3_data=val_data_path, content_type='text/csv')
tuner.fit({'train': train, 'validation': val})

#smclient.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']
#print(tuner.best_training_job())
#tuner.deploy(instance_type='ml.c5.xlarge', initial_instance_count=1) 

"""


# In[6]:


train = sagemaker.inputs.TrainingInput(train_data_path, content_type="application/x-parquet")
val = sagemaker.inputs.TrainingInput(val_data_path, content_type="application/x-parquet")
xgboost_estimator.fit({'train': train, 'validation': val})


# # deploy model for inference - single model case

# In[11]:


training_job = 'xgboost-training-job-2023-12-05-16-18-59-794/'
model_name = model_file_name + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
#model_url = 'https://s3-{}.amazonaws.com/{}/{}'.format(region,bucket,key)
#model_url='s3://innovation-dev-models/sagemaker-xgb/xgboost-training-job-2023-05-05-17-41-32-982/output/model.tar.gz'
model_url='https://innovation-dev-models.s3.eu-west-2.amazonaws.com/sagemaker-xgb/' + training_job + 'output/model.tar.gz'
client = boto3.client("sagemaker")
print(model_url)


# In[15]:


print(model_name)


# In[12]:


primary_container = {
    "Image": xgboost_container,
    "ModelDataUrl": model_url,
}

create_model_response = client.create_model(
    ModelName=model_name, ExecutionRoleArn=role, PrimaryContainer=primary_container
)

print(create_model_response["ModelArn"])


# In[13]:


endpoint_config_name = "xgboost-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(endpoint_config_name)
create_endpoint_config_response = client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "InstanceType": "ml.m5.xlarge",
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 1,
            "ModelName": model_name,
            "VariantName": "traffic-distribution-over-models",
        }
    ],
)

print("Endpoint Config Arn: " + create_endpoint_config_response["EndpointConfigArn"])


# In[14]:


endpoint_name = "xgboost-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(endpoint_name)
create_endpoint_response = client.create_endpoint(
    EndpointName=endpoint_name, 
    EndpointConfigName=endpoint_config_name,
)
print(create_endpoint_response["EndpointArn"])

resp = client.describe_endpoint(EndpointName=endpoint_name)
status = resp["EndpointStatus"]
print("Status: " + status)

while status == "Creating":
    time.sleep(60)
    resp = client.describe_endpoint(EndpointName=endpoint_name)
    status = resp["EndpointStatus"]
    print("Status: " + status)

print("Arn: " + resp["EndpointArn"])
print("Status: " + status)


# # industry version EAC calculation approach

# In[3]:


#!pip install awscli
#!aws configure list
#!aws sts get-caller-identity 
#!cat container/Dockerfile


# In[4]:


#pickle.dump(ind_eac, open('ind_eac.pkl', 'wb'))
#ind_eac = pickle.load(open('ind_eac.pkl', 'rb'))

#from container.ind_eac import ind_eac 


# In[5]:


# in the following cell, try to push docker image once, it will build repo but the push will fail; then go to ECR to change permissions to allow everyone and tick all actions.


# In[8]:


get_ipython().run_cell_magic('sh', '', '\n# The name of our algorithm\nalgorithm_name=ind-eac\n\ncd container\n\nchmod +x ind_eac/train\nchmod +x ind_eac/serve\n\naccount=$(aws sts get-caller-identity --query Account --output text)\n\n# Get the region defined in the current configuration (default to eu-west-2 if none defined)\nregion=$(aws configure get region)\nregion=${region:-eu-west-2}\n\nfullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"\n\n# If the repository doesn\'t exist in ECR, create it.\naws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1\n\nif [ $? -ne 0 ]\nthen\n    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null\nfi\n\n# Get the login command from ECR and execute it directly\naws ecr get-login-password --region ${region}|docker login --username AWS --password-stdin ${fullname}\n#$(aws ecr get-login --region ${region} --no-include-email)\n\n# Build the docker image locally with the image name and then push it to ECR\n# with the full name.\n\ndocker build -t ${algorithm_name} .\ndocker tag ${algorithm_name} ${fullname}\ndocker push ${fullname}\n')


# In[ ]:


algorithm_name='ind-eac'
image = "{}.dkr.ecr.{}.amazonaws.com/{}:latest".format(account, region, algorithm_name)

#same S3 path as the other xgboost model
output_path = f's3://innovation-dev-models/ind-eac'

industry_version_eac_estimator = sagemaker.estimator.Estimator(
    image_uri=image,
    role=role,
    train_instance_count=1,
    train_instance_type="ml.m5.xlarge",
    base_job_name = 'industry-version-eac-calculation-job',
    output_path=output_path,
    sagemaker_session=sess,
)

industry_version_eac_estimator.fit()


# In[ ]:


from sagemaker.predictor import csv_serializer
ind_eac_endpoint_name = "ind-eac-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
model_name = "ind-eac-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
predictor = industry_version_eac_estimator.deploy(1, instance_type='ml.m5.xlarge', serializer=csv_serializer, endpoint_name=ind_eac_endpoint_name, model_name=model_name)


# In[ ]:


predictor.predict(quick_test.values).decode("utf-8")


# # invoke endpoint to make inferences for both models at Sagemaker endpoints

# In[52]:


# create a user csv file for inference
np.savetxt("test_point.csv", df_sample_test.values[:10,1:], delimiter=",") 


# In[49]:


runtime_client = boto3.client("runtime.sagemaker")


# In[51]:


xgboost_endpoint_name='xgboost-endpoint-2023-12-05-15-30-27'
with open("test_point.csv", "r") as f:
    payload = f.read().strip()
response = runtime_client.invoke_endpoint(EndpointName=xgboost_endpoint_name, ContentType="text/csv", TargetModel="sagemaker-xgb.tar.gz", Body=payload)
results = response["Body"].read().decode("ascii")
print("XGBoost predicted EAC: {}".format(results))


# In[ ]:


ind_eac_endpoint_name = 'ind-eac-endpoint-2023-05-17-19-35-28-146'
with open("test_point.csv", "r") as f:
    payload = f.read().strip()
response = runtime_client.invoke_endpoint(EndpointName=ind_eac_endpoint_name, ContentType="text/csv", Body=payload)
results = response["Body"].read().decode("ascii")
print("Industry version calculated EAC: {}".format(results))


# # invoke the industry version eac calculation at Lambda endpoint

# In[64]:


test_event={}
with open("test_point.csv", "r") as f:
    reader = csv.DictReader(f, fieldnames=['sum_dpc', 'cal_aa', 'previous_eac'])
    key=0
    for row in reader:
        key+=1
        test_event[key]=row    


# In[65]:


test_event


# In[66]:


lambda_client = boto3.client("lambda")
payload=json.dumps(test_event)
response = lambda_client.invoke(FunctionName='industry-version-EAC-calculation', Payload=payload)
print(response['Payload'])
print(response['Payload'].read().decode("utf-8"))


# # deployment for multi-container case

# In[ ]:


algorithm_name='ind-eac'
ind_eac_image_uri = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, algorithm_name)
ind_eac_model_url = 's3://innovation-dev-models/ind-eac/industry-version-eac-calculation-job-2023-05-30-16-10-51-024/output/model.tar.gz'
ind_eac_container = {'ContainerHostname': 'ind-eac-model',
                     'Image': ind_eac_image_uri,
                     'ModelDataUrl': ind_eac_model_url}

xgboost_image_uri = sagemaker.image_uris.retrieve("xgboost", region, "1.7-1")
xgboost_model_url = 's3://innovation-dev-models/sagemaker-xgb/xgboost-training-job-2023-05-09-14-47-17-432/output/model.tar.gz'
xgboost_container = {'ContainerHostname': 'xgboost-model',
                     'Image': xgboost_image_uri,
                     'ModelDataUrl': xgboost_model_url}                  


# In[602]:


#multi_container_model_file_name = 'Synthetic-EAC-multi-container-'
#multi_container_model_name = multi_container_model_file_name + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
multi_container_model_name = 'Synthetic-EAC-multi-container'
client = boto3.client("sagemaker")
create_model_response = client.create_model(ModelName=multi_container_model_name,
                                            #PrimaryContainer=primary_container,
                                            Containers=[ind_eac_container, xgboost_container],
                                            InferenceExecutionConfig={'Mode': 'Direct'},
                                            ExecutionRoleArn=role)


# In[603]:


#multi_container_endpoint_config_name = 'Synthetic-EAC-multi-container-endpoint-config-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
multi_container_endpoint_config_name = 'Synthetic-EAC-multi-container-endpoint-config'
create_multi_container_endpoint_config_response = client.create_endpoint_config(EndpointConfigName=multi_container_endpoint_config_name,
                                                ProductionVariants=[{#'VariantName': 'prod',
                                                                     'VariantName': 'traffic-distribution-over-models',
                                                                     'ModelName': multi_container_model_name,
                                                                     'InstanceType': 'ml.m5.xlarge',
                                                                     'InitialInstanceCount': 1,
                                                                     'InitialVariantWeight': 1,
                                                                    }] )


# In[604]:


#multi_container_endpoint_name = 'Synthetic-EAC-multi-container-endpoint-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
multi_container_endpoint_name = 'Synthetic-EAC-multi-container-endpoint'
create_multi_container_endpoint_response = client.create_endpoint(EndpointName=multi_container_endpoint_name, 
                                                                  EndpointConfigName=multi_container_endpoint_config_name)
print(create_multi_container_endpoint_response["EndpointArn"])

resp = client.describe_endpoint(EndpointName=multi_container_endpoint_name)
status = resp["EndpointStatus"]
print("Status: " + status)

while status == "Creating":
    time.sleep(60)
    resp = client.describe_endpoint(EndpointName=multi_container_endpoint_name)
    status = resp["EndpointStatus"]
    print("Status: " + status)
print("Arn: " + resp["EndpointArn"])
print("Status: " + status)


# In[ ]:


# create a user csv file for inference
np.savetxt("test_point.csv", df_sample_test.values[:10,1:], delimiter=",") 
with open("test_point.csv", "r") as f:
    payload = f.read().strip()


# In[ ]:


runtime_client = boto3.client("runtime.sagemaker")
xgboost_response = runtime_client.invoke_endpoint(EndpointName=multi_container_endpoint_name, 
                                          ContentType="text/csv", 
                                          EndpointName="mnist-multi-container-ep",
                                          #ContentType="application/json",
                                          #Accept="application/json",
                                          #Body=json.dumps({"instances": np.expand_dims(tf_samples, 3).tolist()})
                                          TargetContainerHostname="xgboost-model",
                                          #TargetModel="sagemaker-xgb.tar.gz", 
                                          Body=payload)

ind_eac_response = runtime_client.invoke_endpoint(EndpointName=multi_container_endpoint_name, 
                                          ContentType="text/csv", 
                                          EndpointName="mnist-multi-container-ep",
                                          #ContentType="application/json",
                                          #Accept="application/json",
                                          #Body=json.dumps({"inputs": np.expand_dims(tf_samples, 3).tolist()})
                                          TargetContainerHostname="ind-eac-model",
                                          #TargetModel="sagemaker-xgb.tar.gz", 
                                          Body=payload)

xgboost_results = xgboost_response["Body"].read().decode("ascii")
ind_eac_results = ind_eac_response["Body"].read().decode("ascii")
print("XGBoost predicted EAC: {}".format(xgboost_results))
print("Industry-version equation calculated EAC: {}".format(ind_eac_results))


# In[420]:





# # deployment for multi-model case
# #only works for model with same frameworks

# In[455]:


from sagemaker.multidatamodel import MultiDataModel


# In[473]:


ind_eac_image_uri = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, algorithm_name)
ind_eac_model_url = 'https://innovation-dev-models.s3.eu-west-2.amazonaws.com/ind-eac/industry-version-eac-calculation-job-2023-05-30-12-58-03-869/output/model.tar.gz'
ind_eac_container = {'ContainerHostname': 'ind-eac-model',
                     'Image': ind_eac_image_uri,
                     'ModelDataUrl': ind_eac_model_url}

xgboost_image_uri = sagemaker.image_uris.retrieve("xgboost", region, "1.7-1")
xgboost_model_url = 'https://innovation-dev-models.s3.eu-west-2.amazonaws.com/sagemaker-xgb/xgboost-training-job-2023-05-05-17-41-32-982/output/model.tar.gz'
xgboost_container = {'ContainerHostname': 'xgboost-model',
                     'Image': xgboost_image_uri,
                     'ModelDataUrl': xgboost_model_url}  



model_data_url='s3://innovation-dev-models/multi-model/'
multi_model_container = {"Image": ind_eac_image_uri,
             "ModelDataUrl": model_data_url, 
             "Mode": "MultiModel"}


# In[474]:


multi_model_model_name = 'Synthetic-EAC-multi-model'
create_model_response = client.create_model(ModelName=multi_model_model_name, 
                                            ExecutionRoleArn=role, 
                                            PrimaryContainer=multi_model_container)
print("Model Arn: " + create_model_response["ModelArn"])


# In[475]:


multi_model_endpoint_config_name = 'Synthetic-EAC-multi-model-endpoint-config'
create_multi_model_endpoint_config_response = client.create_endpoint_config(EndpointConfigName=multi_model_endpoint_config_name,
                                                                             ProductionVariants=[
                                                                                 {"InstanceType": "ml.m5.xlarge",
                                                                                  "InitialVariantWeight": 1,
                                                                                  "InitialInstanceCount": 1,
                                                                                  "ModelName": multi_model_model_name,
                                                                                  "VariantName": "AllTraffic"}])
print("Endpoint Config Arn: " + create_multi_model_endpoint_config_response["EndpointConfigArn"])


# In[478]:


multi_model_endpoint_name = 'Synthetic-EAC-multi-model-endpoint'
create_multi_model_endpoint_response = client.create_endpoint(EndpointName=multi_model_endpoint_name, 
                                                              EndpointConfigName=multi_model_endpoint_config_name)
print("Endpoint Arn: " + create_multi_model_endpoint_response["EndpointArn"])


resp = client.describe_endpoint(EndpointName=multi_model_endpoint_name)
status = resp["EndpointStatus"]
print("Status: " + status)

while status == "Creating":
    time.sleep(60)
    resp = client.describe_endpoint(EndpointName=multi_model_endpoint_name)
    status = resp["EndpointStatus"]
    print("Status: " + status)
print("Arn: " + resp["EndpointArn"])
print("Status: " + status)


# In[184]:


sagemaker_xgb = xgboost_estimator.create_model(role=role, image_uri=xgboost_container)
endpoint_name = "xgboost-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())


# In[190]:


mme = MultiDataModel(
    name=model_name,
    model_data_prefix=f's3://innovation-dev-models/multi-model/',
    model=sagemaker_xgb,
    sagemaker_session=sess
)
predictor = mme.deploy(
    initial_instance_count=1, instance_type='ml.m5.xlarge', endpoint_name=endpoint_name
)


# In[194]:


list(mme.list_models())


# In[192]:


artifact_path = xgboost_estimator.latest_training_job.describe()["ModelArtifacts"]["S3ModelArtifacts"]
model_name = artifact_path.split("/")[-4] + ".tar.gz"
# This is copying over the model artifact to the S3 location for the MME.
mme.add_model(model_data_source=artifact_path, model_data_path=model_name)


# # batch transformation on S3 bucket data with trained model 

# In[17]:


"""
# serialization and deserialization for mode handler to accept parquet format for batch transform jobs (not applicable still)

from io import BytesIO
from typing import BinaryIO
import pandas as pd
from botocore.response import StreamingBody
def input_fn(
  serialized_input_data: StreamingBody,
  content_type: str = "application/x-parquet",
) -> pd.DataFrame:
  # Deserialize inputs
  if content_type == "application/x-parquet":
    data = BytesIO(serialized_input_data)
    df = pd.read_parquet(data)
    return df
  else:
    raise ValueError(
      "Expected `application/x-parquet`."
    )

def output_fn(output: pd.DataFrame, accept: str = "application/x-parquet") -> BinaryIO:
  # Model output handler 
  if accept == "application/x-parquet":
    buffer = BytesIO()
    output.to_parquet(buffer)
    
    return buffer.getvalue()
  else:
    raise Exception("Requested unsupported ContentType in Accept: " + accept)
"""


# In[7]:


TrainingJobName = 'xgboost-training-job-2023-12-11-12-28-11-261'
xgboost_estimator = Estimator.attach(TrainingJobName)


# In[8]:


# batch transformation from the following 
batch_input_path = 's3://innovation-dev-data-emi-ingress-bucket/batch-transform/input/'
batch_output_path = 's3://innovation-dev-data-emi-ingress-bucket/batch-transform/output/'
xgboost_transformer = xgboost_estimator.transformer(
    instance_count=16,
    instance_type='ml.c5.xlarge',
    max_concurrent_transforms=100,
    max_payload = 1,
    output_path=batch_output_path,
    #accept='application/jsonlines',
    accept='text/csv',
    assemble_with='Line',                         
    #strategy='SingleRecord'
            )

# calls that object's transform method to create a transform job
xgboost_transformer.transform(
    data=batch_input_path, 
    content_type='text/csv', 
    #content_type='application/jsonlines', 
    join_source='Input', 
    split_type='Line', 
    input_filter='$[1:]', 
    output_filter='$[0,-1]'
            )

xgboost_transformer.wait()


# In[ ]:




