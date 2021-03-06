# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# For more information on creating a Dockerfile
# https://docs.docker.com/compose/gettingstarted/#step-2-create-a-dockerfile
# https://github.com/awslabs/amazon-sagemaker-examples/master/advanced_functionality/pytorch_extending_our_containers/pytorch_extending_our_containers.ipynb
ARG REGION=us-east-1

# SageMaker PyTorch image

# needs to also be gpu??
FROM 520713654638.dkr.ecr.$REGION.amazonaws.com/sagemaker-pytorch:0.4.0-cpu-py3

# Installs for gym-duckietown
RUN apt=get update
RUN apt-get install -y freeglut3-dev xvfb xorg-dev libglu1-mesa libgl1-mesa-dev libxinerama1 libxcursor1
RUN git clone -b aido1_lf1_r3-v3 https://github.com/duckietown/gym-duckietown src/gym-duckietown
RUN pip install -e src/gym-duckietown/

ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY /duckietown-rl /opt/ml/code/duckietown-rl

ENV PYTHONPATH="/opt/ml/code/duckietown-rl:/opt/ml/code/:${PYTHONPATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# this environment variable is used by the SageMaker PyTorch container to determine our program entry point
# for training and serving.
# For more information: https://github.com/aws/sagemaker-pytorch-container
ENV SAGEMAKER_PROGRAM train_ddpg.py

# This entrypoint will help us launch the display
ADD entrypoint.sh /entrypoint.sh
RUN ["chmod", "+x", "/entrypoint.sh"]
ENTRYPOINT ["/entrypoint.sh"]
