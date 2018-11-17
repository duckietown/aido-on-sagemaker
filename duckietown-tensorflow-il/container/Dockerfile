FROM tensorflow/tensorflow:1.8.0-py3

RUN apt-get update && apt-get install -y --no-install-recommends nginx curl

# Installs for gym-duckietown
RUN apt-get install -y freeglut3-dev xvfb xorg-dev libglu1-mesa libgl1-mesa-dev libxinerama1 libxcursor1
RUN apt-get install -y git python-pip
RUN git clone -b aido1_lf1_r3-v3 https://github.com/duckietown/gym-duckietown src/gym-duckietown
RUN pip install -e src/gym-duckietown/
RUN pip install opencv-python

ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY /duckietown-il /opt/ml/code/

ENV PYTHONPATH="/opt/ml/code/duckietown-il:/opt/ml/code/:${PYTHONPATH}"

# This entrypoint will help us launch the display
ADD entrypoint.sh /entrypoint.sh
RUN ["chmod", "+x", "/entrypoint.sh"]
ENTRYPOINT ["/entrypoint.sh"]

WORKDIR "/opt/ml/code"

