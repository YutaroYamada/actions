FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip3 install \
  pytorch-lightning==1.6.3 \
  pandas \
  seaborn \
  matplotlib \
  scikit-learn \
  sagemaker-training

# Copies the training code inside the container
COPY train.py /opt/ml/code/train.py
COPY utils.py /opt/ml/code/utils.py

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py