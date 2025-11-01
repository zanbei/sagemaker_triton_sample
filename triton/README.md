# Simplified Whisper ASR SageMaker Triton Deployment

This project demonstrates a simplified deployment of a Triton model on Amazon SageMaker using basic PyTorch tensor operations. It provides a streamlined example of how to deploy PyTorch models with Triton Inference Server.

## Overview

The project uses a custom Docker container to deploy a simplified PyTorch model on SageMaker, leveraging Triton Inference Server for deployment. The model performs basic tensor operations on audio input and returns statistical information.

## Prerequisites

- An AWS account with SageMaker access
- Docker installed on your local machine
- Python 3.10+
- AWS CLI configured with appropriate permissions

## Dependencies

This simplified implementation requires the following Python packages:
- torch==2.3.0
- torchaudio==2.3.0
- transformers==4.41.0
- soundfile
- librosa
- tritonclient[grpc]
- fastapi
- uvicorn
- numpy

## Model Implementation

The simplified model implementation (in `model_repo_whisper_trtllm/whisper/1/model.py`) performs basic PyTorch tensor operations:

1. Takes audio waveforms as input
2. Calculates statistical metrics (mean, standard deviation, energy)
3. Performs normalization and feature extraction
4. Returns these statistics as text output

Unlike the original Whisper ASR implementation, this version does not require:
- TensorRT-LLM model compilation
- Pretrained Whisper models
- Complex model training or fine-tuning

## Model Configuration

The model configuration (in `model_repo_whisper_trtllm/whisper/config.pbtxt`) has been simplified to:

- Remove dependencies on external model paths
- Simplify parameters
- Maintain the same input/output interface for compatibility with the API

## Deployment

The deployment process follows the same steps as the original implementation:

1. Build and push the Docker image to Amazon ECR
2. Create a SageMaker model
3. Create an endpoint configuration
4. Deploy the model to an endpoint

For detailed deployment steps, refer to the `deploy_and_test.ipynb` notebook.

## API Usage

The API interface remains unchanged from the original implementation, ensuring compatibility with existing client code. The key difference is that instead of transcribing speech, the endpoint now returns statistical information about the audio input.

## Docker Image

The Docker image is based on the NVIDIA Triton Server image and includes PyTorch and other necessary dependencies.

## Customization

You can customize the model implementation by modifying:
- The tensor operations in `model.py`
- The model configuration in `config.pbtxt`
- The Docker image configuration in `Dockerfile.server`

## Cleanup

To avoid incurring unnecessary charges, remember to delete the SageMaker endpoint, endpoint configuration, and model when they are no longer needed:

```python
sess.delete_endpoint(endpoint_name)
sess.delete_endpoint_config(endpoint_name)
sess.delete_model(model.name)
