import numpy as np
import triton_python_backend_utils as pb_utils
import json
import torch
import re
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger('whisper_model')

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "TRANSCRIPTS")
        # Convert Triton types to numpy types
        self.out0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get parameters from model config if they exist
        self.parameters = {}
        if 'parameters' in self.model_config:
            parameters = self.model_config['parameters']
            for key, value in parameters.items():
                self.parameters[key] = value["string_value"]

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests.
        logger.info(f"Received inference request with {len(requests)} requests")
        responses = []
        
        for request in requests:
            # Process inputs similar to SageMaker model
            # Get TEXT_PREFIX if available
            text_prefix = None
            try:
                text_prefix_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT_PREFIX")
                logger.info(f"TEXT_PREFIX tensor found: {text_prefix_tensor is not None}")
                if text_prefix_tensor is not None:
                    text_prefix_np = text_prefix_tensor.as_numpy()
                    logger.info(f"TEXT_PREFIX shape: {text_prefix_np.shape}")
                    text_prefix = text_prefix_np.tolist()[0][0]
                    if isinstance(text_prefix, bytes):
                        text_prefix = text_prefix.decode('utf-8')
                    logger.info(f"Processed TEXT_PREFIX: '{text_prefix}'")
            except Exception as e:
                logger.error(f"Error processing TEXT_PREFIX: {e}")
                text_prefix = ""

            # Get WAV audio data
            try:
                wav_tensor = pb_utils.get_input_tensor_by_name(request, "WAV")
                logger.info(f"WAV tensor found: {wav_tensor is not None}")
                if wav_tensor is None:
                    # Fall back to input for compatibility
                    wav_tensor = pb_utils.get_input_tensor_by_name(request, "input")
                    logger.info(f"Using fallback input tensor: {wav_tensor is not None}")
                wav_data = wav_tensor.as_numpy()
                logger.info(f"WAV data shape: {wav_data.shape}")
            except Exception as e:
                logger.error(f"Error processing WAV input: {e}")
                wav_data = np.zeros((1, 16000), dtype=np.float32)  # Default 1 second of silence

            # Get REPETITION_PENALTY if available
            repetition_penalty = 1.0
            try:
                rep_penalty_tensor = pb_utils.get_input_tensor_by_name(request, "REPETITION_PENALTY")
                logger.info(f"REPETITION_PENALTY tensor found: {rep_penalty_tensor is not None}")
                if rep_penalty_tensor is not None:
                    rep_penalty_np = rep_penalty_tensor.as_numpy()
                    logger.info(f"REPETITION_PENALTY shape: {rep_penalty_np.shape}")
                    # Handle both [val] and [[val]] shapes
                    if len(rep_penalty_np.shape) > 1:
                        repetition_penalty = float(rep_penalty_np[0][0])
                    else:
                        repetition_penalty = float(rep_penalty_np[0])
                    logger.info(f"Processed REPETITION_PENALTY: {repetition_penalty}")
            except Exception as e:
                logger.error(f"Error processing REPETITION_PENALTY: {e}")

            # Convert to PyTorch tensor and process
            wav = torch.from_numpy(wav_data)
            if len(wav.shape) == 1:  # If it's a single array, make it batch size 1
                wav = wav.unsqueeze(0)
            wav = wav.to(self.device)
            
            # Simple processing for demonstration
            # In a real implementation, this would use the WhisperTRTLLM model
            # Here we're just generating a transcript based on the input data
            processed = wav.mean(dim=1).tolist()
            
            # Create a mock transcript based on the input data
            mock_transcript = f"Transcription (TEXT_PREFIX: '{text_prefix}', "
            mock_transcript += f"WAV shape: {wav_data.shape}, "
            mock_transcript += f"rep_penalty: {repetition_penalty})"
            
            # Create output tensor with transcript
            output_tensor = pb_utils.Tensor("TRANSCRIPTS", 
                                          np.array([mock_transcript], dtype=self.out0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
        
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
