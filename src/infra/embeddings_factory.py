from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings
import torch
import logging

logger = logging.getLogger(__name__)

SUPPORTED_DEVICES = ["cpu", "cuda", "mps", "npu"]

def _select_device(preferred_device: str, logger_instance: logging.Logger) -> str:
    """
    Selects the appropriate computation device based on preference and availability.
    Falls back to 'cpu' if the preferred device is not available or not recognized.
    """
    actual_device = preferred_device.lower()

    if actual_device not in SUPPORTED_DEVICES:
        logger_instance.warning(
            f"Preferred device '{preferred_device}' is not recognized or supported ({SUPPORTED_DEVICES}). Defaulting to 'cpu'."
        )
        return "cpu"

    if actual_device == "cuda":
        try:
            if torch.cuda.is_available():
                logger_instance.info("CUDA is available. Selecting 'cuda'.")
                return "cuda"
            else:
                logger_instance.warning("CUDA preferred but not available. Falling back to 'cpu'.")
                return "cpu"
        except Exception as e:
            logger_instance.warning(f"Error checking CUDA availability. Falling back to 'cpu'. Exception: {e}")
            return "cpu"

    elif actual_device == "mps":
        try:
            # Check if MPS is available and built
            # hasattr check is good for environments where mps might not even be an attribute
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                logger_instance.info("MPS is available. Selecting 'mps'.")
                return "mps"
            else:
                logger_instance.warning("MPS preferred but not available/built. Falling back to 'cpu'.")
                return "cpu"
        except Exception as e:
            logger_instance.warning(f"Error checking MPS availability. Falling back to 'cpu'. Exception: {e}")
            return "cpu"
            
    elif actual_device == "npu":
        # Assuming NPU usage might have its own checks or is expected to work if specified.
        # For now, if "npu" is chosen and no specific checks fail, we proceed.
        # Future: Add NPU availability checks if Pytorch provides them similar to CUDA/MPS.
        logger_instance.info(f"NPU selected as per configuration. Note: NPU availability check not yet implemented in this factory.")
        return "npu"


    # Default to 'cpu' if it's explicitly chosen or as a final fallback (though covered by initial check)
    logger_instance.info(f"Selecting 'cpu' device.")
    return "cpu"

class EmbeddingFactory:
    @staticmethod
    def get_model(name: str, preferred_device_config: str, show_progress: bool = True):
        """
        Returns an instance of HuggingFaceEmbeddings, attempting to use the preferred device
        (e.g., 'cuda', 'mps') with robust fallback to 'cpu' if the preferred device is
        unavailable, not recognized, or if an error occurs during device checking.

        The initial `import torch` must succeed for GPU checks to be performed.
        If `import torch` itself fails (e.g., due to missing system libraries like libcusparseLt.so.0),
        this method will not be reached or will fail when attempting to use torch.

        Args:
            name (str): The name of the HuggingFace model.
            preferred_device_config (str): The desired device ('cuda', 'mps', 'cpu', 'npu').
                                           Case-insensitive.
            show_progress (bool): Whether to show a progress bar during model download.

        Returns:
            HFEmbeddings: An instance of the embedding model configured for the selected device.
        """
        
        final_device = _select_device(preferred_device_config, logger)
        
        logger.info(f"EmbeddingFactory: Loading model '{name}' using final device: '{final_device}'.")
        
        return HFEmbeddings( 
            model_name=name,
            model_kwargs={'device': final_device},
            show_progress=show_progress
        )
