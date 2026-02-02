import huggingface_hub
import warnings

# Monkey patch for sentence-transformers compatibility
if not hasattr(huggingface_hub, 'cached_download'):
    warnings.warn("Patching huggingface_hub.cached_download for compatibility")
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
