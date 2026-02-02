from llama_cpp import Llama
import os

class MistralLLM:
    def __init__(self, model_path, n_ctx=4096, n_gpu_layers=0):
        """
        Initialize Mistral LLM.
        Args:
            model_path: Path to the GGUF model file.
            n_ctx: Context window size.
            n_gpu_layers: Number of layers to offload to GPU. Set to -1 for all, 0 for CPU.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading LLM from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        print("LLM loaded.")

    def generate_response(self, prompt, max_tokens=256, stop=["USER:", "\n\n"]):
        """
        Generate response from the LLM.
        """
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=stop,
            echo=False
        )
        return output['choices'][0]['text'].strip()

    def stream_response(self, prompt, max_tokens=256, stop=["USER:", "\n\n"]):
        """
        Stream response from the LLM.
        Yields chunks of text.
        """
        stream = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=stop,
            stream=True,
            echo=False
        )
        for output in stream:
            token = output['choices'][0]['text']
            yield token
