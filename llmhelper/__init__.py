"""local_llm_helper — offload second-priority work to a local LLM.

The point of this package is to keep bulk bytes out of an expensive model's
context: a local model reads the raw material and returns a compact answer.
"""

__version__ = "0.1.0"
