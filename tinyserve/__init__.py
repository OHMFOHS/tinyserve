from tinyserve.sampling_params import SamplingParams


def __getattr__(name):  # keep `import tinyserve.ops` light: the engine pulls in transformers
    if name in ("LLM", "LLMEngine"):
        from tinyserve import engine

        return getattr(engine, name)
    raise AttributeError(name)


__all__ = ["LLM", "LLMEngine", "SamplingParams"]
