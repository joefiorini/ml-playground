# Ripped from https://github.com/oobabooga/text-generation-webui/blob/c6cae106e763e66bc99ddac0d350f4ab2810fa12/modules/exllama.py

import sys
from pathlib import Path
from exllama.generator import ExLlamaGenerator
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer


class Logger:
    def warning(self, str):
        print(f"WARNING: {str}")

    def error(self, str):
        print(f"ERROR: {str}")

    def info(self, str):
        print(f"INFO: {str}")

    def debug(self, str):
        print(f"DEBUG: {str}")


logger = Logger()


class ExllamaModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(
        cls, path_to_model, max_seq_len=512, compress_pos_emb=1.0, gpu_split=None
    ):
        path_to_model = Path(path_to_model)
        tokenizer_model_path = path_to_model / "tokenizer.model"
        model_config_path = path_to_model / "config.json"

        # Find the model checkpoint
        model_path = None
        for ext in [".safetensors", ".pt", ".bin"]:
            found = list(path_to_model.glob(f"*{ext}"))
            if len(found) > 0:
                if len(found) > 1:
                    logger.warning(
                        f"More than one {ext} model has been found. The last one will be selected. It could be wrong."
                    )

                model_path = found[-1]
                break

        config = ExLlamaConfig(str(model_config_path))
        config.model_path = str(model_path)
        config.max_seq_len = max_seq_len
        config.compress_pos_emb = compress_pos_emb
        if gpu_split is not None:
            config.set_auto_map(gpu_split)
            config.gpu_peer_fix = True

        model = ExLlama(config)
        tokenizer = ExLlamaTokenizer(str(tokenizer_model_path))
        cache = ExLlamaCache(model)
        generator = ExLlamaGenerator(model, tokenizer, cache)

        result = cls()
        result.config = config
        result.model = model
        result.cache = cache
        result.tokenizer = tokenizer
        result.generator = generator
        return result, result

    def generate_with_streaming(self, prompt, state):
        self.generator.settings.temperature = state["temperature"]
        self.generator.settings.top_p = state["top_p"]
        self.generator.settings.top_k = state["top_k"]
        self.generator.settings.typical = state["typical_p"]
        self.generator.settings.token_repetition_penalty_max = state[
            "repetition_penalty"
        ]
        if state["ban_eos_token"]:
            self.generator.disallow_tokens([self.tokenizer.eos_token_id])
        else:
            self.generator.disallow_tokens(None)

        self.generator.end_beam_search()
        ids = self.generator.tokenizer.encode(prompt)
        self.generator.gen_begin_reuse(ids)
        initial_len = self.generator.sequence[0].shape[0]
        has_leading_space = False
        for i in range(state["max_new_tokens"]):
            token = self.generator.gen_single_token()
            if i == 0 and self.generator.tokenizer.tokenizer.IdToPiece(
                int(token)
            ).startswith("▁"):
                has_leading_space = True

            decoded_text = self.generator.tokenizer.decode(
                self.generator.sequence[0][initial_len:]
            )
            if has_leading_space:
                decoded_text = " " + decoded_text

            yield decoded_text
            if (
                token.item() == self.generator.tokenizer.eos_token_id
                or shared.stop_everything
            ):
                break

    def generate(self, prompt, state):
        output = ""
        for output in self.generate_with_streaming(prompt, state):
            pass

        return output

    def encode(self, string, **kwargs):
        return self.tokenizer.encode(string)
