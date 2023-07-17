import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from threading import Lock
from typing import NamedTuple
import torch
from exllama.tokenizer import ExLlamaTokenizer
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.lora import ExLlamaLora
from exllama.generator import ExLlamaGenerator
from itertools import chain
from . import model_init


class ModelInitConfig(NamedTuple):
    max_tokens: int = 64
    stopping_strings: list = []
    temperature: float = None
    top_k: int = None
    top_p: float = None
    typical_p: float = None
    repetition_penalty: float = None
    num_beams: int = None
    seed: int = -1


def build_settings(data: ModelInitConfig):
    global generator

    return {
        "max_tokens": int(data.get("max_new_tokens", data.get("max_length", 64))),
        "stopping_strings": data.get("stopping_strings", []),
        "temperature": float(data.get("temperature", generator.settings.temperature)),
        "top_k": int(data.get("top_k", generator.settings.top_k)),
        "top_p": float(data.get("top_p", generator.settings.top_p)),
        "typical_p": float(data.get("typical_p", generator.settings.typical)),
        "repetition_penalty": float(
            data.get(
                "repetition_penalty",
                data.get("rep_pen", generator.settings.token_repetition_penalty_max),
            )
        ),
        "num_beams": int(data.get("num_beams", generator.settings.beams)),
        "seed": int(data.get("seed", -1)),
    }


def update_settings(settings):
    global generator, args

    generator.settings.temperature = settings["temperature"]
    generator.settings.top_k = settings["top_k"]
    generator.settings.top_p = settings["top_p"]
    generator.settings.typical = settings["typical_p"]
    generator.settings.min_p

    generator.settings.token_repetition_penalty_max = settings["repetition_penalty"]
    generator.settings.token_repetition_penalty_sustain
    generator.settings.token_repetition_penalty_decay

    generator.settings.beams = settings["num_beams"]
    generator.settings.beam_length = 5


config = model_init.make_config(args)

print(f" -- Loading model...")
model = ExLlama(config)
cache = ExLlamaCache(model)

print(f" -- Loading tokenizer...")
tokenizer = ExLlamaTokenizer(tokenizer)

model_init.print_stats(model)

generator = ExLlamaGenerator(model, tokenizer, cache)


def prepare_context(context):
    global args

    if debug:
        print(f'Context: "{context}"')

    if context is None or len(context) == 0:
        generator.gen_begin_empty()
    else:
        tokenized = tokenizer.encode(context)
        generator.gen_begin_reuse(tokenized)


def check_hold(text, stopping_strings):
    def overlaps_end(stop):
        if len(stop) == 0:
            return False
        if text.endswith(stop):
            return True
        return overlaps_end(stop[0:-1])

    return any(map(overlaps_end, stopping_strings))


def find_first_stop_index(text, stopping_strings):
    text_length = len(text)

    def stopindex(stop):
        if stop in text:
            return text.index(stop)
        else:
            return text_length

    return min(chain(map(stopindex, stopping_strings), [text_length]))


def check_stop(text, stopping_strings):
    return any(map(lambda stop: stop in text, stopping_strings))


generate_lock = Lock()

# Seems to be generated while creating emoji.
# Transforms into emoji after ~4 tokens are generated 🤷.
# Sentencepiece magic I guess.
UNFINISHED_SEQUENCE = b"\xef\xbf\xbd".decode("utf-8")


def ambiguous_depth(generator):
    global UNFINISHED_SEQUENCE

    full_sequence = tokenizer.decode(generator.sequence_actual[0, :])

    if full_sequence.endswith(UNFINISHED_SEQUENCE):
        return full_sequence.index(UNFINISHED_SEQUENCE) - len(full_sequence)

    last = tokenizer.decode(generator.sequence_actual[0, -1:])
    second_last = tokenizer.decode(generator.sequence_actual[0, -2:-1])
    # This is apparently ambiguous.
    if second_last.endswith("\n"):
        return -len(last)

    return None


def _generate(context, settings):
    global tokenizer, generator, generate_lock, args

    stopping_strings = settings["stopping_strings"]
    seed = settings["seed"]

    if seed != -1:
        # this doesn't work
        torch.manual_seed(seed)

    previous_length = 0

    with torch.no_grad(), generate_lock:
        update_settings(settings)

        prepare_context(context)
        context_length = generator.sequence_actual.shape[-1]
        max_tokens = length
        max_generated = min(settings["max_tokens"], max_tokens - context_length)
        generator.begin_beam_search()

        for i in range(1, max_generated):
            token = generator.beam_search()

            generated = tokenizer.decode(generator.sequence_actual[0, -i:])

            if token.item() == tokenizer.eos_token_id:
                break

            if check_stop(generated, stopping_strings):
                break

            new_text = generated[previous_length : ambiguous_depth(generator)]

            if len(new_text) > 0 and not check_hold(generated, stopping_strings):
                yield new_text
                previous_length += len(new_text)

        generator.end_beam_search()

        stop_index = find_first_stop_index(generated, stopping_strings)
        end_offset = ambiguous_depth(generator) or 0
        unjustly_held_text = generated[previous_length : stop_index + end_offset]
        if len(unjustly_held_text) > 0:
            yield unjustly_held_text


def token_count(text):
    return len(tokenizer.encode(text).shape[-1])


def generate(data, settings):
    fragments = _generate(
        data.get("prompt"),
        settings,
    )

    response = "".join(fragments)

    if debug:
        print(f'Response: "{response}"')

    return {"results": [{"text": response}]}
