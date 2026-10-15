import os
import time
from pathlib import Path

import numpy as np
from llama_cpp import Llama, LogitsProcessorList

CONTEXT = 4096
MAX_FREE_TOKENS = 512
SEED = 42


def find_model():
    if os.getenv("MODEL_PATH"):
        path = Path(os.environ["MODEL_PATH"]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    candidates = sorted(
        p for p in Path("models").rglob("*.gguf")
        if "-0000" not in p.name or "-00001-of-" in p.name
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected one GGUF in models/, or set MODEL_PATH"
        )
    return candidates[0]


def chat_prompt(model_path, system, prompt):
    name = model_path.name.lower()

    if "qwen3.5" in name or "qwen35" in name:
        head = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            if system else ""
        )
        return (
            head
            + f"<|im_start|>user\n{prompt}<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    if "glm-4.7" in name or "glm47" in name:
        return (
            "[gMASK]<sop>"
            + (f"<|system|>{system}" if system else "")
            + f"<|user|>{prompt}<|assistant|></think>"
        )

    raise ValueError(
        "Unrecognized model: expected GLM-4.7 or Qwen3.5 filename"
    )


class Runner:
    def __init__(self):
        self.model_path = find_model()
        print("Loading:", self.model_path, flush=True)

        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=CONTEXT,
            n_gpu_layers=-1,
            n_batch=128,
            seed=SEED,
            verbose=False,
        )

        self.stop_ids = {int(self.llm.token_eos())}

        for marker in (
            "<|im_end|>",
            "<|endoftext|>",
            "<|user|>",
            "<|observation|>",
        ):
            ids = self.llm.tokenize(
                marker.encode(),
                add_bos=False,
                special=True,
            )
            if len(ids) == 1:
                self.stop_ids.add(int(ids[0]))

    def close(self):
        self.llm.close()

    def _run(self, prompt_ids, forced_ids=()):
        count = len(forced_ids)

        if len(prompt_ids) + count + MAX_FREE_TOKENS >= CONTEXT:
            raise ValueError(
                "Prompt + prefix too long for 4096-token context"
            )

        def force(input_ids, scores):
            step = len(input_ids) - len(prompt_ids)

            if 0 <= step < count:
                result = np.full_like(scores, -np.inf)
                result[forced_ids[step]] = 0.0
                return result

            return scores

        self.llm.reset()
        self.llm.set_seed(SEED)

        kwargs = dict(
            temp=0.0,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.0,
            min_p=0.0,
            reset=True,
        )

        if count:
            kwargs["logits_processor"] = LogitsProcessorList(
                [force]
            )

        tokens = []
        reason = "exhausted"
        start = time.perf_counter()

        stream = self.llm.generate(prompt_ids, **kwargs)

        try:
            for token in stream:
                token = int(token)

                if len(tokens) >= count and token in self.stop_ids:
                    reason = "eos"
                    break

                tokens.append(token)

                if (
                    len(tokens) <= count
                    and tokens[-1] != forced_ids[len(tokens) - 1]
                ):
                    raise RuntimeError("Forced token mismatch")

                if len(tokens) >= count + MAX_FREE_TOKENS:
                    reason = "max_tokens"
                    break
        finally:
            stream.close()

        seconds = time.perf_counter() - start

        verified = (
            tokens[:count] == list(forced_ids)
            if count else None
        )

        if count and not verified:
            raise RuntimeError("Forced prefix not fully emitted")

        full_bytes = self.llm.detokenize(
            tokens, special=False
        )

        if count:
            prefix_bytes = self.llm.detokenize(
                forced_ids, special=False
            )

            if not full_bytes.startswith(prefix_bytes):
                raise RuntimeError(
                    "Decoded prefix boundary mismatch"
                )

            free_bytes = full_bytes[len(prefix_bytes):]
        else:
            free_bytes = full_bytes

        return {
            "response": full_bytes.decode(
                "utf-8", errors="replace"
            ),
            "free_response": free_bytes.decode(
                "utf-8", errors="replace"
            ),
            "tokens": len(tokens),
            "free_tokens": len(tokens) - count,
            "forced_tokens": count,
            "verified": verified,
            "seconds": round(seconds, 3),
            "tokens_per_second": round(
                len(tokens) / max(seconds, 1e-9), 2
            ),
            "stop_reason": reason,
        }

    def compare(self, system, prompt, prefix):
        if not prompt or not prefix:
            raise ValueError(
                "prompt and forced_prefix must be non-empty"
            )

        prepared = chat_prompt(
            self.model_path, system, prompt
        )

        prompt_ids = self.llm.tokenize(
            prepared.encode("utf-8"),
            add_bos=False,
            special=True,
        )

        forced_ids = self.llm.tokenize(
            prefix.encode("utf-8"),
            add_bos=False,
            special=False,
        )

        if (
            not forced_ids
            or any(t in self.stop_ids for t in forced_ids)
        ):
            raise ValueError("Invalid forced prefix")

        normal = self._run(prompt_ids)
        forced = self._run(prompt_ids, forced_ids)

        return {
            "normal": normal,
            "forced": forced,
            "normal_response": normal["response"],
            "forced_full_response": forced["response"],
            "forced_free_response": forced["free_response"],
            "forced_token_ids_verified": forced["verified"],
        }
