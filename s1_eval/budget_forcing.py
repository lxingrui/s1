"""Budget Forcing 推理：复现 s1 原版「强塞 Wait」机制。

流程（与论文 / README 的 vLLM 示例一致）：

1. 用 chat template 构造 prompt，并追加思考段起始标记（``<|im_start|>think``）；
2. 让模型生成思考，遇到「思考结束」标记就停下；
3. 若还有思考预算，则把结束标记替换成 ``ignore_str``（默认 "Wait"），
   强迫模型继续反思，重复 ``num_ignore`` 次；
4. 追加 ``<|im_start|>answer``，生成最终答案。

vLLM 后端支持批量；transformers 后端为逐条（用于没有 vLLM 的环境）。
两者产出的结果结构一致。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["BudgetForcingPipeline"]


class BudgetForcingPipeline:
    def __init__(self, config: Any, log: Callable[[str], None] | None = None) -> None:
        self.cfg = config
        self.model_cfg: dict[str, Any] = dict(config.get("model", {}) or {})
        self.bf: dict[str, Any] = dict(config.get("budget_forcing", {}) or {})
        self.prompt_cfg: dict[str, Any] = dict(
            (config.get("eval", {}) or {}).get("prompt", {}) or {}
        )
        self.log = log or print

        self.backend: str | None = None
        self._llm = None  # vLLM
        self._hf_model = None  # transformers
        self._hf_tok = None
        self.tokenizer = None

    # ================================================================== #
    # 加载
    # ================================================================== #
    def load(self) -> None:
        backend = str(self.model_cfg.get("backend", "auto")).lower()
        if backend in ("auto", "vllm"):
            try:
                self._load_vllm()
                self.backend = "vllm"
                self.log("[bf] 已用 vLLM 后端加载模型。")
                return
            except Exception as exc:
                if backend == "vllm":
                    raise
                self.log(f"[bf] vLLM 不可用（{exc}），回退到 transformers 后端。")
        self._load_hf()
        self.backend = "transformers"
        self.log("[bf] 已用 transformers 后端加载模型。")

    def _load_vllm(self) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM  # 延迟导入，未安装时报错可被捕获

        tok_name = self.model_cfg.get("tokenizer") or self.model_cfg["name"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_name,
            trust_remote_code=bool(self.model_cfg.get("trust_remote_code", True)),
            cache_dir=self.model_cfg.get("cache_dir") or None,
        )
        kwargs: dict[str, Any] = {
            "model": self.model_cfg["name"],
            "tokenizer": tok_name,
            "tensor_parallel_size": int(self.model_cfg.get("tensor_parallel_size", 1)),
            "max_model_len": int(self.model_cfg.get("max_model_len", 32768)),
            "gpu_memory_utilization": float(
                self.model_cfg.get("gpu_memory_utilization", 0.9)
            ),
            "trust_remote_code": bool(self.model_cfg.get("trust_remote_code", True)),
        }
        dtype = self.model_cfg.get("dtype", "auto")
        if dtype and dtype != "auto":
            kwargs["dtype"] = dtype
        if self.model_cfg.get("cache_dir"):
            kwargs["download_dir"] = self.model_cfg["cache_dir"]
        self._llm = LLM(**kwargs)

    def _load_hf(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok_name = self.model_cfg.get("tokenizer") or self.model_cfg["name"]
        self._hf_tok = AutoTokenizer.from_pretrained(
            tok_name,
            trust_remote_code=bool(self.model_cfg.get("trust_remote_code", True)),
            cache_dir=self.model_cfg.get("cache_dir") or None,
        )
        if self._hf_tok.pad_token is None:
            self._hf_tok.pad_token = self._hf_tok.eos_token

        dtype_name = str(self.model_cfg.get("dtype", "auto"))
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype_name, "auto")

        # accelerate 缺失/版本不匹配时不要传 device_map，否则会直接报错
        try:
            import accelerate  # noqa: F401

            has_accel = True
        except Exception:  # noqa: BLE001
            has_accel = False

        common = {
            "trust_remote_code": bool(self.model_cfg.get("trust_remote_code", True)),
            "cache_dir": self.model_cfg.get("cache_dir") or None,
        }
        device_map = self.model_cfg.get("device_map", "auto")

        def _try_load(**extra):
            """兼容新旧 transformers 的 dtype 参数名。"""
            last_err: Exception | None = None
            for dtype_kw in ({"dtype": torch_dtype}, {"torch_dtype": torch_dtype}):
                try:
                    return AutoModelForCausalLM.from_pretrained(
                        self.model_cfg["name"], **dtype_kw, **extra
                    )
                except TypeError as exc:  # 参数名不被支持
                    last_err = exc
            raise last_err  # type: ignore[misc]

        model = None
        if has_accel and device_map:
            try:
                model = _try_load(device_map=device_map, **common)
            except ValueError as exc:
                if "accelerate" not in str(exc):
                    raise
                # accelerate 缺失或版本不满足要求 → 退回手工投放设备
                print("[bf] device_map 需要可用的 accelerate，改为手动放置。")
        if model is None:
            model = _try_load(**common)
            target = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(target)

        self._hf_model = model
        self._hf_model.eval()
        self.tokenizer = self._hf_tok

    # ================================================================== #
    # 提示词构造
    # ================================================================== #
    def build_prompt(self, question: str) -> str:
        system = self.prompt_cfg.get("system", "") or ""
        query = (self.prompt_cfg.get("query_template") or "{question}").format(
            question=question
        )
        if self.prompt_cfg.get("apply_chat_template", True):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{system}\n\n{query}\n"

    # ================================================================== #
    # token 工具
    # ================================================================== #
    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def _stop_ids(self) -> list[int]:
        ids: list[int] = []
        for s in self.bf.get("stop_strings", ["<|im_start|>", "<|im_end|>"]):
            for tok_id in self._encode(s):
                if tok_id not in ids:
                    ids.append(tok_id)
        return ids

    @staticmethod
    def _strip_trailing_stop(ids: list[int], stop_ids) -> list[int]:
        stop = set(stop_ids)
        while ids and ids[-1] in stop:
            ids.pop()
        return ids

    # ================================================================== #
    # 生成
    # ================================================================== #
    def generate(self, questions: list[str]) -> list[dict[str, Any]]:
        if self.backend == "vllm":
            return self._generate_vllm(questions)
        return [self._generate_hf_one(q) for q in questions]

    # ------------------------- vLLM（批量）---------------------------- #
    def _vllm_generate(self, token_id_batches: list[list[int]], sampling_params):
        """调用 vLLM 生成，兼容 v0 / v1 (Engine) 两种签名。

        vLLM v1 起，``LLM.generate()`` 不再接受 ``prompt_token_ids=`` 关键字参数，
        必须把 token ids 包装进 ``prompts=[{"prompt_token_ids": [...]}, ...]``。
        """
        prompts = [{"prompt_token_ids": ids} for ids in token_id_batches]
        try:
            return self._llm.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        except TypeError:
            # 极老版本 vLLM 只认 prompt_token_ids 关键字参数
            return self._llm.generate(
                prompt_token_ids=token_id_batches,
                sampling_params=sampling_params,
            )

    def _generate_vllm(self, questions: list[str]) -> list[dict[str, Any]]:
        from vllm import SamplingParams

        bf = self.bf
        thinking_start = bf.get("thinking_start", "<|im_start|>think")
        thinking_end = bf.get("thinking_end", "<|im_start|>answer")
        ignore_str = bf.get("ignore_str", "Wait")
        num_ignore = int(bf.get("num_ignore", 1))
        budget = int(bf.get("max_tokens_thinking", 32000))
        stop_ids = self._stop_ids()
        temperature = float(bf.get("temperature", 0.0))
        top_p = float(bf.get("top_p", 1.0))

        base_prompts = [self.build_prompt(q) + thinking_start for q in questions]
        prompt_ids = [self._encode(p) for p in base_prompts]

        n_chunks = num_ignore + 1
        accum: list[list[int]] = [[] for _ in prompt_ids]
        active_prompts = [list(pid) for pid in prompt_ids]
        active_idx = list(range(len(prompt_ids)))

        for chunk in range(n_chunks):
            if not active_prompts:
                break
            sp = SamplingParams(
                max_tokens=budget,
                min_tokens=1,
                stop_token_ids=stop_ids,
                skip_special_tokens=False,
                temperature=temperature,
                top_p=top_p,
            )
            outs = self._vllm_generate(active_prompts, sp)
            next_prompts: list[list[int]] = []
            next_idx: list[int] = []
            for j, out in enumerate(outs):
                idx = active_idx[j]
                gen = list(out.outputs[0].token_ids)
                finish = out.outputs[0].finish_reason
                if finish == "length" or chunk == n_chunks - 1:
                    accum[idx].extend(gen)
                else:
                    gen = self._strip_trailing_stop(gen, stop_ids)
                    accum[idx].extend(gen)
                    ignore_ids = self._encode(ignore_str)
                    accum[idx].extend(ignore_ids)
                    next_prompts.append(active_prompts[j] + gen + ignore_ids)
                    next_idx.append(idx)
            active_prompts, active_idx = next_prompts, next_idx

        # ---------- 最终作答 ----------
        final_prompts: list[list[int]] = []
        for i in range(len(prompt_ids)):
            seq = list(prompt_ids[i]) + accum[i]
            if bf.get("append_thinking_end", True):
                seq = seq + self._encode(thinking_end)
            prefix = bf.get("final_answer_prefix", "") or ""
            if prefix:
                seq = seq + self._encode(prefix)
            final_prompts.append(seq)

        sp_final = SamplingParams(
            max_tokens=int(bf.get("max_tokens_answer", 32768)),
            min_tokens=1,
            stop_token_ids=[self._encode("<|im_end|>")[-1]]
            if self._encode("<|im_end|>")
            else stop_ids,
            skip_special_tokens=False,
            temperature=temperature,
            top_p=top_p,
        )
        outs_final = self._vllm_generate(final_prompts, sp_final)

        results: list[dict[str, Any]] = []
        for i in range(len(questions)):
            answer_ids = list(outs_final[i].outputs[0].token_ids)
            answer_ids = self._strip_trailing_stop(
                answer_ids, self._encode("<|im_end|>")
            )
            thinking_text = self._decode(accum[i])
            answer_text = self._decode(answer_ids)
            results.append(
                {
                    "question": questions[i],
                    "thinking": thinking_text,
                    "answer": answer_text,
                    "response": answer_text,
                    "num_ignores_used": num_ignore,
                    "backend": "vllm",
                }
            )
        return results

    # --------------------- transformers（逐条）----------------------- #
    def _generate_hf_one(self, question: str) -> dict[str, Any]:

        bf = self.bf
        thinking_start = bf.get("thinking_start", "<|im_start|>think")
        thinking_end = bf.get("thinking_end", "<|im_start|>answer")
        ignore_str = bf.get("ignore_str", "Wait")
        num_ignore = int(bf.get("num_ignore", 1))
        budget = int(bf.get("max_tokens_thinking", 32000))
        stop_ids = set(self._stop_ids())
        temperature = float(bf.get("temperature", 0.0))

        prompt = self.build_prompt(question) + thinking_start

        def _gen(text: str, max_new: int, do_stop: bool) -> tuple:
            input_ids = self._hf_tok(text, return_tensors="pt").input_ids.to(
                self._hf_model.device
            )
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new,
                "min_new_tokens": 1,
                "pad_token_id": self._hf_tok.pad_token_id,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
            out = self._hf_model.generate(input_ids, **gen_kwargs)
            new_ids = out[0][input_ids.shape[-1] :].tolist()
            finish_length = len(new_ids) >= max_new
            if do_stop and new_ids and new_ids[-1] in stop_ids:
                new_ids = self._strip_trailing_stop(new_ids, stop_ids)
            return new_ids, finish_length

        thinking_ids: list[int] = []
        n_chunks = num_ignore + 1
        for chunk in range(n_chunks):
            new_ids, hit_len = _gen(prompt, budget, do_stop=True)
            thinking_ids.extend(new_ids)
            prompt = prompt + self._decode(new_ids, skip_special_tokens=False)
            if hit_len or chunk == n_chunks - 1:
                break
            prompt = prompt + ignore_str
            thinking_ids.extend(self._encode(ignore_str))

        if bf.get("append_thinking_end", True):
            prompt = prompt + thinking_end
        prefix = bf.get("final_answer_prefix", "") or ""
        if prefix:
            prompt = prompt + prefix

        answer_ids, _ = _gen(
            prompt, int(bf.get("max_tokens_answer", 32768)), do_stop=False
        )
        thinking_text = self._decode(thinking_ids)
        answer_text = self._decode(answer_ids)
        return {
            "question": question,
            "thinking": thinking_text,
            "answer": answer_text,
            "response": answer_text,
            "num_ignores_used": num_ignore,
            "backend": "transformers",
        }
