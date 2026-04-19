"""
Created on May 28, 2025

@author: Jiale Lao

LOTUS system runner implementation based on generic_runner for movie use case.
"""

import os
from overrides import override
import pandas as pd
import time
from typing import List
import lotus
from lotus.models import LM
import litellm
import re
from PIL import ImageFile

from runner.generic_runner import GenericRunner, GenericQueryMetric

# Allow loading of truncated images (some source images may be incomplete)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Opt-in HTTP-level tracing for diagnosing "no requests reach vLLM" hangs.
# Set LOTUS_DEBUG=1 in the environment to enable.
if os.environ.get("LOTUS_DEBUG"):
    litellm.set_verbose = True
    os.environ.setdefault("LITELLM_LOG", "DEBUG")


# ---------------------------------------------------------------------------
# Harness alignment patches
# ---------------------------------------------------------------------------
# The LLMSQL harness (proxy_model.py) sends a specific request shape to vLLM:
#   - system prompt: short "True/False" data-analyst instruction
#   - user content: [text, then image_url blocks with file:// URLs]
#   - response parser: substring "true" (case-insensitive) → positive
# Lotus's defaults differ on all three, which moves F1 even when the
# *semantics* of the query are identical. These patches override the three
# hook points in lotus 1.1.3 so lotus's sem_filter emits a byte-for-byte
# equivalent request to the one the harness sends in its no_opt baseline.
#
# Opt out with LOTUS_HARNESS_ALIGN=0.
# ---------------------------------------------------------------------------

_HARNESS_PATCHED = False

HARNESS_FILTER_SYSTEM_PROMPT = (
    "You are a helpful data analyst. You will receive data and a query. "
    "Answer ONLY 'True' or 'False'."
)


def _apply_harness_alignment_patches():
    """Monkey-patch lotus 1.1.3 internals to match the LLMSQL harness request shape.

    Idempotent — safe to call multiple times. Opt out via LOTUS_HARNESS_ALIGN=0.
    """
    global _HARNESS_PATCHED
    if _HARNESS_PATCHED:
        return
    if os.environ.get("LOTUS_HARNESS_ALIGN", "1").strip() in ("0", "false", "False"):
        print("[harness-align] disabled via LOTUS_HARNESS_ALIGN=0")
        _HARNESS_PATCHED = True
        return

    # (1) Images: return file:// URLs instead of base64 data URIs.
    try:
        from lotus.dtype_extensions import image as _image_mod

        _orig_get_image = _image_mod.ImageArray.get_image

        def _harness_get_image(self, idx, image_type="Image"):
            if image_type == "base64":
                raw = self._data[idx]
                if isinstance(raw, str):
                    url = raw.strip()
                    if not url.startswith(("file://", "http://", "https://", "data:")):
                        url = "file://" + os.path.abspath(url)
                    # Cache under the same key lotus uses so repeat lookups
                    # don't reenter this function.
                    self._cached_images[(idx, image_type)] = url
                    return url
            return _orig_get_image(self, idx, image_type)

        _image_mod.ImageArray.get_image = _harness_get_image
        print("[harness-align] patched ImageArray.get_image -> file:// URLs")
    except Exception as e:
        print(f"[harness-align] WARNING: could not patch ImageArray.get_image: {e}")

    # (2) filter_formatter: emit harness-style system + user content.
    try:
        from lotus.templates import task_instructions as _ti

        def _harness_filter_formatter(
            model, multimodal_data, user_instruction,
            examples_multimodal_data=None, examples_answer=None,
            cot_reasoning=None, strategy=None, reasoning_instructions="",
        ):
            # Build user content: [text block, then image_url blocks].
            if isinstance(multimodal_data, str):
                text = multimodal_data
                image_inputs = []
            else:
                text = multimodal_data.get("text", "") or ""
                image_data = multimodal_data.get("image", {}) or {}
                image_inputs = [
                    {"type": "image_url", "image_url": {"url": url}}
                    for url in image_data.values()
                ]
            # The harness puts the predicate text first, then the image
            # blocks, with no "Context:" prefix and no trailing instruction.
            user_text = (user_instruction or "").strip()
            if text:
                user_text = f"{user_text}\n{text}".strip() if user_text else text
            content = [{"type": "text", "text": user_text}] + image_inputs
            if not image_inputs:
                # Text-only: flatten to a plain string so litellm doesn't
                # wrap it as a multimodal content array.
                content = user_text
            return [
                {"role": "system", "content": HARNESS_FILTER_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]

        _ti.filter_formatter = _harness_filter_formatter
        print("[harness-align] patched task_instructions.filter_formatter")
    except Exception as e:
        print(f"[harness-align] WARNING: could not patch filter_formatter: {e}")

    # (3) filter_postprocess: substring "true" (case-insensitive) -> positive.
    try:
        from lotus.sem_ops import postprocessors as _pp

        _orig_filter_post = _pp.filter_postprocess

        def _harness_filter_postprocess(llm_answers, default=True, cot_reasoning=False):
            outputs = [("true" in (a or "").strip().lower()) for a in llm_answers]
            explanations = list(llm_answers)
            reasonings = [""] * len(llm_answers)
            try:
                from lotus.types import SemanticFilterPostprocessOutput
                return SemanticFilterPostprocessOutput(
                    raw_outputs=list(llm_answers),
                    outputs=outputs,
                    explanations=explanations,
                )
            except Exception:
                # Fall back to lotus's original parser if the return type
                # isn't importable — at least we still matched (1) and (2).
                return _orig_filter_post(llm_answers, default=default, cot_reasoning=cot_reasoning)

        _pp.filter_postprocess = _harness_filter_postprocess
        print("[harness-align] patched postprocessors.filter_postprocess (substring 'true')")
    except Exception as e:
        print(f"[harness-align] WARNING: could not patch filter_postprocess: {e}")

    _HARNESS_PATCHED = True

# Pricing rules: (text_input, audio_input, output) per 1M tokens
PRICING = {
    "llava-hf/llava-v1.6-34b-hf": {"text": 0.0, "audio": 0.0, "output": 0.0},
    "Qwen/Qwen3.5-122B-A10B-FP8": {"text": 0.0, "audio": 0.0, "output": 0.0},
    "Qwen/Qwen3-VL-30B-A3B-Instruct": {"text": 0.0, "audio": 0.0, "output": 0.0},
    "gpt-4o": {"text": 2.5, "audio": 2.5, "output": 10.0},
    "gpt-4o-mini": {"text": 0.15, "audio": 0.15, "output": 0.6},
    "gpt-4o-audio-preview": {
        "text": 2.5,
        "audio": 2.5,
        "output": 10.0,
    },
    "gpt-5": {"text": 1.25, "audio": 1.25, "output": 10.0},
    "gpt-5-mini": {"text": 0.25, "audio": 0.25, "output": 2.0},
    "gemini-2.0-flash": {
        "text": 0.15,
        "audio": 1.0,
        "output": 0.6,
    },
    "gemini-2.5-flash": {
        "text": 0.3,
        "audio": 1.0,
        "output": 2.5,
    },
    "gemini-2.5-flash-lite": {
        "text": 0.1,
        "audio": 0.3,
        "output": 0.4,
    },
    "gemini-2.5-pro": {
        "text": 1.25,
        "audio": 1.25,
        "output": 10.0,
    },
}


class GenericLotusRunner(GenericRunner):
    """GenericRunner for LOTUS system."""

    # thinking for gemini-2.5-pro can not be disabled, but always has issues
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
        policy="approximate",
        ranking="map",
    ):
        """
        Initialize LOTUS runner.

        Args:
            use_case: The use case to run
            model_name: LLM model to use
        """
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )
        self.policy = policy  # the policy can be "exact" or "approximate"
        self.ranking = ranking  # the ranking method could be "topk" or "map", for sem_topk and sem_map respectively

        # Initialize LOTUS

        # to avoid exceeding the reasoning_token of gemini models, default in LOTUS is 512
        self.max_tokens = 1024

        # Configure LM based on model_name
        self.lm = self._configure_lm()

        lotus.settings.configure(lm=self.lm)

        self._initialize_lotus_with_warmup()

    def _configure_lm(self) -> LM:
        """
        Configure Language Model based on self.model_name.

        Returns:
            Configured LM instance
        """
        base_config = {
            "max_batch_size": self.concurrent_llm_worker,
            "max_tokens": self.max_tokens,
        }

        model_lower = self.model_name.lower()

        if "gemini-2.5-pro" in model_lower or "gemini_2_5_pro" in model_lower:
            # gemini-2.5-pro: reasoning_effort="low", solve the "no content due to length" issue
            # but does not work when concurrent_llm_worker is 20, try using rate_limit to control
            return LM(
                self.model_name,
                rate_limit=2000,
                max_tokens=self.max_tokens,
                reasoning_effort="low",
            )
        # In _configure_lm(), add a branch for local/custom models:
        elif "qwen" in model_lower or "llava" in model_lower or "localhost" in model_lower or "local" in model_lower:
            print("running qwen")
            # Thinking models (e.g. Qwen3.5-35B-A3B) use reasoning_content which
            # exhausts max_tokens, leaving content=None. Use higher max_tokens and
            # pass extra_body to disable thinking mode so all tokens go to content.
            is_thinking_model = any(
                tag in model_lower
                for tag in ["qwen3.5", "qwen3-", "qwen3/"]
                if "instruct" not in model_lower
            )
            # Harness parity: short label responses + deterministic sampling.
            # LLMSQL's proxy_model sends max_tokens=16, temperature=0.0.
            qwen_config = {
                "max_batch_size": self.concurrent_llm_worker,
                "max_tokens": 16,
                "temperature": 0.0,
            }
            extra_body = {}
            if is_thinking_model:
                qwen_config["max_tokens"] = 4096
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            model_id = (
                self.model_name
                if self.model_name.startswith("hosted_vllm/")
                else f"hosted_vllm/{self.model_name}"
            )
            api_base = "http://localhost:8000/v1"
            print(f"[lotus] LM model_id={model_id} api_base={api_base}")
            _apply_harness_alignment_patches()
            return LM(
                model_id,  # LiteLLM prefix for vLLM
                api_base=api_base,
                api_key="EMPTY",  # vLLM ignores it; LiteLLM requires a value
                **({"extra_body": extra_body} if extra_body else {}),
                **qwen_config
            )
    
        elif (
            "gemini-2.5-flash" in model_lower
            or "gemini_2_5_flash" in model_lower
        ):
            # gemini-2.5-flash: reasoning_effort="disable", works well
            return LM(
                self.model_name, **base_config, reasoning_effort="disable"
            )
        elif (
            "gemini-2.0-flash" in model_lower
            or "gemini_2_0_flash" in model_lower
        ):
            # gemini-2.0-flash: no reasoning_effort parameter needed
            return LM(self.model_name, **base_config)
        elif "gpt-5-mini" in model_lower or "gpt_5_mini" in model_lower:
            # gpt-5-mini: reasoning_effort="minimal"
            return LM(
                self.model_name, **base_config, reasoning_effort="minimal"
            )
        else:
            # Default configuration for unknown models (no reasoning_effort)
            print(
                f"Warning: Unknown model '{self.model_name}', using default configuration"
            )
            return LM(self.model_name, **base_config)

    def _initialize_lotus_with_warmup(self):
        """Initialize LOTUS and perform connection warmup to avoid first-query errors."""

        # Configure LOTUS
        lotus.settings.configure(lm=self.lm)

        # Strategy 1: Perform a simple warmup query
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Make a simple test call to establish connection.
                # lotus.LM expects messages as List[List[Dict]] (one
                # conversation per row). Passing a raw string was a no-op
                # for many lotus versions, so use a real chat message.
                print(
                    f"[lotus] warmup attempt {attempt + 1}: sending test request..."
                )
                self.lm.__call__(
                    [[{"role": "user", "content": "hi"}]],
                    show_progress_bar=False,
                )
                print(
                    f"LOTUS connection warmed up successfully on attempt {attempt + 1}"
                )
                break

            except Exception as e:
                import traceback
                print(f"Warmup attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                traceback.print_exc()
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Don't fail initialization, just log the warning
                    print(
                        "Warning: Connection warmup failed, but continuing..."
                    )

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost based on prompt and completion tokens.
        
        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            
        Returns:
            Total cost in USD
        """
        model_name = self.model_name.lower()
        
        # Find matching pricing configuration
        pricing_config = None
        for model_key in PRICING:
            if model_key.lower() in model_name:
                pricing_config = PRICING[model_key]
                break
        
        if pricing_config is None:
            print(f"Warning: No pricing found for model '{self.model_name}', cost calculation skipped")
            return 0.0
        
        # Calculate cost: prices are per 1M tokens
        input_cost = (prompt_tokens / 1_000_000) * pricing_config["text"]
        output_cost = (completion_tokens / 1_000_000) * pricing_config["output"]
        total_cost = input_cost + output_cost
        
        return total_cost

    @override
    def get_system_name(self) -> str:
        return "lotus"

    def execute_query(self, query_id: int) -> GenericQueryMetric:
        """
        Execute a specific query using LOTUS and return metric with results.

        Args:
            query_id: ID of the query (e.g., 1 for Q1, 5 for Q5)

        Returns:
            QueryMetric object containing results DataFrame and metrics
        """
        # Create appropriate metric object
        metric = GenericQueryMetric(query_id=query_id, status="pending")

        # Reset token stats before each query
        try:
            lotus.settings.lm.reset_stats()
        except Exception as e:
            print(f"  Warning: Could not reset stats: {e}")

        try:
            query_fn = self._discover_query_impl(query_id)
            print(f"[lotus] Executing Q{query_id}...", flush=True)
            start_time = time.time()
            results = query_fn()
            execution_time = time.time() - start_time
            print(f"[lotus] Q{query_id} finished in {execution_time:.2f}s", flush=True)

            # Store results in metric
            metric.execution_time = execution_time
            metric.results = results
            metric.status = "success"

            # Get token usage and cost
            self._update_token_usage(metric)

        except Exception as e:
            # Handle failure
            metric.status = "failed"
            metric.error = str(e)
            metric.results = self._get_empty_results_dataframe(query_id)
            print(f"  Error in Q{query_id} execution: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Reset stats after storing
            try:
                lotus.settings.lm.reset_stats()
            except:
                pass

        return metric

    def _get_empty_results_dataframe(self, query_id: int) -> pd.DataFrame:
        """
        Get empty DataFrame with correct columns for a query.

        Args:
            query_id: ID of the query

        Returns:
            Empty DataFrame with correct columns
        """
        if query_id == 1:
            return pd.DataFrame(columns=["reviewId", "movieId", "reviewText"])
        elif query_id == 5:
            return pd.DataFrame(
                columns=[
                    "id",
                    "reviewId_left",
                    "reviewText_left",
                    "reviewId_right",
                    "reviewText_right",
                ]
            )
        else:
            return pd.DataFrame()

    def _update_token_usage(self, metric: GenericQueryMetric):
        """Update metric with token usage and cost information."""
        try:
            # Get usage stats from LOTUS using the correct API
            prompt_tokens = lotus.settings.lm.stats.physical_usage.prompt_tokens
            completion_tokens = lotus.settings.lm.stats.physical_usage.completion_tokens
            total_tokens = lotus.settings.lm.stats.physical_usage.total_tokens
            total_cost = lotus.settings.lm.stats.physical_usage.total_cost
            
            # Calculate cost using our token consumption
            calculated_cost = self._calculate_cost(prompt_tokens, completion_tokens)

            # Print both tokens for comparison
            print(f"prompt token: {prompt_tokens}, completion token: {completion_tokens}")
            print(f"total tokens: {total_tokens}")
            
            # Print both costs for comparison
            print(f"  LOTUS cost: ${total_cost:.6f}")
            print(f"  Calculated cost based on token consumption: ${calculated_cost:.6f}")

            metric.token_usage = total_tokens
            metric.money_cost = calculated_cost

        except Exception as e:
            print(f"  Warning: Could not get token usage: {e}")
            metric.token_usage = 0
            metric.money_cost = 0.0

    def _discover_queries(self) -> List[int]:
        """
        Discover available queries for LOTUS.

        Any method named ``_execute_q<i>`` (where <i> is an integer ≥1) is treated
        as an implemented query.  The function returns the list of those integer
        IDs in ascending order.

        Returns:
            List of available query IDs
        """
        # Return only implemented queries

        pattern = re.compile(r"_execute_q(\d+)$")
        query_ids: List[int] = []

        # `dir(self)` lists all attribute names visible on the instance;
        # we then pick out callables whose names match our pattern.
        for attr_name in dir(self):
            match = pattern.match(attr_name)
            if match:
                attr = getattr(self, attr_name, None)
                if callable(attr):
                    query_ids.append(int(match.group(1)))

        return sorted(query_ids)
