"""`mlx_lm.server` with one defect patched at runtime, before it starts.

The defect, observed four times across V1-V4: the model emits
`<tool_call>...</tool_call>` with no `<function=...>` inside,
`mlx_lm/tool_parsers/qwen3_coder.py` raises `ValueError("No function
provided.")`, the exception escapes `do_POST` into `socketserver`, and the
connection closes with no response. The client sees a transport error, the raw
generation is destroyed, and a whole episode dies on one malformed turn.

The patch does not touch site-packages. It rebinds the module attribute before
the server imports it, and turns the unparseable call into a synthetic
`PythonInterpreter` call whose code is the raw block content. For this model
family the content of a bare `<tool_call>` is nearly always code it forgot to
wrap; as code it either runs or produces a SyntaxError observation — and either
way the loop continues and the model sees what happened, which is the same
errors-become-observations rule the rest of the harness follows.

Started by serve.sh; the argv stays greppable for the run manifest.
"""

import json
import sys
import threading

from mlx_lm.tool_parsers import qwen3_coder

_original = qwen3_coder.parse_tool_call


def parse_tool_call_or_recover(model_output, tools=None):
    """Parse a tool call, or hand back a synthetic one instead of raising.

    `mlx_lm`'s Qwen3-Coder parser raises `ValueError` on a `<tool_call>` block
    with no `<function=>` inside it. That exception escapes `do_POST`, unwinds
    into `socketserver`, and drops the connection — so a malformed block from
    the model killed the episode rather than producing a bad turn. Recovering
    here keeps a generation defect a generation defect.
    """
    try:
        return _original(model_output, tools)
    except Exception:                                              # noqa: BLE001
        text = model_output
        for tag in (qwen3_coder.tool_call_start, qwen3_coder.tool_call_end):
            text = text.replace(tag, "")
        return dict(name="PythonInterpreter", arguments={"code": text.strip()})


qwen3_coder.parse_tool_call = parse_tool_call_or_recover


CACHE_RESET_PATH = "/admin/prompt-cache/reset"
_cache_reset_lock = threading.Lock()
_cache_reset_epoch = 0


def install_cache_reset_endpoint():
    """Expose a local episode-boundary reset with a verifiable attestation.

    The endpoint changes no sampling or model state. It clears only reusable
    prompt/state-machine caches, and reports the before/after counts so a run
    never has to infer isolation from the fact that it requested it.
    """
    from mlx_lm import server

    original_do_post = server.APIHandler.do_POST

    def do_post_with_cache_reset(self):
        global _cache_reset_epoch
        if self.path != CACHE_RESET_PATH:
            return original_do_post(self)
        with _cache_reset_lock:
            cache = self.response_generator.prompt_cache
            before = {
                "sequences": len(cache),
                "bytes": cache.nbytes,
                "by_type": cache.stats_by_type(),
                "state_machines": len(self.response_generator._state_machine_cache),
            }
            cache.trim_to(n_sequences=0, n_bytes=0)
            self.response_generator._state_machine_cache.clear()
            _cache_reset_epoch += 1
            after = {
                "sequences": len(cache),
                "bytes": cache.nbytes,
                "by_type": cache.stats_by_type(),
                "state_machines": len(self.response_generator._state_machine_cache),
            }
        payload = json.dumps({
            "ok": after["sequences"] == 0 and after["bytes"] == 0,
            "policy": "mlx_prompt_cache_reset_v1",
            "epoch": _cache_reset_epoch,
            "before": before,
            "after": after,
        }, sort_keys=True).encode()
        self._set_completion_headers(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    server.APIHandler.do_POST = do_post_with_cache_reset

if __name__ == "__main__":
    from mlx_lm.server import main

    install_cache_reset_endpoint()
    print("serve_patched: qwen3_coder.parse_tool_call wrapped with recovery",
          file=sys.stderr)
    main()
