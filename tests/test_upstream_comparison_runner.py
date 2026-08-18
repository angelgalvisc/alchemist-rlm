from pathlib import Path


def test_upstream_runner_keeps_the_baseline_external_and_pinned():
    text = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "run_upstream_rlm_pairs.py"
    ).read_text()
    assert 'PINNED_UPSTREAM_COMMIT = "caf0bffa1acec17c062559433b4cd4ed92eee3d6"' in text
    assert "--upstream-checkout" in text
    assert "sys.path.insert(0, str(upstream))" in text
    assert '"typed_semantic_operations": "none"' in text
    assert '"output_repair": "none"' in text
    assert '"chat_template_kwargs": {"enable_thinking": False}' in text
    assert "check_official_binding(items, frozen)" in text
    assert "official_context(items, frozen)" in text
    assert "items[number - 1]" not in text
    assert 'os.environ.get("ALCHEMIST_MODEL")' in text
    assert "/Users/" not in text


def test_public_name_distinguishes_both_harnesses():
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    assert readme.startswith("# Alchemist-RLM Harness")
    assert "**Upstream RLM**" in readme
    assert "unmodified official implementation" in readme
