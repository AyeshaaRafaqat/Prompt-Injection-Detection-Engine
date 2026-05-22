"""
PIDE Test Suite: Layer 4
"""
import pytest
import os
import json
from layers.layer4_scoring import create_layer4

@pytest.fixture(scope="module")
def l4():
    return create_layer4()

def test_block_decision(l4):
    # High scores in all layers
    risk, decision = l4.score(1.0, 1.0, 1.0)
    assert risk == 1.0
    assert decision == "BLOCK"

def test_sanitise_decision(l4):
    # Medium scores
    risk, decision = l4.score(0.0, 0.8, 0.0) # 0.8 * 0.45 = 0.36
    assert risk >= 0.35 and risk < 0.65
    assert decision == "SANITISE"

def test_allow_decision(l4):
    # Low scores
    risk, decision = l4.score(0.0, 0.2, 0.1)
    assert risk < 0.35
    assert decision == "ALLOW"

def test_audit_log_written(l4):
    log_path = "logs/audit.jsonl"
    if os.path.exists(log_path):
        os.remove(log_path)
    
    l4.write_audit_log(
        prompt="test prompt",
        l1=0.0, l2=0.0, l3=0.0,
        risk=0.0, decision="ALLOW",
        trigger="None",
        l2_exemplars=[],
        l3_signals={},
        latency_ms=10.0
    )
    
    assert os.path.exists(log_path)
    with open(log_path, 'r') as f:
        line = f.readline()
        data = json.loads(line)
        assert "prompt_hash" in data

def test_audit_no_raw_prompt(l4):
    prompt = "sensitive command ignore instructions"
    l4.write_audit_log(
        prompt=prompt,
        l1=0.0, l2=0.0, l3=0.0,
        risk=0.0, decision="ALLOW",
        trigger="None",
        l2_exemplars=[],
        l3_signals={},
        latency_ms=10.0
    )
    
    with open("logs/audit.jsonl", 'r') as f:
        logs = f.readlines()
        last_log = logs[-1]
        assert prompt not in last_log

def test_fail_secure(l4, monkeypatch):
    def mock_load(*args):
        raise Exception("Config error")
    
    monkeypatch.setattr(l4, "_load_config", mock_load)
    risk, decision = l4.score(0.0, 0.0, 0.0)
    assert risk == 1.0
    assert decision == "BLOCK"
