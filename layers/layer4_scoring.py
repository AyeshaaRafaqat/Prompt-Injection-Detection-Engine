"""
PIDE Layer 4: Risk Engine
Aggregates scores from L1, L2, and L3 into a final decision.
Handles decision routing, hot-reloading of config, and audit logging.
"""

import yaml
import json
import logging
import hashlib
import time
import os
from typing import Tuple, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PIDE.Layer4")

class RiskEngine:
    """
    Final scoring engine for PIDE.
    Implements weighted risk calculation and audit logging.
    """

    def __init__(self, config_path: str = "config/scoring.yaml") -> None:
        """
        Initializes the risk engine.
        
        Args:
            config_path: Path to the scoring configuration.
        """
        self.config_path = config_path
        self._load_config()

    def _load_config(self) -> None:
        """Loads (or reloads) weights and thresholds from YAML."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            self.weights = self.config['weights']
            self.thresholds = self.config['thresholds']
            
            logger.info("Risk Engine configuration (re)loaded.")
        except Exception as e:
            logger.error(f"Error loading L4 config: {e}")
            # Fallback to defaults if load fails to maintain availability
            self.weights = {"l1_weight": 0.25, "l2_weight": 0.45, "l3_weight": 0.30}
            self.thresholds = {"block_threshold": 0.65, "sanitise_threshold": 0.35}

    def score(self, l1: float, l2: float, l3: float) -> Tuple[float, str]:
        """
        Calculates final risk score and decision.
        Hot-reloads config before calculation for agility.
        
        Args:
            l1: Score from Layer 1 (Regex)
            l2: Score from Layer 2 (Embedding)
            l3: Score from Layer 3 (Heuristic)
            
        Returns:
            Tuple of (risk_score, decision).
            Decision is one of: BLOCK, SANITISE, ALLOW.
        """
        try:
            # Hot-reload check (could be optimized with file watch, but simple read is safe for now)
            self._load_config()
            
            # Layer 1 Short-circuit override: high-confidence direct regex match is definitive
            if l1 == 1.0:
                return 1.0, "BLOCK"
            
            risk = (
                self.weights['l1_weight'] * l1 +
                self.weights['l2_weight'] * l2 +
                self.weights['l3_weight'] * l3
            )
            
            risk = min(1.0, max(0.0, risk))
            
            if risk >= self.thresholds['block_threshold']:
                decision = "BLOCK"
            elif risk >= self.thresholds['sanitise_threshold']:
                decision = "SANITISE"
            else:
                decision = "ALLOW"
                
            return float(risk), decision
            
        except Exception as e:
            logger.error(f"L4 Exception: {e}")
            return 1.0, "BLOCK" # Fail-secure

    def write_audit_log(
        self, 
        prompt: str, 
        l1: float, 
        l2: float, 
        l3: float,
        risk: float, 
        decision: str, 
        trigger: str,
        l2_exemplars: List[str], 
        l3_signals: Dict[str, float],
        latency_ms: float
    ) -> None:
        """
        Writes a privacy-preserving audit entry to logs/audit.jsonl.
        
        Args:
            prompt: Raw prompt (used only for hashing).
        """
        try:
            log_dir = os.path.dirname(self.config['logging']['audit_log_path'])
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            # Privacy: hash the prompt, don't store raw content
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            
            entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "prompt_hash": prompt_hash,
                "scores": {"l1": l1, "l2": l2, "l3": l3},
                "risk": risk,
                "decision": decision,
                "trigger": trigger,
                "l2_exemplars": l2_exemplars,
                "l3_signals": l3_signals,
                "latency_ms": latency_ms,
                "weights": self.weights
            }
            
            with open(self.config['logging']['audit_log_path'], 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

def create_layer4(config_path: str = "config/scoring.yaml") -> RiskEngine:
    """Factory function for Layer 4."""
    return RiskEngine(config_path)
