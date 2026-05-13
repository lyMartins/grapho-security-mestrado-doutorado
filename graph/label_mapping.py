"""Shared label mappings for LLM and Hackmageddon threat classes."""

from __future__ import annotations


ATTACK_TO_THREAT_LABEL = {
    "account takeover": "account_takeover",
    "attack": "other_cyber",
    "business email compromise": "credential_theft",
    "coordinated inauthentic behavior": "disinformation_or_influence",
    "credential stuffing": "credential_theft",
    "ddos": "ddos_or_disruption",
    "defacement": "ddos_or_disruption",
    "flash loan": "fraud_or_scam",
    "malicious script injection": "vulnerability_or_exploit",
    "malvertising": "malware",
    "malware": "malware",
    "misconfiguration": "vulnerability_or_exploit",
    "ransomware": "ransomware",
    "scam": "fraud_or_scam",
    "targeted attack": "other_cyber",
    "unknown": "other_cyber",
    "vulnerability": "vulnerability_or_exploit",
}

DEFAULT_THREAT_LABELS = [
    "account_takeover",
    "botnet_or_c2",
    "credential_theft",
    "data_breach_or_leak",
    "ddos_or_disruption",
    "disinformation_or_influence",
    "exfiltration",
    "fraud_or_scam",
    "initial_access_activity",
    "insider_threat",
    "lateral_movement",
    "malware",
    "not_a_threat",
    "other_cyber",
    "phishing",
    "physical_or_hybrid_threat",
    "privilege_escalation",
    "ransomware",
    "reconnaissance",
    "supply_chain_compromise",
    "vulnerability_or_exploit",
    "wiper_or_destruction",
]


def build_label_mapping(labels: set[str] | None = None) -> dict[str, int]:
    """Return a stable label-to-id mapping with known labels first."""
    ordered = list(DEFAULT_THREAT_LABELS)
    for label in sorted(labels or set()):
        if label and label not in ordered:
            ordered.append(label)
    return {label: index for index, label in enumerate(ordered)}

