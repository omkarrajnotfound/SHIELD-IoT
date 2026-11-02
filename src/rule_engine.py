"""
Phase E: Automated Security Rule Engine
Translates local TreeSHAP attribution values into natural language,
actionable incident response playbooks for SOC security operators.

Outputs:
  - results/sample_alert_report.txt
  - results/batch_alert_summary.json
"""
import os
import sys
import time
import json
import joblib
import datetime
import numpy as np
import pandas as pd
import shap

# Directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# Security Domain Knowledge Base (Feature Semantics & Rationale)
# -------------------------------------------------------------
FEATURE_RATIONALE = {
    "DNS_TTL_ANSWER": {
        "unit": "seconds",
        "description": "DNS Time-to-Live on query response",
        "rationale_high": "Unusually high DNS TTL suggests attempt to lock in compromised name resolution.",
        "rationale_low": "DNS Time-to-Live dropped near zero — characteristic of fast-flux DNS or short-lived Command-and-Control (C2) domain beaconing."
    },
    "TCP_WIN_MAX_IN": {
        "unit": "bytes",
        "description": "Maximum inbound TCP receive window size",
        "rationale_high": "TCP receive window abnormally expanded ({val:.0f} bytes) — indicates bulk encrypted payload delivery or data exfiltration burst.",
        "rationale_low": "Restricted TCP receive window ({val:.0f} bytes) — consistent with resource exhaustion or SYN scanning."
    },
    "TCP_WIN_MAX_OUT": {
        "unit": "bytes",
        "description": "Maximum outbound TCP receive window size",
        "rationale_high": "Large outbound TCP window ({val:.0f} bytes) indicates sustained high-throughput remote egress.",
        "rationale_low": "Constrained outbound TCP window indicates socket throttling or connection probing."
    },
    "FLOW_DURATION_MILLISECONDS": {
        "unit": "ms",
        "description": "Bidirectional network flow duration",
        "rationale_high": "Prolonged flow duration ({val:.1f} ms) indicates persistent reverse shell or interactive session tunneling.",
        "rationale_low": "Extremely brief flow duration ({val:.1f} ms) — consistent with automated attack bursts, rapid port scans, or SYN floods."
    },
    "MAX_IP_PKT_LEN": {
        "unit": "bytes",
        "description": "Maximum IP packet length observed in flow",
        "rationale_high": "Maximum packet size reached MTU limit ({val:.0f} bytes) — consistent with bulk data dumping or large exploit payload transfer.",
        "rationale_low": "Uniformly diminutive packets ({val:.0f} bytes) indicate reconnaissance probing or denial-of-service packet storm."
    },
    "MIN_IP_PKT_LEN": {
        "unit": "bytes",
        "description": "Minimum IP packet length observed in flow",
        "rationale_high": "Abnormally elevated minimum packet length ({val:.0f} bytes) — packets carry continuous payload with no bare ACKs.",
        "rationale_low": "Minimal header-only packet size ({val:.0f} bytes) — indicates probe or ACK-less synchronization attempt."
    },
    "SHORTEST_FLOW_PKT": {
        "unit": "bytes",
        "description": "Shortest packet observed across flow",
        "rationale_high": "Shortest packet contains substantial payload ({val:.0f} bytes), anomalous for normal handshakes.",
        "rationale_low": "Zero-payload framing packets ({val:.0f} bytes) typical of reconnaissance and port scanning sweeps."
    },
    "TCP_FLAGS": {
        "unit": "bitmap",
        "description": "Cumulative TCP control flags",
        "rationale_high": "Anomalous TCP flag combination (bitmap={val:.0f}) — non-standard flag states (PSH/URG/RST) used to evade basic stateful filters.",
        "rationale_low": "TCP flag state indicates bare SYN or FIN sequence without corresponding acknowledgment."
    },
    "CLIENT_TCP_FLAGS": {
        "unit": "bitmap",
        "description": "TCP flags set by client initiator",
        "rationale_high": "Client pushed abnormal TCP control flags ({val:.0f}) — typical of custom exploit scripts or evasion frameworks.",
        "rationale_low": "Client initiated connection with anomalous flag distribution."
    },
    "SERVER_TCP_FLAGS": {
        "unit": "bitmap",
        "description": "TCP flags returned by server responder",
        "rationale_high": "Server responder returned abnormal control flags ({val:.0f}) — suggests service disruption or unusual server reset.",
        "rationale_low": "Absence of standard server TCP acknowledgment flags."
    },
    "SRC_TO_DST_AVG_THROUGHPUT": {
        "unit": "bps",
        "description": "Average throughput from source to destination",
        "rationale_high": "Elevated outbound throughput ({val:,.0f} bps) indicates volumetric exfiltration or intensive injection traffic.",
        "rationale_low": "Low outbound throughput indicates periodic low-and-slow heartbeat beaconing."
    },
    "DST_TO_SRC_AVG_THROUGHPUT": {
        "unit": "bps",
        "description": "Average throughput from destination to source",
        "rationale_high": "Heavy inbound throughput ({val:,.0f} bps) suggests large payload ingress or remote script retrieval.",
        "rationale_low": "Negligible inbound return throughput typical of blind injection or unreachable spoofed target."
    },
    "L7_PROTO": {
        "unit": "protocol id",
        "description": "Application Layer Protocol Identifier",
        "rationale_high": "Unrecognized or non-standard application layer protocol identifier ({val:.0f}) on transport port.",
        "rationale_low": "Non-standard Layer 7 payload framing indicates encrypted tunneling over plain transport."
    },
    "MIN_TTL": {
        "unit": "hops",
        "description": "Minimum IP Time-To-Live across flow",
        "rationale_high": "Elevated TTL indicates local subnet or internal lateral propagation.",
        "rationale_low": "Low TTL indicates packet originated from multiple external hops or spoofed transit path."
    }
}

# -------------------------------------------------------------
# Incident Response Playbooks (Attack-Specific Action Templates)
# -------------------------------------------------------------
PLAYBOOKS = {
    "ransomware": {
        "risk_level": "★★★★★ CRITICAL",
        "severity": "CRITICAL",
        "threat_category": "Cryptographic Endpoint Hostage Attack",
        "mitre_technique": "T1486 (Data Encrypted for Impact)",
        "actions": [
            "[X] IMMEDIATELY isolate endpoint from local VLAN/WLAN to halt lateral encryption spread.",
            "[X] Block outbound DNS queries and perimeter C2 destination IP addresses.",
            "[X] Capture volatile memory (RAM) dump immediately for cryptographic key extraction.",
            "[V] Verify backup storage immutability and air-gap integrity before initiating node remediation."
        ]
    },
    "backdoor": {
        "risk_level": "★★★★★ CRITICAL",
        "severity": "CRITICAL",
        "threat_category": "Unauthorized Persistent Remote Access",
        "mitre_technique": "T1059 (Command and Scripting Interpreter)",
        "actions": [
            "[X] Terminate suspect process PID and sever persistent outbound reverse shell socket.",
            "[X] Quarantine affected host and inspect persistence hooks (cron, systemd, Windows registry).",
            "[X] Invalidate all active Kerberos/SSH credentials associated with the compromised host.",
            "[V] Review bastion and lateral jump logs for post-exploitation reconnaissance."
        ]
    },
    "mitm": {
        "risk_level": "★★★★☆ HIGH",
        "severity": "HIGH",
        "threat_category": "Man-In-The-Middle / Session Interception",
        "mitre_technique": "T1557 (Adversary-in-the-Middle)",
        "actions": [
            "[X] Flush ARP cache on local subnet default gateway and verify static gateway MAC bindings.",
            "[X] Enforce Dynamic ARP Inspection (DAI) and DHCP Snooping on edge access switches.",
            "[X] Invalidate active TLS/OAuth web sessions and force client re-authentication.",
            "[V] Inspect subnet routing tables for illegitimate ICMP redirect messages."
        ]
    },
    "ddos": {
        "risk_level": "★★★★☆ HIGH",
        "severity": "HIGH",
        "threat_category": "Distributed Denial of Service (Volumetric Flood)",
        "mitre_technique": "T1498 (Network Denial of Service)",
        "actions": [
            "[X] Engage upstream ISP BGP Flowspec drop rules to discard volumetric attack traffic.",
            "[X] Apply TCP SYN Proxy / SYN Cookies at perimeter edge firewall / load balancer.",
            "[X] Rate-limit offending CIDR prefix on border router ingress interface.",
            "[V] Divert edge ingress traffic through cloud DDoS scrubbing center."
        ]
    },
    "dos": {
        "risk_level": "★★★★☆ HIGH",
        "severity": "HIGH",
        "threat_category": "Denial of Service (Resource Starvation)",
        "mitre_technique": "T1499 (Endpoint Denial of Service)",
        "actions": [
            "[X] Throttle ingress connection rate from source subnet on local firewall.",
            "[X] Adjust server TCP keepalive and connection timeout parameters to free orphaned sockets.",
            "[X] Implement temporary drop ACL on offending host IP address.",
            "[V] Verify web server connection pool health and activate standby service node."
        ]
    },
    "injection": {
        "risk_level": "★★★★☆ HIGH",
        "severity": "HIGH",
        "threat_category": "Application Layer Code/Query Injection",
        "mitre_technique": "T1055 (Process/Query Injection)",
        "actions": [
            "[X] Block client IP address at Web Application Firewall (WAF) layer.",
            "[X] Inspect web server access logs for SQLi / command syntax in URL and POST parameters.",
            "[X] Review backend database query logs for unauthorized schema modification or data extraction.",
            "[V] Verify parameterized query compliance and patch vulnerable web service endpoint."
        ]
    },
    "password": {
        "risk_level": "★★★☆☆ MEDIUM",
        "severity": "MEDIUM",
        "threat_category": "Credential Brute-Force / Password Spraying",
        "mitre_technique": "T1110 (Brute Force)",
        "actions": [
            "[X] Temporarily lock target account and enforce exponential authentication backoff.",
            "[X] Block source IP from SSH, Telnet, FTP, and RDP edge listener ports.",
            "[X] Require immediate Multi-Factor Authentication (MFA) challenge upon next login.",
            "[V] Audit authentication logs for distributed botnet credential stuffing patterns."
        ]
    },
    "scanning": {
        "risk_level": "★★★☆☆ MEDIUM",
        "severity": "MEDIUM",
        "threat_category": "Network Reconnaissance / Port Scanning",
        "mitre_technique": "T1046 (Network Service Discovery)",
        "actions": [
            "[X] Add scanning source IP to automatic 24-hour firewall drop list.",
            "[X] Verify perimeter port exposure and disable unused open ports (e.g., Telnet, SMB).",
            "[X] Enable port-scan threshold alerting on local Network Intrusion Detection System.",
            "[V] Correlate scanning IP against external Threat Intelligence feeds (AbuseIPDB, AlienVault)."
        ]
    },
    "xss": {
        "risk_level": "★★★☆☆ MEDIUM",
        "severity": "MEDIUM",
        "threat_category": "Cross-Site Scripting (Client-Side Injection)",
        "mitre_technique": "T1189 (Drive-by Compromise)",
        "actions": [
            "[X] Invalidate active session tokens associated with vulnerable application paths.",
            "[X] Implement HTML entity encoding and context-aware escaping on reflective form inputs.",
            "[X] Enforce strict Content Security Policy (CSP) header: default-src 'self'.",
            "[V] Inspect web application response payloads for reflected script fragments."
        ]
    }
}


class SHAPToRuleEngine:
    """
    Core automated security rule engine.
    Converts TreeSHAP feature contributions into operator-readable alert reports.
    """
    def __init__(self, model_path, scaler_path, features_path, encoder_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)
        self.classes = list(self.encoder.classes_)
        
        with open(features_path, "r", encoding="utf-8") as f:
            self.features = json.load(f)["final_features"]
            
        print("[SHAPToRuleEngine] Initializing TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)
        print("[SHAPToRuleEngine] Engine ready!")

    def generate_alert(self, flow_scaled_row, flow_orig_row, alert_id=None, timestamp=None):
        if alert_id is None:
            alert_id = f"#ALT-{datetime.datetime.now().strftime('%Y%m%d')}-{np.random.randint(100, 999)}"
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Model inference
        row_scaled_df = pd.DataFrame([flow_scaled_row[self.features]], columns=self.features)
        pred_idx = self.model.predict(row_scaled_df)[0]
        pred_class = self.classes[pred_idx]
        probs = self.model.predict_proba(row_scaled_df)[0]
        confidence = float(probs[pred_idx]) * 100

        # Skip benign alerts
        if pred_class == "Benign":
            return None

        # 2. Local TreeSHAP Attribution Calculation
        shap_values = self.explainer(row_scaled_df)
        shap_contributions = shap_values.values[0, :, pred_idx]

        # 3. Identify Top-3 Contributing Features
        sorted_indices = np.argsort(shap_contributions)[::-1]
        top_feature_indices = sorted_indices[:3]

        triggers = []
        for rank, feat_idx in enumerate(top_feature_indices, start=1):
            feat_name = self.features[feat_idx]
            contribution = float(shap_contributions[feat_idx])
            unscaled_val = float(flow_orig_row[feat_name])
            scaled_val = float(flow_scaled_row[feat_name])

            feat_info = FEATURE_RATIONALE.get(feat_name, {
                "unit": "value",
                "description": feat_name,
                "rationale_high": f"Feature {feat_name} showed abnormal elevation.",
                "rationale_low": f"Feature {feat_name} showed anomalous suppression."
            })

            # Select rationale based on whether value is high or low relative to median
            rationale_tmpl = feat_info["rationale_high"] if scaled_val >= 0 else feat_info["rationale_low"]
            rationale_text = rationale_tmpl.format(val=unscaled_val)

            triggers.append({
                "rank": rank,
                "feature": feat_name,
                "unscaled_value": unscaled_val,
                "unit": feat_info["unit"],
                "shap_contribution": round(contribution, 2),
                "rationale": rationale_text
            })

        # 4. Lookup Incident Response Playbook
        playbook = PLAYBOOKS.get(pred_class, {
            "risk_level": "★★★★☆ HIGH",
            "severity": "HIGH",
            "threat_category": "Suspicious Intrusion Activity",
            "mitre_technique": "T1000",
            "actions": [
                "[X] Isolate source IP from network.",
                "[X] Capture network packet dump.",
                "[V] Review system security logs."
            ]
        })

        # 5. Format Plain Text Report
        ascii_report = self._format_ascii_report(
            alert_id=alert_id,
            timestamp=timestamp,
            attack_type=pred_class.upper(),
            confidence=confidence,
            playbook=playbook,
            triggers=triggers
        )

        # 6. Format Structured JSON Alert (SIEM/Syslog)
        json_alert = {
            "alert_id": alert_id,
            "timestamp": timestamp,
            "attack_type": pred_class,
            "threat_category": playbook["threat_category"],
            "mitre_technique": playbook["mitre_technique"],
            "confidence_pct": round(confidence, 2),
            "severity": playbook["severity"],
            "risk_level": playbook["risk_level"],
            "primary_triggers": triggers,
            "recommended_actions": playbook["actions"]
        }

        return ascii_report, json_alert

    def _format_ascii_report(self, alert_id, timestamp, attack_type, confidence, playbook, triggers):
        lines = []
        lines.append("=" * 65)
        lines.append("           REAL-TIME SECURITY ALERT REPORT (AI-SOC)")
        lines.append("=" * 65)
        lines.append(f"  Alert ID    : {alert_id}")
        lines.append(f"  Timestamp   : {timestamp}")
        lines.append(f"  Alert Type  : {attack_type} (Confidence: {confidence:.1f}%)")
        lines.append(f"  Threat Desc : {playbook['threat_category']}")
        lines.append(f"  MITRE ATT&CK: {playbook['mitre_technique']}")
        lines.append(f"  Risk Level  : {playbook['risk_level']}")
        lines.append("")
        lines.append("  PRIMARY TRIGGER REASONS (SHAP EXPLAINABILITY RATIONALE):")
        for trig in triggers:
            val_str = f"{trig['unscaled_value']:.1f} {trig['unit']}" if trig['unit'] != "bitmap" else f"0x{int(trig['unscaled_value']):02X}"
            lines.append(f"    {trig['rank']}. {trig['feature']} = {val_str}")
            lines.append(f"       -> {trig['rationale']}")
            lines.append(f"          Contribution: +{trig['shap_contribution']:.2f} toward {attack_type}")
            lines.append("")
        lines.append("  RECOMMENDED ACTIONS (INCIDENT RESPONSE PLAYBOOK):")
        for act in playbook["actions"]:
            lines.append(f"    {act}")
        lines.append("=" * 65)
        return "\n".join(lines)


def run_rule_engine_pipeline():
    print("=" * 80)
    print("PHASE E: AUTOMATED SECURITY RULE ENGINE PIPELINE")
    print("=" * 80)

    model_path = os.path.join(MODELS_DIR, "lightgbm_smote_balanced_top14.joblib")
    scaler_path = os.path.join(PROCESSED_DIR, "robust_scaler.pkl")
    features_path = os.path.join(PROCESSED_DIR, "selected_features.json")
    encoder_path = os.path.join(PROCESSED_DIR, "label_encoder_multi.pkl")
    data_path = os.path.join(PROCESSED_DIR, "train_test_data.pkl")
    target_path = os.path.join(PROCESSED_DIR, "train_test_multi_targets.pkl")

    engine = SHAPToRuleEngine(model_path, scaler_path, features_path, encoder_path)

    # Load test data and unscale to original units
    print("\nLoading test flows and performing inverse scaling to physical units...")
    _, X_test_scaled, _, _ = joblib.load(data_path)
    _, y_test = joblib.load(target_path)
    classes = engine.classes

    scaler = joblib.load(scaler_path)
    X_clipped = np.clip(X_test_scaled.values, -100.0, 100.0)
    X_test_unscaled_arr = scaler.inverse_transform(X_clipped)
    X_test_unscaled = pd.DataFrame(X_test_unscaled_arr, columns=X_test_scaled.columns, index=X_test_scaled.index)

    # Get predictions
    preds = engine.model.predict(X_test_scaled[engine.features])

    # Select representative True Positive alerts for all attack categories
    attack_classes = [c for c in classes if c != "Benign"]
    print(f"\nExtracting representative detected flows across {len(attack_classes)} attack categories...")

    all_ascii_reports = []
    all_json_alerts = []

    for idx, c_name in enumerate(attack_classes, start=1):
        c_idx = classes.index(c_name)
        tp_indices = np.where((y_test.values == c_idx) & (preds == c_idx))[0]

        if len(tp_indices) == 0:
            print(f"  [Warning] No true positive sample found for {c_name}")
            continue

        sample_idx = tp_indices[0]
        flow_scaled = X_test_scaled.iloc[sample_idx]
        flow_unscaled = X_test_unscaled.iloc[sample_idx]

        alert_id = f"#ALT-20261102-{idx:03d}"
        timestamp = f"2026-11-02 {14 + (idx//3):02d}:{10 + (idx*5)%50:02d}:34"

        ascii_rep, json_alert = engine.generate_alert(
            flow_scaled_row=flow_scaled,
            flow_orig_row=flow_unscaled,
            alert_id=alert_id,
            timestamp=timestamp
        )

        all_ascii_reports.append(ascii_rep)
        all_json_alerts.append(json_alert)
        print(f"  [Generated Alert {idx}] {c_name.upper():<12} -> Conf: {json_alert['confidence_pct']:.1f}% | Triggers: {json_alert['primary_triggers'][0]['feature']}")

    # 1. Save Full Sample Alert Report (Plain Text)
    report_file_path = os.path.join(RESULTS_DIR, "sample_alert_report.txt")
    full_report_text = "\n\n".join(all_ascii_reports)
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(full_report_text)
    print(f"\nSaved human-auditable alert report to: {report_file_path}")

    # 2. Save Batch Alert Summary (Machine-Readable JSON)
    batch_json_path = os.path.join(RESULTS_DIR, "batch_alert_summary.json")
    batch_data = {
        "generator": "AI-SOC SHAP-to-Rule Automated Engine",
        "timestamp_generated": datetime.datetime.now().isoformat(),
        "total_alerts": len(all_json_alerts),
        "alerts": all_json_alerts
    }
    with open(batch_json_path, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, indent=2)
    print(f"Saved machine-readable SIEM batch log to: {batch_json_path}")

    # Display First Alert (Safe ASCII console print)
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUT PREVIEW (OPERATOR ALERT REPORT):")
    print("=" * 80)
    try:
        print(all_ascii_reports[0])
    except UnicodeEncodeError:
        print(all_ascii_reports[0].encode("ascii", errors="replace").decode("ascii"))

    return all_ascii_reports, all_json_alerts

if __name__ == "__main__":
    run_rule_engine_pipeline()
