#!/usr/bin/env python3
"""x402 Endpoint Validator — CI compliance checker.

Reads input via environment variables prepared by entrypoint.sh, validates
each endpoint against five checks, writes a JSON report, and exits with
status 0 (all checks meet thresholds) or 1 (one or more failed).

Checks per endpoint
-------------------
1. HTTP reachability — GET returns a non-5xx status; expectation is 200 for
   resource roots and 402 for paywalled paths. The check records the status
   and treats network errors / 5xx as failures.
2. /.well-known/x402 manifest — fetched at the origin, must parse as JSON
   and contain either an `accepts[]` (per resource) or a `resources[]` /
   `payment` block compatible with x402 spec v1 or v2.
3. x402 v0.7 body conformance on POST — POSTs an empty JSON body with no
   X-PAYMENT header and verifies that the 402 response body contains the
   required keys (`x402Version`, `accepts`, `error`) and that each
   `accepts[]` entry carries `scheme`, `network`, and a price field
   (`maxAmountRequired` for v1+, or `amount` for early v2 drafts).
4. Response time P95 — five sequential probes; the 95th percentile of the
   latency samples (ms) must be below `threshold_p95_ms`.
5. Payment-required behavior — confirms that the unauthenticated POST in
   check 3 returned HTTP 402 (not 401/403/200/empty body).

Spec reference: the response-body shape mirrored here is the
`x402Version=1` shape observed across the dominant cohort of live x402
endpoints (canonical hash `ad0c163412139f1e`, 97,532 scans across 10,841
URLs per the operator-launch-kit pattern analysis). Both v1 and v2 schemas
are accepted to avoid penalising early adopters.

Exit codes
----------
- 0: all endpoints passed (or fail_on == "never")
- 1: one or more endpoints failed (subject to fail_on policy)
- 2: input parsing error
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - yaml is in image but stay defensive
    yaml = None  # type: ignore

USER_AGENT = "x402-endpoint-validator/1.0 (+https://smartflowproai.com/atlas)"
DEFAULT_TIMEOUT_S = 10
P95_SAMPLE_COUNT = 5
REQUIRED_402_KEYS = ("x402Version", "accepts", "error")
REQUIRED_ACCEPT_KEYS = ("scheme", "network")
PRICE_KEYS = ("maxAmountRequired", "amount", "price")


def log(msg: str) -> None:
    print(f"[x402-validator] {msg}", flush=True)


def parse_endpoints(raw: str, workspace: str) -> list[str]:
    """Resolve the `endpoints` input into a list of URLs.

    Accepts: a single URL string, a JSON array literal, or a path to a
    YAML/JSON file inside the workspace whose top-level value is either a
    list or an object with an `endpoints` key.
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("endpoints input is empty")

    if raw.startswith("http://") or raw.startswith("https://"):
        # Could still be space- or newline-separated multi-URL.
        parts = [p.strip() for p in raw.split() if p.strip()]
        return parts

    if raw.startswith("[") or raw.startswith("{"):
        data = json.loads(raw)
        return _coerce_list(data)

    candidate = os.path.join(workspace, raw)
    if os.path.isfile(candidate):
        with open(candidate, "r", encoding="utf-8") as f:
            text = f.read()
        if candidate.endswith((".yml", ".yaml")):
            if yaml is None:
                raise RuntimeError("PyYAML is required to read YAML configs")
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return _coerce_list(data)

    raise ValueError(f"could not interpret endpoints input: {raw!r}")


def _coerce_list(data: Any) -> list[str]:
    if isinstance(data, list):
        urls = data
    elif isinstance(data, dict) and "endpoints" in data:
        urls = data["endpoints"]
    else:
        raise ValueError("endpoints config must be a list or {endpoints: [...]}")
    out: list[str] = []
    for item in urls:
        if isinstance(item, str):
            out.append(item.strip())
        elif isinstance(item, dict) and "url" in item:
            out.append(str(item["url"]).strip())
        else:
            raise ValueError(f"unrecognised endpoint entry: {item!r}")
    return [u for u in out if u]


def origin_of(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def check_reachability(url: str) -> dict[str, Any]:
    started = time.monotonic()
    try:
        resp = requests.get(
            url,
            timeout=DEFAULT_TIMEOUT_S,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            allow_redirects=True,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        passed = resp.status_code < 500 and resp.status_code != 0
        return {
            "passed": passed,
            "status_code": resp.status_code,
            "response_time_ms": elapsed_ms,
            "note": "Non-5xx status received" if passed else "Server error",
        }
    except requests.RequestException as exc:
        return {
            "passed": False,
            "status_code": None,
            "response_time_ms": None,
            "note": f"network error: {exc.__class__.__name__}",
        }


def check_manifest(url: str) -> dict[str, Any]:
    manifest_url = origin_of(url) + "/.well-known/x402"
    try:
        resp = requests.get(
            manifest_url,
            timeout=DEFAULT_TIMEOUT_S,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
    except requests.RequestException as exc:
        return {
            "passed": False,
            "manifest_url": manifest_url,
            "note": f"network error: {exc.__class__.__name__}",
        }

    if resp.status_code != 200:
        return {
            "passed": False,
            "manifest_url": manifest_url,
            "status_code": resp.status_code,
            "note": "manifest not reachable (expected 200)",
        }

    try:
        data = resp.json()
    except ValueError:
        return {
            "passed": False,
            "manifest_url": manifest_url,
            "status_code": resp.status_code,
            "note": "manifest body is not valid JSON",
        }

    schema_ok = False
    schema_kind = "unknown"
    if isinstance(data, dict):
        if "accepts" in data and isinstance(data["accepts"], list):
            schema_ok = bool(data["accepts"])
            schema_kind = "v1-accepts"
        elif "resources" in data and "payment" in data:
            schema_ok = isinstance(data["resources"], list) and bool(data["resources"])
            schema_kind = "v2-provider-payment"
        elif "endpoints" in data and isinstance(data["endpoints"], list):
            schema_ok = bool(data["endpoints"])
            schema_kind = "legacy-endpoints"

    return {
        "passed": schema_ok,
        "manifest_url": manifest_url,
        "status_code": resp.status_code,
        "schema_kind": schema_kind,
        "x402_version": data.get("x402Version") if isinstance(data, dict) else None,
        "note": "manifest parsed and schema recognised" if schema_ok else "manifest schema not recognised",
    }


def check_402_body(url: str) -> dict[str, Any]:
    """POST with no X-PAYMENT header, expect HTTP 402 with conformant body."""
    try:
        resp = requests.post(
            url,
            timeout=DEFAULT_TIMEOUT_S,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json", "Content-Type": "application/json"},
            json={},
        )
    except requests.RequestException as exc:
        return {
            "passed": False,
            "payment_required_passed": False,
            "status_code": None,
            "note": f"network error: {exc.__class__.__name__}",
        }

    status = resp.status_code
    is_402 = status == 402

    body_ok = False
    missing_keys: list[str] = []
    accepts_ok = False
    body_note = ""
    parsed: Any = None
    try:
        parsed = resp.json()
    except ValueError:
        body_note = "402 body is not valid JSON"
    else:
        if not isinstance(parsed, dict):
            body_note = "402 body is not a JSON object"
        else:
            missing_keys = [k for k in REQUIRED_402_KEYS if k not in parsed]
            accepts = parsed.get("accepts")
            if isinstance(accepts, list) and accepts:
                accepts_ok = all(
                    isinstance(item, dict)
                    and all(k in item for k in REQUIRED_ACCEPT_KEYS)
                    and any(k in item for k in PRICE_KEYS)
                    for item in accepts
                )
            body_ok = not missing_keys and accepts_ok
            if not body_ok and not body_note:
                if missing_keys:
                    body_note = f"missing required keys: {missing_keys}"
                elif not accepts_ok:
                    body_note = "accepts[] entries missing scheme/network/price"

    return {
        "passed": is_402 and body_ok,
        "payment_required_passed": is_402,
        "status_code": status,
        "body_conformant": body_ok,
        "missing_keys": missing_keys,
        "x402_version": parsed.get("x402Version") if isinstance(parsed, dict) else None,
        "note": body_note or ("402 with conformant body" if (is_402 and body_ok) else "non-conformant"),
    }


def check_p95(url: str, threshold_ms: int) -> dict[str, Any]:
    samples: list[int] = []
    errors = 0
    for _ in range(P95_SAMPLE_COUNT):
        started = time.monotonic()
        try:
            resp = requests.get(
                url,
                timeout=DEFAULT_TIMEOUT_S,
                headers={"User-Agent": USER_AGENT},
            )
            _ = resp.status_code
        except requests.RequestException:
            errors += 1
            continue
        samples.append(int((time.monotonic() - started) * 1000))

    if not samples:
        return {
            "passed": False,
            "samples_ms": [],
            "p95_ms": None,
            "threshold_ms": threshold_ms,
            "errors": errors,
            "note": "no successful probes",
        }

    # statistics.quantiles requires n>=2; for small samples use the
    # interpolation-free "max of sample" as a conservative P95.
    if len(samples) >= 4:
        p95 = int(statistics.quantiles(samples, n=20)[18])
    else:
        p95 = max(samples)

    return {
        "passed": p95 <= threshold_ms,
        "samples_ms": samples,
        "p95_ms": p95,
        "threshold_ms": threshold_ms,
        "errors": errors,
        "note": f"P95={p95}ms (threshold {threshold_ms}ms)",
    }


def enhanced_check(url: str, api_key: str) -> dict[str, Any]:
    """Query Mapper API for paid-tier intel (wash flag, reputation, history)."""
    try:
        resp = requests.get(
            f"https://api.smartflowproai.com/v1/endpoints/{url}",
            headers={"X-API-Key": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "reputation_score": data.get("reputation_score"),
                "wash_flag": data.get("wash_flag"),
                "facilitator_mediated": data.get("is_facilitator_mediated"),
                "on_chain_volume_30d": data.get("on_chain_volume_usdc_30d"),
                "ok": True,
            }
        return {"ok": False, "error": f"api_status_{resp.status_code}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def validate_endpoint(url: str, threshold_ms: int, api_key: str = "") -> dict[str, Any]:
    log(f"validating {url}")
    reach = check_reachability(url)
    manifest = check_manifest(url)
    body = check_402_body(url)
    perf = check_p95(url, threshold_ms)
    endpoint_passed = all((
        reach["passed"],
        manifest["passed"],
        body["passed"],
        perf["passed"],
        body.get("payment_required_passed", False),
    ))
    result = {
        "url": url,
        "passed": endpoint_passed,
        "checks": {
            "reachability": reach,
            "manifest": manifest,
            "body_conformance": body,
            "response_time_p95": perf,
            "payment_required": {
                "passed": body.get("payment_required_passed", False),
                "status_code": body.get("status_code"),
                "note": "HTTP 402 returned" if body.get("payment_required_passed") else "did not return 402",
            },
        },
    }
    if api_key:
        enhanced = enhanced_check(url, api_key)
        result["enhanced"] = enhanced
        if enhanced.get("ok"):
            log(
                f"  enhanced: reputation={enhanced.get('reputation_score')} "
                f"wash_flag={enhanced.get('wash_flag')} "
                f"facilitator_mediated={enhanced.get('facilitator_mediated')} "
                f"on_chain_volume_30d={enhanced.get('on_chain_volume_30d')}"
            )
        else:
            log(f"  enhanced: lookup failed ({enhanced.get('error')})")
    return result


def print_upsell(api_key_present: bool) -> None:
    """Lead-funnel CTA appended to every Action run."""
    if api_key_present:
        return  # paid user, no need to upsell
    log("")
    log("⚡ Free tier scan complete.")
    log("⚡ Want wash detection, operator farm flags, endpoint reputation history, & cross-endpoint correlation?")
    log("⚡ Subscribe to Mapper API: https://hypersub.xyz/s/smartflow-scorecard ($15-$4999/mo)")
    log("⚡ Query directly: https://api.smartflowproai.com/v1/endpoints/{url}?key=YOUR_KEY")
    log("")


def maybe_post_webhook(webhook_url: str, report: dict[str, Any]) -> None:
    if not webhook_url:
        return
    summary = report["summary"]
    text = (
        f"x402 validator run: {summary['endpoints_checked']} checked, "
        f"{summary['failures']} failed, status={'PASS' if summary['all_passed'] else 'FAIL'}"
    )
    try:
        requests.post(
            webhook_url,
            timeout=DEFAULT_TIMEOUT_S,
            json={"text": text, "report_summary": summary},
            headers={"User-Agent": USER_AGENT},
        )
    except requests.RequestException as exc:
        log(f"webhook delivery failed: {exc}")


def decide_exit_code(report: dict[str, Any], fail_on: str) -> int:
    if fail_on == "never":
        return 0
    summary = report["summary"]
    if fail_on == "critical":
        critical = 0
        for ep in report["endpoints"]:
            checks = ep["checks"]
            if not checks["manifest"]["passed"] or not checks["body_conformance"]["passed"]:
                critical += 1
        return 1 if critical else 0
    # default: "any"
    return 0 if summary["all_passed"] else 1


def main() -> int:
    endpoints_raw = os.environ.get("X402V_ENDPOINTS", "")
    workspace = os.environ.get("X402V_WORKSPACE", os.getcwd())
    try:
        threshold_ms = int(os.environ.get("X402V_THRESHOLD_P95", "1000"))
    except ValueError:
        log("threshold-p95 must be an integer")
        return 2
    tier = (os.environ.get("X402V_TIER", "free") or "free").lower()
    pro_key = os.environ.get("X402V_PRO_LICENSE_KEY", "")
    webhook_url = os.environ.get("X402V_WEBHOOK_URL", "")
    report_path = os.environ.get("X402V_REPORT_PATH", "x402-validator-report.json")
    fail_on = (os.environ.get("X402V_FAIL_ON", "any") or "any").lower()
    api_key = os.environ.get("INPUT_API_KEY", "") or os.environ.get("X402V_API_KEY", "")

    if tier == "pro" and not pro_key:
        log("tier=pro requires pro-license-key — falling back to free tier")
        tier = "free"
        webhook_url = ""

    try:
        urls = parse_endpoints(endpoints_raw, workspace)
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        log(f"failed to parse endpoints: {exc}")
        return 2
    if not urls:
        log("no endpoints to validate")
        return 2

    log(f"tier={tier}, threshold_p95_ms={threshold_ms}, endpoints={len(urls)}, enhanced={'on' if api_key else 'off'}")

    results = [validate_endpoint(u, threshold_ms, api_key) for u in urls]
    failures = sum(1 for r in results if not r["passed"])
    summary = {
        "endpoints_checked": len(results),
        "failures": failures,
        "all_passed": failures == 0,
        "threshold_p95_ms": threshold_ms,
        "tier": tier,
        "enhanced_enabled": bool(api_key),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validator_version": "1.0.1",
    }
    report = {"summary": summary, "endpoints": results}

    # Ensure report directory exists.
    abs_report = os.path.abspath(report_path)
    os.makedirs(os.path.dirname(abs_report) or ".", exist_ok=True)
    with open(abs_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
    log(f"wrote report to {abs_report}")

    if tier == "pro":
        maybe_post_webhook(webhook_url, report)

    print_upsell(api_key_present=bool(api_key))

    return decide_exit_code(report, fail_on)


if __name__ == "__main__":
    sys.exit(main())
