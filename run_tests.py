# server.py integration tests — uses Python to avoid PowerShell JSON escaping issues
import urllib.request
import json
import sys

BASE = "http://localhost:8000"
passed = 0
failed = 0

def test(name, method, path, body=None, expected_status=200):
    global passed, failed
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else (b"this is not json" if body is None and "json" in name.lower() else None)
    # Simplified: pass data explicitly
    headers = {"Content-Type": "application/json"} if method == "POST" else {}
    req = urllib.request.Request(url, data=None, headers=headers, method=method)
    if method == "POST" and body is not None:
        req.data = json.dumps(body).encode()

    try:
        resp = urllib.request.urlopen(req)
        status = resp.status
        resp_body = resp.read().decode()
    except urllib.error.HTTPError as e:
        status = e.code
        resp_body = e.read().decode()

    tag = "PASS" if status == expected_status else "FAIL"
    if tag == "PASS":
        passed += 1
    else:
        failed += 1

    # Truncate long bodies
    display = resp_body[:200] + "..." if len(resp_body) > 200 else resp_body
    print(f"\n{'=' * 60}")
    print(f"[{tag}] {name}")
    print(f"  Expected: {expected_status}  |  Got: {status}")
    print(f"  Body: {display}")


# ---- Test 1: Health Check ----
test("Test 1: Health Check", "GET", "/api/health", expected_status=200)

# ---- Test 2: Serve UI ----
test("Test 2: Serve UI", "GET", "/", expected_status=200)

# ---- Test 5: Empty Body ----
test("Test 5: Empty Body → Validation Error", "POST", "/api/evaluate", body={}, expected_status=400)

# ---- Test 6: Bad URL format ----
test("Test 6: Bad URL Format", "POST", "/api/evaluate",
     body={"github_url": "not-a-url"}, expected_status=400)

# ---- Test 7: Bad Deployment Enum ----
test("Test 7: Bad Deployment Enum", "POST", "/api/evaluate",
     body={"model_profile": {"name": "Bot", "type": "chatbot", "use_case": "test", "deployment": "banana"}},
     expected_status=400)

# ---- Test 8: Missing Required Fields ----
test("Test 8: Missing Required Fields", "POST", "/api/evaluate",
     body={"model_profile": {"name": "Bot"}}, expected_status=400)

# ---- Test 9: Valid Manual Profile ----
test("Test 9: Valid Manual Profile (Happy Path)", "POST", "/api/evaluate",
     body={"model_profile": {
         "name": "TestBot", "type": "chatbot", "use_case": "customer support",
         "deployment": "cloud", "auth": "oauth", "finetune_data": "none", "logging": "enabled"
     }}, expected_status=200)

# ---- Test 10: Valid GitHub URL ----
test("Test 10: Valid GitHub URL (Happy Path)", "POST", "/api/evaluate",
     body={"github_url": "https://github.com/example/repo"}, expected_status=200)

# ---- Test 11: Invalid JSON body ----
# Send raw malformed text manually
req11 = urllib.request.Request(BASE + "/api/evaluate", data=b"this is not json",
                               headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req11)
    s11 = r.status; b11 = r.read().decode()
except urllib.error.HTTPError as e:
    s11 = e.code; b11 = e.read().decode()

tag11 = "PASS" if s11 == 400 else "FAIL"
if tag11 == "PASS": passed += 1
else: failed += 1
print(f"\n{'=' * 60}")
print(f"[{tag11}] Test 11: Invalid JSON Body")
print(f"  Expected: 400  |  Got: {s11}")
print(f"  Body: {b11[:200]}")

# ---- Test 12: Wrong HTTP Method ----
req12 = urllib.request.Request(BASE + "/api/evaluate", method="GET")
try:
    r = urllib.request.urlopen(req12)
    s12 = r.status; b12 = r.read().decode()
except urllib.error.HTTPError as e:
    s12 = e.code; b12 = e.read().decode()

tag12 = "PASS" if s12 == 405 else "FAIL"
if tag12 == "PASS": passed += 1
else: failed += 1
print(f"\n{'=' * 60}")
print(f"[{tag12}] Test 12: Wrong HTTP Method (GET /api/evaluate)")
print(f"  Expected: 405  |  Got: {s12}")
print(f"  Body: {b12[:200]}")

# ---- Summary ----
print(f"\n{'=' * 60}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
if failed > 0:
    sys.exit(1)
