import sys
import json
import base64
import requests
from pathlib import Path

# Guard rail so a very large upload/response doesn't silently balloon
# memory across the base64 round-trips (upload -> request.json ->
# in-memory response -> result.json -> app.py decode). Adjust if the
# app needs to support larger files.
MAX_CONTENT_BYTES = 200 * 1024 * 1024  # 200 MB


def main():
    request_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])

    payload = json.loads(request_path.read_text(encoding="utf-8"))
    endpoint = payload["endpoint"]
    timeout = int(payload.get("timeout", 300))

    files = {}
    for field, item in payload.get("files", {}).items():
        content = base64.b64decode(item["content"])
        if len(content) > MAX_CONTENT_BYTES:
            result_path.write_text(
                json.dumps({
                    "ok": False,
                    "status_code": 0,
                    "content": "",
                    "content_type": "",
                    "error": (
                        f"File '{item.get('name', field)}' is too large "
                        f"({len(content) / (1024 * 1024):.1f} MB); the "
                        f"limit is {MAX_CONTENT_BYTES // (1024 * 1024)} MB."
                    ),
                }),
                encoding="utf-8",
            )
            return
        files[field] = (
            item["name"],
            content,
            item.get("mime", "text/csv")
        )

    try:
        response = requests.post(
            endpoint,
            files=files,
            data=payload.get("data", {}),
            timeout=timeout,
        )

        if len(response.content) > MAX_CONTENT_BYTES:
            result = {
                "ok": False,
                "status_code": response.status_code,
                "content": "",
                "content_type": "",
                "error": (
                    "The API response was too large "
                    f"({len(response.content) / (1024 * 1024):.1f} MB); "
                    f"the limit is {MAX_CONTENT_BYTES // (1024 * 1024)} MB."
                ),
            }
        else:
            result = {
                "ok": response.status_code == 200,
                "status_code": response.status_code,
                "content": base64.b64encode(response.content).decode("ascii"),
                "content_type": response.headers.get("content-type", ""),
                "error": "",
            }

            if response.status_code != 200:
                try:
                    result["error"] = response.json().get(
                        "detail", "Unknown API error"
                    )
                except Exception:
                    result["error"] = response.text

    except requests.exceptions.Timeout:
        result = {
            "ok": False,
            "status_code": 0,
            "content": "",
            "content_type": "",
            "error": "The request timed out.",
        }
    except requests.exceptions.ConnectionError:
        result = {
            "ok": False,
            "status_code": 0,
            "content": "",
            "content_type": "",
            "error": "Could not connect to the API.",
        }
    except Exception as exc:
        result = {
            "ok": False,
            "status_code": 0,
            "content": "",
            "content_type": "",
            "error": str(exc),
        }

    result_path.write_text(
        json.dumps(result),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()