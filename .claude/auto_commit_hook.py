import json
import subprocess
import sys

REPO = "e:/workspace/MemISTD"


def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    tool_input = payload.get("tool_input", {}) or {}
    tool_response = payload.get("tool_response", {}) or {}

    file_path = tool_input.get("file_path") or tool_response.get("filePath")
    if not file_path:
        return 0

    file_path = file_path.replace("\\", "/")
    repo = REPO.replace("\\", "/")

    rel_path = file_path[len(repo) + 1 :] if file_path.startswith(repo + "/") else file_path

    if run(["git", "-C", REPO, "rev-parse", "--is-inside-work-tree"]).returncode != 0:
        return 0

    if run(["git", "-C", REPO, "add", "--", rel_path]).returncode != 0:
        return 0

    has_no_change = run(["git", "-C", REPO, "diff", "--cached", "--quiet", "--", rel_path]).returncode == 0
    if has_no_change:
        return 0

    run(["git", "-C", REPO, "commit", "-m", "chore: auto-sync after edit"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
