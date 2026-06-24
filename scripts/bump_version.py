#!/usr/bin/env python3
import re
import sys
import subprocess
import os

def main():
    # 1. Get latest commit message to see if there's an override command like [major] or [minor]
    try:
        commit_msg = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%B"],
            text=True
        ).strip().lower()
    except Exception as e:
        print(f"Warning: Could not get git commit message: {e}", file=sys.stderr)
        commit_msg = ""
        
    bump_type = "patch"
    if "[major]" in commit_msg or "#major" in commit_msg:
        bump_type = "major"
    elif "[minor]" in commit_msg or "#minor" in commit_msg:
        bump_type = "minor"

    # 2. Read pyproject.toml
    pyproject_path = "pyproject.toml"
    if not os.path.exists(pyproject_path):
        print(f"Error: {pyproject_path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(pyproject_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Match version = "x.y.z"
    version_regex = r'^version\s*=\s*"([^"]+)"'
    match = re.search(version_regex, content, re.MULTILINE)
    if not match:
        print("Error: Could not find version in pyproject.toml", file=sys.stderr)
        sys.exit(1)

    old_version = match.group(1)
    parts = list(map(int, old_version.split(".")))
    if len(parts) != 3:
        print(f"Error: Version {old_version} is not valid semver (x.y.z)", file=sys.stderr)
        sys.exit(1)

    # 3. Calculate new version
    if bump_type == "major":
        parts[0] += 1
        parts[1] = 0
        parts[2] = 0
    elif bump_type == "minor":
        parts[1] += 1
        parts[2] = 0
    else:
        parts[2] += 1

    new_version = ".".join(map(str, parts))
    print(f"Bumping version from {old_version} to {new_version} ({bump_type})")

    # 4. Replace version in content and write back
    new_content = re.sub(
        r'(^version\s*=\s*")([^"]+)(")',
        rf'\g<1>{new_version}\g<3>',
        content,
        flags=re.MULTILINE
    )

    with open(pyproject_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    # Output new version so GitHub Actions can use it if needed
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as env_file:
            env_file.write(f"new_version={new_version}\n")
            env_file.write(f"old_version={old_version}\n")

if __name__ == "__main__":
    main()
