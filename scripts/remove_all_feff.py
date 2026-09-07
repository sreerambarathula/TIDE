import os

count = 0
for root, _, files in os.walk("."):
    if ".venv" in root or ".git" in root:
        continue
    for f in files:
        if f.endswith((".py", ".sh", ".pbs", ".md", ".json", ".toml")):
            p = os.path.join(root, f)
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as fp:
                    text = fp.read()
                if "\ufeff" in text:
                    cleaned = text.replace("\ufeff", "")
                    with open(p, "w", encoding="utf-8") as fp:
                        fp.write(cleaned)
                    count += 1
                    print(f"Removed \\ufeff from {p}")
            except Exception as e:
                print(f"Error reading {p}: {e}")

print(f"Done. Cleaned {count} files.")
