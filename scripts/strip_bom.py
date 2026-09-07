import os

count = 0
for root, _, files in os.walk("."):
    if ".venv" in root or ".git" in root:
        continue
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(root, f)
            with open(p, "rb") as fp:
                data = fp.read()
            if data.startswith(b"\xef\xbb\xbf"):
                with open(p, "wb") as fp:
                    fp.write(data[3:])
                count += 1
                print(f"Stripped BOM from {p}")

print(f"Finished cleaning BOM. Cleaned {count} files.")
