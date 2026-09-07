import os
import re

count = 0
for root, _, files in os.walk("."):
    if ".venv" in root or ".git" in root:
        continue
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(root, f)
            with open(p, "r", encoding="utf-8", errors="ignore") as fp:
                text = fp.read()
            original = text
            # Replace os.path
            text = text.replace("os.path", "os.path")
            # Replace \ufeff
            text = text.replace("\ufeff", "")
            # Replace duplicate os.path.normpath(...))
            text = re.sub(r'os\.path\.normpath\(\s*os\.path\.normpath\(', 'os.path.normpath(', text)

            if text != original:
                with open(p, "w", encoding="utf-8") as fp:
                    fp.write(text)
                count += 1
                print(f"Fixed {p}")

print(f"Total files fixed: {count}")
