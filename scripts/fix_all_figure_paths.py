import os
import re

count = 0
for root, _, files in os.walk("Figures"):
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(root, f)
            with open(p, "r", encoding="utf-8", errors="ignore") as fp:
                text = fp.read()
            orig = text

            # Fix rsys.executable
            text = text.replace("rsys.executable", "sys.executable")
            text = text.replace("r'sys.executable'", "sys.executable")
            text = text.replace('"sys.executable"', "sys.executable")

            # Fix triple closing parentheses: os.path.normpath(os.path.join(...))) -> os.path.normpath(os.path.join(...))
            text = re.sub(r'os\.path\.normpath\(\s*os\.path\.join\(([^)]+)\)\s*\)\s*\)', r'os.path.normpath(os.path.join(\1))', text)

            if text != orig:
                with open(p, "w", encoding="utf-8") as fp:
                    fp.write(text)
                count += 1
                print(f"Fixed {p}")

print(f"Total files fixed: {count}")
