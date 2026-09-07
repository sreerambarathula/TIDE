import os
import re
import py_compile
import sys

count = 0
for root, _, files in os.walk("."):
    if any(x in root for x in [".venv", ".hpc_venv", ".git", "__pycache__"]):
        continue
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(root, f)
            with open(p, "r", encoding="utf-8", errors="ignore") as fp:
                text = fp.read()
            orig = text

            text = text.replace('.py"))]', '.py"))]')
            text = text.replace('.npz"))', '.npz"))')
            text = text.replace('.png")),', '.png")),')
            text = text.replace('figures"))', 'figures"))')
            text = text.replace('generate_fig3_ground_truth_data.py"))]', 'generate_fig3_ground_truth_data.py"))]')
            text = text.replace('generate_fig4_ground_truth_data.py"))]', 'generate_fig4_ground_truth_data.py"))]')

            if 'scripts' in p and 'test_attached_parity.py' in p:
                text = text.replace('"fig4_decoupling_data.npz"))\n', '"fig4_decoupling_data.npz"))\n')


            if text != orig:
                with open(p, "w", encoding="utf-8") as fp:
                    fp.write(text)
                count += 1
                print(f"Fixed {p}")

print(f"Total files updated: {count}")

# Now compile all python files to verify
errors = []
total_checked = 0
for root, _, files in os.walk("."):
    if any(x in root for x in [".venv", ".hpc_venv", ".git", "__pycache__"]):
        continue
    for f in files:
        if f.endswith(".py"):
            total_checked += 1
            p = os.path.join(root, f)
            try:
                py_compile.compile(p, doraise=True)
            except Exception as e:
                errors.append((p, str(e)))

print(f"\nVerification: Checked {total_checked} Python files.")
if errors:
    print(f"FOUND {len(errors)} SYNTAX ERRORS:")
    for p, err in errors:
        print(f"  {p}:\n    {err}")
    sys.exit(1)
else:
    print("ALL PYTHON FILES PASSED SYNTAX COMPILATION SUCCESSFULLY (0 ERRORS)!")

