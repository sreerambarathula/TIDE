"""Sanitize all hardcoded absolute Windows paths across the repository to ensure
100% Linux HPC / cross-platform compatibility.
"""
import os
import re
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def sanitize_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original = content

    # 1. Replace hardcoded python executable invocations
    content = re.sub(
        r'[\"\'][dD]:[\\\\/]+AGravity[\\\\/]+Tide_Tutor[\\\\/]+\.venv[\\\\/]+Scripts[\\\\/]+python(?:\.exe)?[\"\']',
        'sys.executable',
        content
    )

    # 2. Determine relative expression to REPO_ROOT from this file
    rel = os.path.relpath(ROOT_DIR, os.path.dirname(file_path)).replace("\\", "/")
    
    # 3. Handle data/generated paths
    content = re.sub(
        r'[\"\'][dD]:[\\\\/]+AGravity[\\\\/]+Tide_Tutor[\\\\/]+data[\\\\/]+generated[\\\\/]+([a-zA-Z0-9_\-\.]+)[\"\']',
        r'os.path.normpath(os.path.join(os.path.dirname(__file__), "' + rel + r'", "data", "generated", "\1"))',
        content
    )
    content = re.sub(
        r'os\.path\.normpath\([\s\n]*r?[\"\'][dD]:[\\\\/]+AGravity[\\\\/]+Tide_Tutor[\\\\/]+data[\\\\/]+generated[\\\\/]+([a-zA-Z0-9_\-\.]+)[\"\'][\s\n]*\)',
        r'os.path.normpath(os.path.join(os.path.dirname(__file__), "' + rel + r'", "data", "generated", "\1"))',
        content
    )
    content = re.sub(
        r'os\.path\.normpath\([\s\n]*r?[\"\'][dD]:[\\\\/]+AGravity[\\\\/]+Tide_Tutor[\\\\/]+data[\\\\/]+generated[\"\'][\s\n]*\)',
        r'os.path.normpath(os.path.join(os.path.dirname(__file__), "' + rel + r'", "data", "generated"))',
        content
    )

    # 4. Handle manuscript/figures paths
    content = re.sub(
        r'[\"\'][dD]:[\\\\/]+AGravity[\\\\/]+Tide_Tutor[\\\\/]+manuscript[\\\\/]+figures[\\\\/]+([a-zA-Z0-9_\-\.]+)[\"\']',
        r'os.path.normpath(os.path.join(os.path.dirname(__file__), "' + rel + r'", "manuscript", "figures", "\1"))',
        content
    )
    content = re.sub(
        r'[\"\'][dD]:[\\\\/]+AGravity[\\\\/]+Tide_Tutor[\\\\/]+manuscript[\\\\/]+figures[\"\']',
        r'os.path.normpath(os.path.join(os.path.dirname(__file__), "' + rel + r'", "manuscript", "figures"))',
        content
    )
    content = re.sub(
        r'os\.path\.normpath\([\s\n]*r?[\"\'][dD]:[\\\\/]+AGravity[\\\\/]+Tide_Tutor[\\\\/]+manuscript[\\\\/]+figures[\\\\/]+([a-zA-Z0-9_\-\.]+)[\"\'][\s\n]*\)',
        r'os.path.normpath(os.path.join(os.path.dirname(__file__), "' + rel + r'", "manuscript", "figures", "\1"))',
        content
    )
    content = re.sub(
        r'os\.path\.normpath\([\s\n]*r?[\"\'][dD]:[\\\\/]+AGravity[\\\\/]+Tide_Tutor[\\\\/]+manuscript[\\\\/]+figures[\"\'][\s\n]*\)',
        r'os.path.normpath(os.path.join(os.path.dirname(__file__), "' + rel + r'", "manuscript", "figures"))',
        content
    )

    # 5. Handle Figures/figure_X paths
    content = re.sub(
        r'[\"\'][dD]:[\\\\/]+AGravity[\\\\/]+Tide_Tutor[\\\\/]+Figures[\\\\/]+(figure_[0-9]+|graphical_abstract)[\\\\/]+([a-zA-Z0-9_\-\.]+)[\"\']',
        r'os.path.normpath(os.path.join(os.path.dirname(__file__), "' + rel + r'", "Figures", "\1", "\2"))',
        content
    )
    content = re.sub(
        r'os\.path\.normpath\([\s\n]*r?[\"\'][dD]:[\\\\/]+AGravity[\\\\/]+Tide_Tutor[\\\\/]+Figures[\\\\/]+(figure_[0-9]+|graphical_abstract)[\\\\/]+([a-zA-Z0-9_\-\.]+)[\"\'][\s\n]*\)',
        r'os.path.normpath(os.path.join(os.path.dirname(__file__), "' + rel + r'", "Figures", "\1", "\2"))',
        content
    )
    content = re.sub(
        r'os\.path\.normpath\([\s\n]*r?[\"\'][dD]:[\\\\/]+AGravity[\\\\/]+Tide_Tutor[\\\\/]+Figures[\\\\/]+(figure_[0-9]+|graphical_abstract)[\"\'][\s\n]*\)',
        r'os.path.normpath(os.path.join(os.path.dirname(__file__), "' + rel + r'", "Figures", "\1"))',
        content
    )

    # 6. Handle scripts paths
    content = re.sub(
        r'[\"\'][dD]:[\\\\/]+AGravity[\\\\/]+Tide_Tutor[\\\\/]+scripts[\\\\/]+([a-zA-Z0-9_\-\.]+)[\"\']',
        r'os.path.normpath(os.path.join(os.path.dirname(__file__), "' + rel + r'", "scripts", "\1"))',
        content
    )
    content = re.sub(
        r'os\.path\.normpath\([\s\n]*r?[\"\'][dD]:[\\\\/]+AGravity[\\\\/]+Tide_Tutor[\\\\/]+scripts[\\\\/]+([a-zA-Z0-9_\-\.]+)[\"\'][\s\n]*\)',
        r'os.path.normpath(os.path.join(os.path.dirname(__file__), "' + rel + r'", "scripts", "\1"))',
        content
    )

    # 7. Handle docs and manuscript docs
    content = re.sub(
        r'[\"\'][dD]:[\\\\/]+AGravity[\\\\/]+Tide_Tutor[\\\\/]+(docs|manuscript)[\\\\/]+([a-zA-Z0-9_\-\.]+)[\"\']',
        r'os.path.normpath(os.path.join(os.path.dirname(__file__), "' + rel + r'", "\1", "\2"))',
        content
    )

    # If changes made, ensure 'import sys' and 'import os' exist in file
    if content != original:
        if "import sys" not in content:
            content = "import sys\n" + content
        if "import os" not in content:
            content = "import os\n" + content

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False

def main():
    modified = []
    for base in ["Figures", "scripts", "src", "tests"]:
        base_dir = os.path.join(ROOT_DIR, base)
        if not os.path.exists(base_dir):
            continue
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".py"):
                    full_p = os.path.join(root, file)
                    if sanitize_file(full_p):
                        modified.append(os.path.relpath(full_p, ROOT_DIR))
    
    print(f"Sanitized {len(modified)} files for cross-platform execution:")
    for m in modified:
        print(f"  - {m}")

if __name__ == "__main__":
    main()
