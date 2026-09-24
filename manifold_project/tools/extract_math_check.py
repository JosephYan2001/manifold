import argparse
import re
from pathlib import Path


project = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Extract Markdown formulas for local TeX checking.")
parser.add_argument("source", nargs="?", type=Path, help="Optional Markdown path; defaults to the CTDE direction manuscript.")
args = parser.parse_args()
source = args.source or project / "docs/策略流形研究设计.CTDE本地策略改进版.md"
markdown = source.read_text(encoding="utf-8")
display_math = re.findall(r"\$\$(.*?)\$\$", markdown, flags=re.DOTALL)
without_display = re.sub(r"\$\$.*?\$\$", "", markdown, flags=re.DOTALL)
inline_math = re.findall(
    r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)",
    without_display,
    flags=re.DOTALL,
)

parts = [
    r"\documentclass{ctexart}",
    r"\usepackage{amsmath,amssymb,mathtools,bm,mathrsfs}",
    r"\begin{document}",
]
for index, formula in enumerate(display_math, start=1):
    parts.extend([rf"\noindent Display {index}:\par", r"\[", formula.strip(), r"\]"])
for index, formula in enumerate(inline_math, start=1):
    parts.extend(
        [rf"\noindent Inline {index}:\par", r"\[", r"\displaystyle " + formula.strip(), r"\]"]
    )
parts.append(r"\end{document}")

output = project / "build/mathcheck.tex"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(parts), encoding="utf-8")
print(f"display={len(display_math)} inline={len(inline_math)}")
