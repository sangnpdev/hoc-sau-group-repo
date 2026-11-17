"""Ghép báo cáo và slide từ template + bảng/hình/số liệu sinh tự động, rồi xuất .docx/.pptx/.pdf.

Placeholder trong template:
  {{TAB:A.md|chú thích}}         → "Bảng N. chú thích" + nội dung results/tables/A.md
  {{FIG:A_error.png|chú thích}}  → hình results/figures/A_error.png, chú thích "Hình N. …"
  {{S:A:200:test_rmse}}          → giá trị mean trong results/A_summary.csv tại dòng khoá = 200
  {{S:A:200:test_rmse:std}}      → độ lệch chuẩn tương ứng
  {{J:final_selection:test_rmse}}→ giá trị trong results/final_selection.json
Yêu cầu: pandoc, xelatex (báo cáo PDF), libreoffice (PDF của slide) trong PATH.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src.experiments import GROUP_KEYS, RESULTS_DIR  # noqa: E402

import pandas as pd  # noqa: E402

FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TAB_DIR = os.path.join(RESULTS_DIR, "tables")


def fmt(v, col):
    if isinstance(v, str):
        return v
    if col.endswith("r2"):
        return f"{v:.7f}" if v > 0.9999 else f"{v:.4f}"
    if any(col.endswith(k) for k in ("mse", "rmse", "mae")) or (isinstance(v, float) and 0 < abs(v) < 1e-2):
        return f"{v:.2e}"
    if isinstance(v, float) and not float(v).is_integer():
        return f"{v:.3g}" if abs(v) < 100 else f"{v:.0f}"
    return str(int(v)) if isinstance(v, float) else str(v)


def lookup_summary(group, key, col, stat="mean"):
    s = pd.read_csv(os.path.join(RESULTS_DIR, f"{group}_summary.csv"))
    kcols = GROUP_KEYS[group]
    keys = key.split(",")
    mask = pd.Series(True, index=s.index)
    for kc, kv in zip(kcols, keys):
        mask &= s[kc].astype(str) == kv
    row = s[mask]
    assert len(row) == 1, (group, key, col)
    return fmt(float(row.iloc[0][f"{col}_{stat}"]), col)


def lookup_json(name, key):
    with open(os.path.join(RESULTS_DIR, f"{name}.json")) as f:
        v = json.load(f)[key]
    return fmt(v, key) if not isinstance(v, list) else "-".join(map(str, v))


def number_headings(text):
    """Đánh số '# ' và '## ' (bỏ qua khối code và front matter) để .docx và .pdf có cùng số mục
    (pandoc 2.9 không đánh số mục cho .docx)."""
    out, h1, h2, in_code, in_yaml = [], 0, 0, False, False
    for i, line in enumerate(text.split("\n")):
        if line.strip() == "---" and (i == 0 or in_yaml):
            in_yaml = not in_yaml
        elif line.startswith("```"):
            in_code = not in_code
        elif not in_code and not in_yaml and line.startswith("# "):
            h1, h2 = h1 + 1, 0
            line = f"# {h1} {line[2:]}"
        elif not in_code and not in_yaml and line.startswith("## "):
            h2 += 1
            line = f"## {h1}.{h2} {line[3:]}"
        out.append(line)
    return "\n".join(out)


def render(text, fig_width="15cm", slide=False):
    counters = {"fig": 0, "tab": 0}
    if not slide:
        text = number_headings(text)
    # {{S:tune:BEST:...}} → tổ hợp được chọn ghi trong baseline.json (không gõ cứng kết quả tinh chỉnh)
    with open(os.path.join(RESULTS_DIR, "baseline.json")) as f:
        b = json.load(f)
    text = text.replace("{{S:tune:BEST:", f"{{{{S:tune:{b['lr']:g},{b['activation']},{b['batch_size']}:")
    # Đánh số trước theo thứ tự xuất hiện để {{REF:tên_file}} trả về số Hình/Bảng tương ứng
    numbers, nf, nt = {}, 0, 0
    for kind, name in re.findall(r"\{\{(FIG|TAB):([^|}]+)\|", text):
        if kind == "FIG":
            nf += 1; numbers[name] = nf
        else:
            nt += 1; numbers[name] = nt
    text = re.sub(r"\{\{REF:([^}]+)\}\}", lambda m: str(numbers[m.group(1)]), text)

    def tab(m):
        name, cap = m.group(1), m.group(2)
        counters["tab"] += 1
        body = open(os.path.join(TAB_DIR, name)).read()
        if slide:
            return body
        return f"**Bảng {counters['tab']}.** {cap}\n\n{body}\n"

    def fig(m):
        name, cap, width = m.group(1), m.group(2), m.group(3) or fig_width
        counters["fig"] += 1
        path = os.path.join(FIG_DIR, name)
        assert os.path.exists(path), path
        if slide:
            return f"![{cap}]({path})"
        return f"![Hình {counters['fig']}. {cap}]({path}){{width={width}}}"

    text = re.sub(r"\{\{TAB:([^|}]+)\|([^}]*)\}\}", tab, text)
    text = re.sub(r"\{\{FIG:([^|}]+)\|([^|}]*)(?:\|([^}]*))?\}\}", fig, text)
    text = re.sub(r"\{\{S:([^:}]+):([^:}]+):([^:}]+)(?::(mean|std))?\}\}",
                  lambda m: lookup_summary(m.group(1), m.group(2), m.group(3), m.group(4) or "mean"), text)
    text = re.sub(r"\{\{J:([^:}]+):([^}]+)\}\}", lambda m: lookup_json(m.group(1), m.group(2)), text)
    left = re.findall(r"\{\{[^}]*\}\}", text)
    assert not left, f"placeholder chưa thay: {left}"
    return text


TEXT_WIDTH_TWIPS = 9071  # A4 (11906) trừ lề trái 3 cm (1701) và lề phải 2 cm (1134)


def make_reference_docx(path):
    """reference.docx cho pandoc: Times New Roman 13pt, khổ A4, lề 3/2/2/2 cm."""
    raw = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"], check=True,
                         capture_output=True).stdout
    tmp = path + ".tmp"
    open(tmp, "wb").write(raw)
    with zipfile.ZipFile(tmp) as zin, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            buf = zin.read(item.filename)
            if item.filename == "word/styles.xml":
                s = buf.decode("utf8")
                s = re.sub(r"<w:rFonts [^>]*/>",
                           '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
                           'w:eastAsia="Times New Roman" w:cs="Times New Roman"/>', s)
                s = re.sub(r'<w:sz w:val="24" ?/>', '<w:sz w:val="26"/>', s)
                buf = s.encode("utf8")
            elif item.filename == "word/document.xml":
                s = buf.decode("utf8")
                s = re.sub(r"<w:pgSz [^>]*/>", '<w:pgSz w:w="11906" w:h="16838"/>', s)
                s = re.sub(r"<w:pgMar [^>]*/>", '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" '
                           'w:left="1701" w:header="708" w:footer="708" w:gutter="0"/>', s)
                buf = s.encode("utf8")
            zout.writestr(item, buf)
    os.remove(tmp)


def _cell_text(tc):
    return "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", tc))


def fix_docx_tables(path):
    """pandoc 2.9 ghi bảng với tblGrid rỗng và độ rộng 0 → một số trình đọc (LibreOffice) chỉ hiện vài cột.
    Ghi lại tblGrid với độ rộng cột tỉ lệ theo độ dài nội dung, bảng rộng 100% vùng văn bản."""
    tmp = path + ".tmp"
    shutil.move(path, tmp)
    with zipfile.ZipFile(tmp) as zin, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            buf = zin.read(item.filename)
            if item.filename == "word/document.xml":
                doc = buf.decode("utf8")

                def fix(m):
                    tbl = m.group(0)
                    rows = re.findall(r"<w:tr>.*?</w:tr>", tbl, re.S)
                    cells = [re.findall(r"<w:tc>.*?</w:tc>", r, re.S) for r in rows]
                    ncol = max(len(c) for c in cells)
                    lens = [max([min(len(_cell_text(r[j])), 40) for r in cells if j < len(r)] + [3]) for j in range(ncol)]
                    widths = [int(TEXT_WIDTH_TWIPS * l / sum(lens)) for l in lens]
                    grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>"
                    tbl = tbl.replace("<w:tblGrid />", grid)
                    tbl = tbl.replace('<w:tblW w:type="pct" w:w="0.0" />', '<w:tblW w:type="pct" w:w="5000"/>')
                    return tbl
                doc = re.sub(r"<w:tbl>.*?</w:tbl>", fix, doc, flags=re.S)
                buf = doc.encode("utf8")
            zout.writestr(item, buf)
    os.remove(tmp)


LATEX_HEADER = r"""
\usepackage{float}
\floatplacement{figure}{H}
\usepackage{caption}
\captionsetup[figure]{labelformat=empty,font=small}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small}
\AtBeginEnvironment{Shaded}{\small}
\setlength{\parskip}{4pt}
"""


def main():
    out_dir = HERE
    ref = os.path.join(HERE, "reference.docx")
    make_reference_docx(ref)

    md = render(open(os.path.join(HERE, "report_template.md")).read())
    md_path = os.path.join(out_dir, "BaoCao_MLP.md")
    open(md_path, "w").write(md)
    common = ["--toc", "--toc-depth=2", "-M", "toc-title=Mục lục"]   # số mục do number_headings() đánh

    docx = os.path.join(out_dir, "BaoCao_MLP.docx")
    subprocess.run(["pandoc", md_path, "-o", docx, "--reference-doc", ref, *common], check=True)
    fix_docx_tables(docx)

    hdr = os.path.join(out_dir, "latex_header.tex")
    open(hdr, "w").write(LATEX_HEADER)
    subprocess.run(["pandoc", md_path, "-o", os.path.join(out_dir, "BaoCao_MLP.pdf"), "--pdf-engine=xelatex",
                    "-H", hdr, "-V", "mainfont=Liberation Serif", "-V", "monofont=Noto Sans Mono",
                    "-V", "fontsize=12pt", "-V", "papersize=a4", "-V", "geometry=left=3cm,right=2cm,top=2cm,bottom=2cm",
                    "-V", "colorlinks=true", "-V", "linkcolor=black", "-V", "toccolor=black", *common], check=True)

    sl = render(open(os.path.join(HERE, "slides_template.md")).read(), slide=True)
    sl_path = os.path.join(out_dir, "Slide_MLP.md")
    open(sl_path, "w").write(sl)
    pptx = os.path.join(out_dir, "Slide_MLP.pptx")
    subprocess.run(["pandoc", sl_path, "-o", pptx, "--slide-level=2"], check=True)
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", out_dir, pptx], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("built:", sorted(f for f in os.listdir(out_dir) if f.split(".")[-1] in ("docx", "pdf", "pptx")))


if __name__ == "__main__":
    for tool in ("pandoc", "soffice", "xelatex"):
        if not shutil.which(tool):
            sys.exit(f"thiếu {tool}: cần pandoc, libreoffice và xelatex (TeX Live)")
    main()
