#!/usr/bin/env python3
"""55_build_docx.py -- rebuild the submission .docx files via pandoc, then scrub
raw image filenames from the wp:docPr / pic:cNvPr descr attributes.

Reads paper1_submission.md and supplementary_data_S1.md from the project root
(two levels above this script) and writes the .docx next to them.

Implementation notes:
  * BASE is derived from the script location, so the tree is relocatable.
  * DESCR has 6 entries, one per figure (GA + Figures 1-5 or S1-S6).
  * expect_pics defaults to 6; the filename-leak assertion covers fig0-fig5 and
    allows digits (_300dpi).
  * descr ordering: pandoc emits two descr attributes per image (wp:docPr and
    pic:cNvPr). The script walks the document once, taking the same caption for
    each pair, so captions cannot rotate out of sync.
  * if the target file is locked (Word / previewer holds a handle), it falls
    back to writing *_REBUILT.docx rather than closing another process.
    不擅自关闭占用进程。
"""
import os
import re
import shutil
import subprocess
import zipfile

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DESCR = [
    "Graphical abstract: four-layer framework of the alignment-free dual-modal pipeline.",
    "Figure 1. Dual-modal supervision adds structural signal beyond the protein language model.",
    "Figure 2. Structural gain is confined to large, stability-coupled assays.",
    "Figure 3. External benchmarking against published zero-shot predictors and independent experiments.",
    "Figure 4. Diagnose-fix-validate cycle restoring graph learning to parity with the GBM.",
    "Figure 5. Practical decision map for when dual-modal supervision is worth its cost.",
]
SUPP_DESCR = [
    "Figure S1. Distributions of the nine dual-modal features on TEM-1.",
    "Figure S2. Collinearity among the four structural features.",
    "Figure S3. Per-variant agreement between ESM-2 650M and 3B zero-shot scores.",
    "Figure S4. Calibration and net benefit of the out-of-fold predictions.",
    "Figure S5. Complete GNN diagnostic experiments.",
    "Figure S6. Five-protein evaluation at the 3B backbone.",
]
PANDOC = shutil.which("pandoc") or r"C:\ProgramData\anaconda3\Library\bin\pandoc.exe"


def typeset_tables(xml, half_points=16, width_pct=5000):
    """Make tables readable in Word: text at 8 pt and the full text width.

    Pandoc emits table runs without a size, so they inherit 12 pt from Normal;
    in a seven-column table that wraps every cell over two or three lines. The
    width is set to 100 %% of the text column so the extra space is used.
    """
    def fix_run(m):
        r = m.group(0)
        r = re.sub(r'<w:sz w:val="\d+"/>', '<w:sz w:val="%d"/>' % half_points, r)
        r = re.sub(r'<w:szCs w:val="\d+"/>', '<w:szCs w:val="%d"/>' % half_points, r)
        if "<w:sz " not in r:
            rpr = '<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (half_points, half_points)
            if "<w:rPr>" in r:
                r = r.replace("<w:rPr>", "<w:rPr>" + rpr, 1)
            else:
                r = r.replace("<w:r>", "<w:r><w:rPr>" + rpr + "</w:rPr>", 1)
        return r

    def fix_table(m):
        t = m.group(0)
        if "<w:tblW" in t:
            t = re.sub(r'<w:tblW[^/]*/>', '<w:tblW w:w="%d" w:type="pct"/>' % width_pct, t, count=1)
        else:
            t = t.replace("<w:tblPr>", '<w:tblPr><w:tblW w:w="%d" w:type="pct"/>' % width_pct, 1)
        t = t.replace('<w:tblLayout w:type="fixed"/>', '<w:tblLayout w:type="autofit"/>')
        t = re.sub(r"<w:r>.*?</w:r>", fix_run, t, flags=re.S)
        return t

    return re.sub(r"<w:tbl>.*?</w:tbl>", fix_table, xml, flags=re.S)


def build(src, dst, expect_pics=6, descrs=None):
    lst = descrs or DESCR
    pandoc_out = os.path.join(BASE, dst + ".pandoc.docx")
    scrub_out = os.path.join(BASE, dst + ".scrub.docx")
    target = os.path.join(BASE, dst)

    # 1) pandoc 写临时文件，不直接碰可能被占用的目标
    subprocess.run([PANDOC, src, "-o", os.path.basename(pandoc_out),
                    "--reference-doc=bioinformatics_reference.docx",
                    "--resource-path=."], cwd=BASE, check=True)

    # 2) 按文档顺序替换 descr；每 2 个 descr 对应一张图
    with zipfile.ZipFile(pandoc_out) as zin, \
         zipfile.ZipFile(scrub_out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                xml = data.decode("utf-8")
                pat = re.compile(r'(<(?:wp:docPr|pic:cNvPr)[^>]*?)descr="[^"]*"')
                i = [0]

                def repl(m):
                    d = lst[(i[0] // 2) % len(lst)]
                    i[0] += 1
                    return m.group(1) + 'descr="' + d + '"'

                xml = pat.sub(repl, xml)
                xml = typeset_tables(xml)
                data = xml.encode("utf-8")
            zout.writestr(item, data)

    # 3) 落盘：目标被占用则回退到 _REBUILT
    final = target
    try:
        os.replace(scrub_out, target)
    except PermissionError:
        final = os.path.join(BASE, os.path.splitext(dst)[0] + "_REBUILT.docx")
        os.replace(scrub_out, final)
        print("WARN: %s 被其他进程独占，改写出 %s" % (dst, os.path.basename(final)))
    os.remove(pandoc_out)

    # 4) 校验
    with zipfile.ZipFile(final) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    n_pics = xml.count("<pic:pic>")
    n_descr = len(re.findall(r'descr="', xml))
    assert n_pics == expect_pics, "%s: %d pictures (expected %d)" % (
        os.path.basename(final), n_pics, expect_pics)
    for fn in ["fig0", "fig1", "fig2", "fig3", "fig4", "fig5"]:
        assert not re.search(fn + r"[A-Za-z0-9_]*\.png", xml), \
            "%s: raw %s filename leaked" % (dst, fn)
    # 6 条 DESCR 是否按图顺序落位
    for k, d in enumerate(lst[:expect_pics]):
        assert d[:24] in xml, "descr #%d missing" % k
    print("%s OK, %d pictures, %d descr slots, %d captions mapped, no raw filenames"
          % (os.path.basename(final), n_pics, n_descr, expect_pics))
    return final


if __name__ == "__main__":
    build("paper1_submission.md", "paper1_submission.docx")
    build("supplementary_data_S1.md", "supplementary_data_S1.docx",
          expect_pics=6, descrs=SUPP_DESCR)
