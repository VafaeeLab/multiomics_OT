#!/usr/bin/env python3
"""Rewrite CITE ADT (mod2) feature names to RNA gene symbols.

For every dataset whose mod2 panel looks like an antibody (ADT) panel, this:
  1. normalizes each ADT name (strips Hu./HuMs./TotalSeq/clone-suffix/dedup-.N),
  2. maps it to an HGNC gene symbol via (a) exact match to an RNA gene, then
     (b) a curated CD-number/protein -> gene alias table,
  3. drops isotype controls / hashtags / titration / Fc reagents,
  4. writes a REWRITTEN mod2 features.tsv (same #rows, same order as the matrix)
     into --outdir using the SAME basename as the original, plus a per-dataset
     mapping CSV and a coverage summary.

The rewritten file is row-aligned with the matrix (names only change), so the
label-transfer tools' feature-name intersection picks up the now-matching genes;
unmapped/dropped antibodies keep a non-matching sentinel name and fall out.
Read-only on inputs. Idempotent.
"""
from __future__ import annotations
import argparse, csv, re
from pathlib import Path

# ---- curated antibody/CD-number -> HGNC gene symbol (only where the token is
#      NOT already a gene symbol; exact-gene matches are resolved against RNA). ----
ALIAS = {
    "CD3":"CD3E","CD8":"CD8A","CD10":"MME","CD11A":"ITGAL","CD11B":"ITGAM","CD11C":"ITGAX",
    "CD13":"ANPEP","CD15":"FUT4","CD16":"FCGR3A","CD18":"ITGB2","CD20":"MS4A1","CD202H7":"MS4A1",
    "CD21":"CR2","CD23":"FCER2","CD25":"IL2RA","CD26":"DPP4","CD29":"ITGB1","CD30":"TNFRSF8",
    "CD31":"PECAM1","CD32":"FCGR2A","CD35":"CR1","CD39":"ENTPD1","CD41":"ITGA2B","CD42B":"GP1BA",
    "CD43":"SPN","CD45":"PTPRC","CD45RA":"PTPRC","CD45RB":"PTPRC","CD45RO":"PTPRC","CD45RB220":"PTPRC",
    "CD49A":"ITGA1","CD49B":"ITGA2","CD49D":"ITGA4","CD49F":"ITGA6","CD51":"ITGAV","CD54":"ICAM1",
    "CD56":"NCAM1","CD57":"B3GAT1","CD61":"ITGB3","CD62L":"SELL","CD62P":"SELP","CD62E":"SELE",
    "CD64":"FCGR1A","CD66B":"CEACAM8","CD71":"TFRC","CD73":"NT5E","CD85G":"LILRA4","CD85J":"LILRB1",
    "CD88":"C5AR1","CD90":"THY1","CD94":"KLRD1","CD95":"FAS","CD98":"SLC3A2","CD100":"SEMA4D",
    "CD102":"ICAM2","CD103":"ITGAE","CD105":"ENG","CD106":"VCAM1","CD107A":"LAMP1","CD110":"MPL",
    "CD112":"NECTIN2","CD115":"CSF1R","CD116":"CSF2RA","CD117":"KIT","CD119":"IFNGR1","CD120A":"TNFRSF1A",
    "CD120B":"TNFRSF1B","CD122":"IL2RB","CD123":"IL3RA","CD124":"IL4R","CD126":"IL6R","CD127":"IL7R",
    "CD131":"CSF2RB","CD132":"IL2RG","CD133":"PROM1","CD134":"TNFRSF4","CD135":"FLT3","CD137":"TNFRSF9",
    "CD137L":"TNFSF9","CD138":"SDC1","CD140A":"PDGFRA","CD140B":"PDGFRB","CD141":"THBD","CD142":"F3",
    "CD144":"CDH5","CD146":"MCAM","CD150":"SLAMF1","CD152":"CTLA4","CD154":"CD40LG","CD155":"PVR",
    "CD158":"KIR2DL1","CD158B":"KIR2DL3","CD158E1":"KIR3DL1","CD158F":"KIR2DL5A","CD161":"KLRB1",
    "CD162":"SELPLG","CD169":"SIGLEC1","CD171":"L1CAM","CD172A":"SIRPA","CD178":"FASLG","CD179A":"VPREB1",
    "CD183":"CXCR3","CD184":"CXCR4","CD185":"CXCR5","CD186":"CXCR6","CD192":"CCR2","CD193":"CCR3",
    "CD194":"CCR4","CD195":"CCR5","CD196":"CCR6","CD197":"CCR7","CD198":"CCR8","CD199":"CCR9",
    "CD201":"PROCR","CD202B":"TEK","CD203C":"ENPP3","CD204":"MSR1","CD205":"LY75","CD206":"MRC1",
    "CD218A":"IL18R1","CD223":"LAG3","CD230":"PRNP","CD235A":"GYPA","CD235AB":"GYPA","CD243":"ABCB1",
    "CD252":"TNFSF4","CD253":"TNFSF10","CD254":"TNFSF11","CD257":"TNFSF13B","CD267":"TNFRSF13B",
    "CD268":"TNFRSF13C","CD269":"TNFRSF17","CD270":"TNFRSF14","CD271":"NGFR","CD272":"BTLA",
    "CD273":"PDCD1LG2","CD275":"ICOSLG","CD278":"ICOS","CD279":"PDCD1","CD284":"TLR4","CD294":"PTGDR2",
    "CD301":"CLEC10A","CD303":"CLEC4C","CD304":"NRP1","CD305":"LAIR1","CD307C":"FCRL3","CD307D":"FCRL4",
    "CD307E":"FCRL5","CD309":"KDR","CD314":"KLRK1","CD317":"BST2","CD319":"SLAMF7","CD324":"CDH1",
    "CD325":"CDH2","CD326":"EPCAM","CD328":"SIGLEC7","CD335":"NCR1","CD337":"NCR3","CD338":"ABCG2",
    "CD340":"ERBB2","CD352":"SLAMF6","CD354":"TREM1","CD357":"TNFRSF18","CD365":"HAVCR1","CD366":"HAVCR2",
    "CD370":"CLEC9A",
    # named (non-CD) reagents
    "B7H4":"VTCN1","C5L2":"C5AR2","FCERIA":"FCER1A","GARP":"LRRC32","GP130":"IL6ST","GPR56":"ADGRG1",
    "GALECTIN9":"LGALS9","HLAABC":"HLA-A","HLAA2":"HLA-A","HLADR":"HLA-DRA","HLADRDPDQ":"HLA-DRA",
    "INTEGRINB7":"ITGB7","LOX1":"OLR1","MICAMICB":"MICA","NKP80":"KLRF1","NOTCH1":"NOTCH1",
    "NOTCH3":"NOTCH3","PODOPLANIN":"PDPN","TCR":"TRAC","TCRAB":"TRAC","TCRGD":"TRDC","TCRVA7":"TRAV1-2",
    "TCRVA72":"TRAV1-2","TCRVD2":"TRDV2","TCRVG9":"TRGV9","TIM4":"TIMD4","TNFA":"TNF","TSLPR":"CRLF2",
    "VEGFR":"FLT4","VEGFR3":"FLT4","CADHERIN11":"CDH11","CADHERIN":"CDH11","FRB":"FOLR2",
    "MAC2":"LGALS3","MASTCELLTRYPTASE":"TPSAB1","IGLIGHTCHAINK":"IGKC","IGLIGHTCHAINL":"IGLC1",
    "IGM":"IGHM","IGD":"IGHD","IGE":"IGHE",
}
DROP_RE = re.compile(r"^(ISOTYPE|IGGFC|IGG1|IGG2|IGG2A|IGG2B|HASHTAG|TNC\d)", re.I)
ADT_SIG = re.compile(r"^(Hu\.|HuMs|HuMsRt|Ms\.|Isotype|Ig[MDGE]|CD\d|HLA[.\-]|TCR|integrin)", re.I)
PREFIX_RE = re.compile(r"^(anti[-_ ]?human|anti[-_ ]?mouse|HuMsRt|HuMs|Hu|Ms|Rt)[._ -]", re.I)
TOTALSEQ_RE = re.compile(r"[._ -]?TotalSeq[ABC]?\d*$", re.I)


def normalize_token(name: str):
    s = name.strip()
    s2 = PREFIX_RE.sub("", s)
    had_prefix = (s2 != s)
    s2 = TOTALSEQ_RE.sub("", s2)
    s2 = s2.split("_", 1)[0]            # drop clone suffix, e.g. CD14_M5E2
    if not had_prefix:
        s2 = re.sub(r"\.\d+$", "", s2)  # bare panels (datasets11/17) use .N as a dedup suffix
    return re.sub(r"[^A-Za-z0-9]", "", s2).upper(), had_prefix


def read_names(path: str):
    if not path or str(path).strip() in ("", ".", "nan", "None"):
        return None
    p = Path(path)
    if not p.exists() or p.is_dir():
        return None
    op = __import__("gzip").open if p.suffix == ".gz" else open
    out = []
    with op(p, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                out.append(re.split(r"\t", line)[-1].strip() if "\t" in line else line.strip())
    return out


def is_adt_panel(names):
    if not names or len(names) > 800:
        return False
    hits = sum(1 for n in names if ADT_SIG.match(n))
    return hits / len(names) >= 0.3


def remap_dataset(adt_names, rna_names):
    rna_upper = {}
    for g in rna_names:
        rna_upper.setdefault(g.upper(), g)
    rows, used = [], {}
    counts = {"gene_exact": 0, "alias": 0, "drop": 0, "unmapped": 0}
    for nm in adt_names:
        tok, _ = normalize_token(nm)
        if re.match(r"^(Isotype|TNC_\d|Hashtag)", nm, re.I) or DROP_RE.match(tok) or tok in ("", "ISOTYPE"):
            action, assigned = "drop", f"DROP__{nm}"
        elif tok in rna_upper:
            action, assigned = "gene_exact", rna_upper[tok]
        elif tok in ALIAS:
            g = ALIAS[tok]
            assigned = rna_upper.get(g.upper(), g)
            action = "alias"
        else:
            action, assigned = "unmapped", f"UNMAPPED__{nm}"
        # de-duplicate gene symbols so only one antibody claims each gene axis
        if action in ("gene_exact", "alias"):
            key = assigned.upper()
            if key in used:
                used[key] += 1
                assigned = f"{assigned}.dup{used[key]}"  # won't intersect RNA -> effectively dropped
            else:
                used[key] = 0
        counts[action] += 1
        in_rna = action in ("gene_exact",) or (action == "alias" and assigned.upper() in rna_upper)
        rows.append({"original": nm, "token": tok, "action": action,
                     "assigned": assigned, "in_rna": in_rna})
    return rows, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "adt_gene_alias_map.csv").write_text(
        "adt_token,gene_symbol\n" + "\n".join(f"{k},{v}" for k, v in sorted(ALIAS.items())) + "\n")

    summary = ["DATASET\tADT\tgene_exact\talias\tdrop\tunmapped\tmapped_in_rna"]
    for r in csv.DictReader(open(args.inventory)):
        ds = r.get("dataset_id", "?")
        adt = read_names(r.get("features2_path", "")); rna = read_names(r.get("features1_path", ""))
        if adt is None or rna is None or not is_adt_panel(adt):
            continue
        rows, counts = remap_dataset(adt, rna)
        bn = Path(r["features2_path"]).name
        if bn.endswith(".gz"):
            bn = bn[:-3]
        (outdir / bn).write_text("\n".join(x["assigned"] for x in rows) + "\n")
        with open(outdir / f"{ds}__adt_mapping.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["original", "token", "action", "assigned", "in_rna"])
            w.writeheader(); w.writerows(rows)
        mapped_in_rna = sum(1 for x in rows if x["in_rna"])
        summary.append(f"{ds}\t{len(adt)}\t{counts['gene_exact']}\t{counts['alias']}\t"
                       f"{counts['drop']}\t{counts['unmapped']}\t{mapped_in_rna}")
        print(f"[OK] {ds}: {mapped_in_rna}/{len(adt)} ADT -> RNA genes  "
              f"(exact {counts['gene_exact']}, alias {counts['alias']}, drop {counts['drop']}, "
              f"unmapped {counts['unmapped']})  -> {outdir/bn}")
    (outdir / "adt_remap_summary.tsv").write_text("\n".join(summary) + "\n")
    print(f"[DONE] summary -> {outdir/'adt_remap_summary.tsv'}")


if __name__ == "__main__":
    main()
