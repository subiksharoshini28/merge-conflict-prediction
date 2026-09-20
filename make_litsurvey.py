# Generates an editable .docx Literature Survey for the Merge Conflict Predictor,
# matching the reference format (Title / Team / Patent table / Existing Problem / Proposed Solution).
import zipfile, os

def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def run(text, bold=False, size=22, color=None):
    rpr = ""
    if bold: rpr += "<w:b/>"
    if color: rpr += f'<w:color w:val="{color}"/>'
    rpr += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    return f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r>'

def para(runs, size=22, after=120, before=0, align=None, ind=0):
    ppr = f'<w:spacing w:before="{before}" w:after="{after}"/>'
    if align: ppr += f'<w:jc w:val="{align}"/>'
    if ind: ppr += f'<w:ind w:left="{ind}"/>'
    if isinstance(runs, str):
        runs = [(runs, False)]
    body = "".join(run(t, b, size) for t, b in runs)
    return f'<w:p><w:pPr>{ppr}</w:pPr>{body}</w:p>'

def heading(text, size=26, before=240):
    return para([(text, True)], size=size, before=before, after=100)

def bullet(text, size=22):
    return para([("•  ", False), (text, False)], size=size, after=60, ind=360)

def cell(paras_html, width, shade=None):
    shd = f'<w:shd w:val="clear" w:color="auto" w:fill="{shade}"/>' if shade else ""
    return (f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shd}'
            f'<w:tcMar><w:top w:w="60" w:type="dxa"/><w:left w:w="90" w:type="dxa"/>'
            f'<w:bottom w:w="60" w:type="dxa"/><w:right w:w="90" w:type="dxa"/></w:tcMar>'
            f'</w:tcPr>{paras_html}</w:tc>')

def tcell(text, width, bold=False, shade=None, size=19):
    return cell(para([(text, bold)], size=size, after=40), width, shade)

W = (1650, 3855, 3855)  # column widths (dxa)

def table(rows):
    border = ('<w:tblBorders>'
              + "".join(f'<w:{s} w:val="single" w:sz="6" w:space="0" w:color="808080"/>'
                        for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
              + '</w:tblBorders>')
    grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in W) + "</w:tblGrid>"
    trs = ""
    for cells in rows:
        trs += "<w:tr>" + "".join(cells) + "</w:tr>"
    return (f'<w:tbl><w:tblPr><w:tblW w:w="9360" w:type="dxa"/>{border}'
            f'<w:tblLook w:val="04A0"/></w:tblPr>{grid}{trs}</w:tbl>')

# ---------------- CONTENT ----------------
TITLE = ("AI-Driven Predictive Merge Conflict Detection Framework Using Dependency "
         "Graphs and Explainable Artificial Intelligence")

TEAM = ["1. Subiksha Roshini Durai Murugan (Reg. No. __________)",
        "2. Tamizhiniyan K. (Reg. No. __________)"]

PATENTS = [
 ("US 11,150,893 B2",
  "Discloses a collaborative software-development tool that detects potential code-change conflicts between developers working concurrently on a shared codebase and notifies them in real time during development, before an explicit merge is attempted.",
  "Closely related in providing early, pre-merge awareness of conflicts in collaborative development. However, it operates by monitoring concurrent edits rather than modelling code dependencies, produces notifications rather than a calibrated conflict probability or risk score, cannot detect dependency-mediated conflicts between files that are never edited together, and offers no explainable justification or recommended merge strategy."),
 ("US 11,347,500 B2",
  "Describes a machine-learning system that uses code vector representations to detect code conflicts and redundancies during code development and to assist in resolving them.",
  "Relevant for applying learned code representations to conflict detection. However, it focuses on textual/semantic code similarity and redundancy rather than the structural dependency relationships between Git branches, performs detection during development rather than forecasting conflict risk before merging two branches, and provides no dependency-graph analysis or explainable risk scoring."),
 ("US 12,541,362 B2",
  "Presents advanced computation models that analyse planned code changes to detect scheduling conflicts within software-development workflows.",
  "Related in reasoning about conflicts among planned changes, but it targets workflow/scheduling conflicts rather than integration (merge) conflicts, does not construct a software dependency graph, and produces no predictive merge-risk estimate or explainable per-decision breakdown."),
 ("CN 117931275 A",
  "Discloses a machine-learning method that learns from historical merge conflicts in order to automatically resolve future merge conflicts.",
  "The closest prior art, sharing the use of historical merge data and machine learning. However, it acts after or during a merge to resolve conflicts that have already occurred, whereas the proposed invention predicts the probability of a conflict BEFORE the merge is attempted using dependency-graph features, and issues an explainable risk score together with a recommended action."),
 ("CN 117827271 A",
  "Uses contextual code representations and pretrained language models to automatically generate merge-conflict resolutions at multiple (hybrid) granularities.",
  "Relevant as an automated conflict-resolution approach. However, it is concerned with generating resolutions for conflicts that have already materialised, does not model inter-file architectural dependencies across branches, and performs neither proactive pre-merge prediction nor explainable risk scoring."),
 ("US 2015/0220332 A1",
  "Applies rule-based merge rules and validation techniques to automatically resolve conflicts that prevent blocks of program code from being properly merged.",
  "Related to handling merge conflicts, but it is a rule-based resolution mechanism triggered at merge time, with no machine-learning prediction, no dependency-graph analysis, and no CI/CD-aware prioritisation or explainable output."),
 ("US 8,683,443 B2",
  "Detects software integration conflicts, blocks conflicting changes, and notifies developers before integration in a streamlined workflow.",
  "Relevant for detecting integration conflicts prior to integration. However, it provides binary detection and blocking rather than a calibrated conflict probability or risk score, does not analyse dependency graphs or software metrics, and gives developers no explainable reasoning or AI-based merge-strategy recommendation."),
 ("Crystal / Speculative Merging (Brun et al., ESEC/FSE 2011) — academic",
  "An academic approach that continuously and speculatively performs branch merges (and builds/tests) in the background to proactively surface textual, build, and test conflicts before developers manually merge.",
  "Directly relevant to proactive, pre-merge conflict detection. However, it is reactive and computationally expensive because it must speculatively execute the actual merge and build for every branch pair; it does not predict conflict likelihood from lightweight dependency-graph features, scales poorly with many branches, and offers no explainable risk score or recommended action."),
 ("Socio-technical merge-conflict prediction (ML over Git-history metrics) — academic",
  "A body of academic work that trains classifiers on flat Git-history and socio-technical metrics (commit counts, changed files, developer overlap, file overlap) to estimate the likelihood of a future merge conflict.",
  "The most direct methodological precursor, sharing pre-merge prediction from history features. However, such models rely only on flat overlap/activity metrics and are therefore blind to dependency-mediated conflicts between branches that share no file; they typically report accuracy on an imbalanced dataset, and provide neither dependency-graph structural features nor explainable, actionable output — the specific gaps this invention closes."),
]

PROBLEM_INTRO = ("Current approaches to handling software merge conflicts in collaborative, version-controlled "
 "development predominantly rely on detecting conflicts during or after the merge (e.g., three-way textual "
 "diff or git merge-tree), rule-based automatic resolution, or single-modality machine-learning classifiers "
 "trained on flat Git-history metrics. Although such methods can identify or resolve conflicts once they arise, "
 "they exhibit several technical limitations:")

PROBLEMS = [
 ("Detection Only During or After the Merge",
  ["Most existing systems surface a conflict only when the merge is actually attempted, by which point the "
   "integration is already blocked and developer time is lost.",
   "They act as detectors, not predictors, providing no advance warning to schedule or route work."]),
 ("Absence of Predictive Capability",
  ["Available systems perform post-hoc detection rather than forward-looking risk estimation.",
   "They cannot forecast rising conflict risk from early, pre-merge signals before two branches are combined."]),
 ("Flat, Single-Modality Feature Analysis",
  ["Prediction models typically use only flat Git-history metrics (number of commits, changed files, churn, "
   "file overlap) in isolation.",
   "Such features ignore the architectural dependency structure of the code and therefore capture only part of "
   "the true conflict risk."]),
 ("Blindness to Dependency-Mediated (No-Shared-File) Conflicts",
  ["A semantic conflict can arise when two branches modify different, dependency-linked files (e.g., one renames "
   "a method, another still calls it), producing code that merges cleanly yet fails to build.",
   "Flat overlap features are structurally incapable of seeing such conflicts because the two branches share no file."]),
 ("Mishandled Class Imbalance and Misleading Evaluation",
  ["Conflicts are a small minority of merges, yet many systems are evaluated on overall accuracy, which a "
   "trivial predict-clean model can inflate above 88% while catching no conflicts at all.",
   "This yields models that appear accurate but are operationally useless for the conflict class."]),
 ("Black-Box Decision Making",
  ["Deep-learning conflict/phishing-style classifiers commonly output an opaque score without indicating which "
   "feature or dependency drove the decision.",
   "An unexplained score gives developers no basis to trust, verify, or act on the prediction."]),
 ("No Actionable Guidance",
  ["Existing tools, at best, flag a conflict but do not recommend how to proceed (e.g., a merge ordering, a "
   "coordination point, or which module to test).",
   "The burden of deciding what to do remains entirely with the developer."]),
 ("Costly Speculative Merging",
  ["Proactive academic tools that speculatively merge and build every branch pair are computationally heavy and "
   "scale poorly as the number of active branches grows.",
   "They also provide a binary result with no probability, ranking, or explanation."]),
]

SOLUTION_INTRO = ("The proposed invention, “AI-Driven Predictive Merge Conflict Detection Framework Using "
 "Dependency Graphs and Explainable Artificial Intelligence,” introduces a point-in-time, dependency-graph-aware "
 "predictive architecture that estimates the probability of a merge conflict BEFORE two branches are merged, and "
 "accompanies every prediction with an explainable risk score and a recommended action.")

SOLUTIONS = [
 ("Point-in-Time, Leakage-Free Feature Extraction at the Merge Base",
  "Every feature is computed at the common-ancestor (merge-base) commit of the two branches by reading stored "
  "Git objects, without checking out or mutating the working tree. This guarantees the prediction uses only "
  "information available before the merge and prevents any leakage of the merge outcome into the features."),
 ("Dependency-Graph Structural Sensing",
  "A file-level dependency graph is built from import/reference declarations at the merge base, and cross-branch "
  "interaction features are derived — cross-edges, minimum graph distance, shared neighbours, shared modules, "
  "and fan-in/fan-out — that reveal dependency-mediated conflicts even when the two branches share no common file."),
 ("Software-Metric Fusion",
  "Established software metrics — Coupling Between Objects (CBO), Cyclomatic Complexity, Dependency Depth, code "
  "churn, and change coupling — are fused with the git-history and graph features so the model reasons over the "
  "structure and complexity of the changed code, not merely its size."),
 ("Class-Imbalance-Aware Predictive Model",
  "A class-imbalance-aware classifier outputs a calibrated conflict probability and is evaluated strictly on the "
  "conflict class using precision, recall, F1 and PR-AUC — never accuracy — and under a leave-one-repository-out "
  "protocol that measures generalisation to a completely unseen project."),
 ("Explainable Artificial Intelligence Layer",
  "Each prediction is explained by two complementary methods: global permutation importance (which features the "
  "model relies on overall) and local counterfactual attribution (which features drove this particular prediction). "
  "The design admits SHAP and GNNExplainer as future exact-attribution layers."),
 ("Risk Scoring and Recommended Action",
  "The calibrated probability is mapped to a discrete risk score (Low / Medium / High), and the dominant risk "
  "drivers are translated into a concrete recommended action — such as a merge ordering, a coordination point on a "
  "shared module, or a rebase/incremental-integration suggestion."),
 ("Detection of Semantic, No-Shared-File Conflicts",
  "The framework targets higher-order semantic conflicts — branches that merge cleanly textually yet break the "
  "build — and demonstrates, through a controlled, compile-verified experiment, that such no-shared-file conflicts "
  "are real and are detectable via the dependency graph where flat overlap features are blind."),
 ("CI/CD-Integrable, Continuous Operation",
  "Because all analysis is performed server-side at the merge base without a working-tree checkout, the predictor "
  "can be embedded in a Continuous-Integration pipeline to gate merges, order or schedule pending merges, and "
  "prioritise regression tests for the implicated dependencies — continuously and at scale."),
]

# ---------------- BUILD DOCUMENT ----------------
parts = []
parts.append(para([("Literature Survey", True)], size=24, after=40, align="center", color="2E4B8E") if False else "")
parts.append(para([("Title:", True)], size=24, after=40))
parts.append(para([(TITLE, True)], size=28, after=160))
parts.append(para([("Team Members:", True)], size=24, after=60))
for m in TEAM:
    parts.append(para(m, size=22, after=40))
parts.append(heading("Literature Survey:", size=26, before=240))

rows = [[tcell("Patent / Publication", W[0], bold=True, shade="D9E1F2"),
         tcell("Description", W[1], bold=True, shade="D9E1F2"),
         tcell("Relevance to your invention", W[2], bold=True, shade="D9E1F2")]]
for num, desc, rel in PATENTS:
    rows.append([tcell(num, W[0], bold=True), tcell(desc, W[1]), tcell(rel, W[2])])
parts.append(table(rows))

parts.append(heading("Existing Problem:", size=26, before=280))
parts.append(para(PROBLEM_INTRO, size=22, after=120))
for h, bl in PROBLEMS:
    parts.append(para([(h, True)], size=22, before=80, after=40))
    for b in bl:
        parts.append(bullet(b))

parts.append(heading("Proposed Solution:", size=26, before=280))
parts.append(para(SOLUTION_INTRO, size=22, after=120))
for h, txt in SOLUTIONS:
    parts.append(para([(h, True)], size=22, before=80, after=30))
    parts.append(para(txt, size=22, after=60))

sect = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" '
        'w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>')
document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
 '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
 '<w:body>' + "".join(parts) + sect + '</w:body></w:document>')

CT = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
 '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
 '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
 '<Default Extension="xml" ContentType="application/xml"/>'
 '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
 '</Types>')
RELS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
 '</Relationships>')

out = r"C:\Users\tamiz\Downloads\Merge_Conflict_Predictor_Literature_Survey.docx"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", CT)
    z.writestr("_rels/.rels", RELS)
    z.writestr("word/document.xml", document)
print("wrote", out)
