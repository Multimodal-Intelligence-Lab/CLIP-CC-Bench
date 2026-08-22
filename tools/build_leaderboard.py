#!/usr/bin/env python3
"""Generate every CLIP-CC-Bench leaderboard surface from the per-judge result trees.

One computation feeds all surfaces, so the README table, the project page, the
Hugging Face Space and leaderboard.json can never disagree with each other.

Data
    results/embedding_models/aggregated_results/<judge>/<vlm>.json   paper pool (frozen)
    results/leaderboard/additions/<judge>/<slug>.json                community pool
    results/leaderboard/models.json                                  display names + params

Ranking rule (paper, section 5)
    Borda points per judge = V - rank by mean HM-CF, descending.
    Final order = total Borda descending, ties broken by the FULL-PRECISION mean
    HM-CF. An exact per-judge tie has no defined tiebreak and is a hard error.

Guard rails
    * the paper's 17-model result is recomputed on every run and compared to the
      frozen results/leaderboard/paper_v1.json; any drift refuses all output,
    * generated regions live between <!-- LEADERBOARD:...:BEGIN/END --> markers;
      a missing or duplicated marker is a hard error and leaves the file untouched,
    * hand-written prose that states a number (the Space "finding" block) is
      asserted, never rewritten.

Usage
    python3 tools/build_leaderboard.py                 # regenerate every surface
    python3 tools/build_leaderboard.py --check         # exit 2 if anything would change
    python3 tools/build_leaderboard.py --freeze-paper  # write paper_v1.json (once, never over)
"""

import argparse
import csv
import datetime
import json
import os
import sys
from pathlib import Path

SCHEMA_VERSION = 1

# Every model is scored on the same 199 clips (clip 126 is absent from the paper's
# model outputs and is excluded throughout).
EXPECTED_EVALUATIONS = 199

RULE = (
    "Borda points per judge = V - rank by mean HM-CF (descending); "
    "final order = total Borda (descending), ties broken by the full-precision "
    "mean HM-CF (descending)."
)

PAPER = "paper"
COMMUNITY = "community"

# The Space "finding" block states these numbers in prose. They are asserted, not
# generated: if the live pool moves past them a human has to rewrite the sentence.
FINDING_TOP_COARSE_2DP = "0.73"
FINDING_TOP_FINE_2DP = "0.63"
FINDING_BEST_HM_2DP = "0.67"

# Spelled-out counts used in the Space lead sentence. 5 is the judge count; 17-25
# covers a plausible leaderboard size. Anything else falls back to digits.
_NUMBER_WORDS = {
    5: "five",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
    21: "twenty-one",
    22: "twenty-two",
    23: "twenty-three",
    24: "twenty-four",
    25: "twenty-five",
}

ROOT = Path(__file__).resolve().parents[1]
CANON_DIR = ROOT / "results" / "embedding_models" / "aggregated_results"
LEADERBOARD_DIR = ROOT / "results" / "leaderboard"
ADDITIONS_DIR = LEADERBOARD_DIR / "additions"
REGISTRY_PATH = LEADERBOARD_DIR / "models.json"
PAPER_V1_PATH = LEADERBOARD_DIR / "paper_v1.json"
LEADERBOARD_JSON_PATH = LEADERBOARD_DIR / "leaderboard.json"
BENCH_README_PATH = ROOT / "README.md"

AGGREGATED_RESULTS_REL = "results/embedding_models/aggregated_results/"


class BuildError(Exception):
    """Fatal condition. Nothing is written and the process exits non-zero."""


# --------------------------------------------------------------------------- #
# text plumbing
# --------------------------------------------------------------------------- #

def number_word(count):
    """Spelled-out count, or the digits when we have no word for it."""
    return _NUMBER_WORDS.get(count, str(count))


def markers(name):
    """(begin, end) marker comments. The unnamed pair wraps the main table."""
    tag = "LEADERBOARD:%s:" % name if name else "LEADERBOARD:"
    return "<!-- %sBEGIN -->" % tag, "<!-- %sEND -->" % tag


def replace_block(text, name, body_lines, where):
    """Replace the lines between a BEGIN/END marker pair.

    Hard-fails unless each marker occurs exactly once, in order: a surface that
    lost or gained a marker must be fixed by hand, not silently rewritten.
    """
    begin, end = markers(name)
    for token in (begin, end):
        found = text.count(token)
        if found != 1:
            raise BuildError(
                "%s: expected exactly one '%s' marker, found %d" % (where, token, found)
            )
    begin_at = text.index(begin)
    end_at = text.index(end)
    if end_at < begin_at:
        raise BuildError("%s: '%s' appears before '%s'" % (where, end, begin))

    body_start = text.index("\n", begin_at) + 1      # first line after BEGIN
    end_line_start = text.rindex("\n", 0, end_at) + 1  # start of the END line
    body = "".join(line + "\n" for line in body_lines)
    return text[:body_start] + body + text[end_line_start:]


def replace_line(text, prefix, new_line, where):
    """Replace the one line starting with `prefix`; hard-fails on none or several."""
    lines = text.split("\n")
    hits = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    if len(hits) != 1:
        raise BuildError(
            "%s: expected exactly one line starting with '%s', found %d"
            % (where, prefix, len(hits))
        )
    lines[hits[0]] = new_line
    return "\n".join(lines)


def read_text(path):
    return path.read_text(encoding="utf-8")


def write_atomic(path, text):
    """Write through a temp file in the same directory, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".build_leaderboard.tmp")
    tmp.write_text(text, encoding="utf-8")
    if path.exists():
        os.chmod(tmp, os.stat(path).st_mode & 0o7777)
    os.replace(tmp, path)


def json_text(document):
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #

def read_result(path):
    """The four fields the leaderboard needs out of one aggregated result file."""
    try:
        data = json.loads(read_text(path))
        return {
            "hm": data["hybrid_stats"]["hm_cf"]["mean"],
            "coarse": data["coarse_grained_stats"]["mean"],
            "fine": data["fine_grained_stats"]["f1"]["mean"],
            "total_evaluations": data["total_evaluations"],
            "error_rate": data.get("error_rate"),
        }
    except KeyError as exc:
        raise BuildError("%s: missing field %s" % (path, exc))
    except ValueError as exc:
        raise BuildError("%s: not valid JSON (%s)" % (path, exc))


def load_tree(root):
    """<root>/<judge>/<slug>.json -> {judge: {slug: metrics}}; {} when root is absent."""
    tree = {}
    if not root.is_dir():
        return tree
    for judge_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        tree[judge_dir.name] = {
            path.stem: read_result(path) for path in sorted(judge_dir.glob("*.json"))
        }
    return tree


def load_registry():
    if not REGISTRY_PATH.exists():
        raise BuildError("%s is missing" % REGISTRY_PATH)
    registry = json.loads(read_text(REGISTRY_PATH))
    if "models" not in registry:
        raise BuildError("%s: no 'models' object" % REGISTRY_PATH)
    return registry


def build_pool(paper_tree, additions_tree):
    """Merge both trees into {"judges": [...], "models": {slug: {...}}}."""
    judges = sorted(set(paper_tree) | set(additions_tree))
    models = {}
    for tree, provenance in ((paper_tree, PAPER), (additions_tree, COMMUNITY)):
        for judge, entries in tree.items():
            for slug, metrics in entries.items():
                model = models.setdefault(
                    slug, {"provenance": provenance, "per_judge": {}}
                )
                model["per_judge"][judge] = metrics
    return {"judges": judges, "models": models}


def paper_subset(pool):
    return {
        "judges": pool["judges"],
        "models": {
            slug: model
            for slug, model in pool["models"].items()
            if model["provenance"] == PAPER
        },
    }


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #

def validate(pool, registry, paper_tree, additions_tree):
    judges = pool["judges"]
    models = pool["models"]
    if not judges:
        raise BuildError("no judge directories under %s" % CANON_DIR)
    if not models:
        raise BuildError("no model results under %s" % CANON_DIR)

    paper_slugs = {slug for entries in paper_tree.values() for slug in entries}
    addition_slugs = {slug for entries in additions_tree.values() for slug in entries}
    lowered = {slug.lower(): slug for slug in paper_slugs}
    for slug in sorted(addition_slugs):
        clash = lowered.get(slug.lower())
        if clash is not None:
            raise BuildError(
                "slug collision: addition '%s' collides with paper model '%s' "
                "(slugs must be unique case-insensitively)" % (slug, clash)
            )

    # every judge has to score exactly the same models
    scored = {judge: set() for judge in judges}
    for tree in (paper_tree, additions_tree):
        for judge, entries in tree.items():
            scored[judge] |= set(entries)
    for judge in judges:
        missing = sorted(set(models) - scored[judge])
        if missing:
            raise BuildError(
                "judge '%s' has no results for: %s" % (judge, ", ".join(missing))
            )

    for slug in sorted(models):
        for judge in judges:
            metrics = models[slug]["per_judge"][judge]
            if metrics["total_evaluations"] != EXPECTED_EVALUATIONS:
                raise BuildError(
                    "%s/%s: total_evaluations is %r, expected %d"
                    % (judge, slug, metrics["total_evaluations"], EXPECTED_EVALUATIONS)
                )
            if metrics["error_rate"] is not None and metrics["error_rate"] != 0.0:
                raise BuildError(
                    "%s/%s: error_rate is %r, expected 0.0"
                    % (judge, slug, metrics["error_rate"])
                )

    # an exact per-judge tie has no defined tiebreak
    for judge in judges:
        seen = {}
        for slug in sorted(models):
            value = models[slug]["per_judge"][judge]["hm"]
            if value in seen:
                raise BuildError(
                    "exact per-judge tie under '%s': '%s' and '%s' both score %r on "
                    "mean HM-CF; the ranking rule has no tiebreak for this, resolve it "
                    "by hand" % (judge, seen[value], slug, value)
                )
            seen[value] = slug

    entries = registry["models"]
    missing = sorted(set(models) - set(entries))
    if missing:
        raise BuildError(
            "%s has no entry for: %s" % (REGISTRY_PATH, ", ".join(missing))
        )
    displays = {}
    for slug in sorted(models):
        entry = entries[slug]
        display = entry.get("display")
        params = entry.get("params")
        if not display:
            raise BuildError("%s: '%s' has no display name" % (REGISTRY_PATH, slug))
        if not params:
            raise BuildError("%s: '%s' has no params" % (REGISTRY_PATH, slug))
        if display in displays:
            raise BuildError(
                "%s: display name '%s' is used by both '%s' and '%s'"
                % (REGISTRY_PATH, display, displays[display], slug)
            )
        displays[display] = slug


# --------------------------------------------------------------------------- #
# compute
# --------------------------------------------------------------------------- #

def mean(values):
    return sum(values) / len(values)


def compute_rows(pool, registry):
    """Borda per judge, then the final order. Means stay at full precision."""
    judges = pool["judges"]
    models = pool["models"]
    slugs = sorted(models)
    n_models = len(slugs)

    borda = dict.fromkeys(slugs, 0)
    for judge in judges:
        ordered = sorted(
            slugs, key=lambda slug: models[slug]["per_judge"][judge]["hm"], reverse=True
        )
        for rank, slug in enumerate(ordered, start=1):
            borda[slug] += n_models - rank

    rows = []
    for slug in slugs:
        per_judge = models[slug]["per_judge"]
        entry = registry["models"][slug]
        mean_hm = mean([per_judge[judge]["hm"] for judge in judges])
        rows.append(
            {
                "slug": slug,
                "display": entry["display"],
                "provenance": models[slug]["provenance"],
                "params": entry["params"],
                "borda": borda[slug],
                "mean_hm_cf": mean_hm,
                "mean_coarse": mean([per_judge[judge]["coarse"] for judge in judges]),
                "mean_fine": mean([per_judge[judge]["fine"] for judge in judges]),
                "hm_cf_2dp": f"{mean_hm:.2f}",
            }
        )

    # total Borda first, then the unrounded mean; both descending
    rows.sort(key=lambda row: (row["borda"], row["mean_hm_cf"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def max_borda(n_judges, n_models):
    return n_judges * (n_models - 1)


def paper_document(rows, judges):
    return {
        "schema_version": SCHEMA_VERSION,
        "rule": RULE,
        "judges": list(judges),
        "V": len(rows),
        "max_borda": max_borda(len(judges), len(rows)),
        "rows": [
            {
                "slug": row["slug"],
                "display": row["display"],
                "borda": row["borda"],
                "mean_hm_cf": row["mean_hm_cf"],
                "hm_cf_2dp": row["hm_cf_2dp"],
                "mean_coarse": row["mean_coarse"],
                "mean_fine": row["mean_fine"],
                "rank": row["rank"],
            }
            for row in rows
        ],
    }


def leaderboard_document(rows, judges, generated, registry):
    documented = []
    for row in rows:
        entry = {
            "slug": row["slug"],
            "display": row["display"],
            "provenance": row["provenance"],
            "params": row["params"],
            "borda": row["borda"],
            "mean_hm_cf": row["mean_hm_cf"],
            "mean_coarse": row["mean_coarse"],
            "mean_fine": row["mean_fine"],
            "hm_cf_2dp": row["hm_cf_2dp"],
            "rank": row["rank"],
        }
        if row["provenance"] == COMMUNITY:
            extra = {
                key: value
                for key, value in registry["models"][row["slug"]].items()
                if key not in ("display", "params")
            }
            entry.update(extra)
        documented.append(entry)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated": generated,
        "judges": list(judges),
        "V": len(rows),
        "max_borda": max_borda(len(judges), len(rows)),
        "rule": RULE,
        "rows": documented,
    }


PAPER_ROW_FIELDS = (
    "slug",
    "display",
    "borda",
    "mean_hm_cf",
    "hm_cf_2dp",
    "mean_coarse",
    "mean_fine",
    "rank",
)


def check_paper_regression(paper_rows, judges, path):
    """Recompute the paper pool and compare it to the frozen copy, field by field."""
    if not path.exists():
        raise BuildError(
            "%s is missing; run 'build_leaderboard.py --freeze-paper' once and commit it"
            % path
        )
    frozen = json.loads(read_text(path))
    current = paper_document(paper_rows, judges)
    for key in ("V", "max_borda", "judges"):
        if frozen.get(key) != current[key]:
            raise BuildError(
                "paper regression: %s is %r, frozen copy says %r"
                % (key, current[key], frozen.get(key))
            )
    frozen_rows = frozen.get("rows", [])
    if len(frozen_rows) != len(current["rows"]):
        raise BuildError(
            "paper regression: recomputed %d rows, frozen copy has %d"
            % (len(current["rows"]), len(frozen_rows))
        )
    for got, want in zip(current["rows"], frozen_rows):
        for field in PAPER_ROW_FIELDS:
            if got[field] != want.get(field):
                raise BuildError(
                    "paper regression at rank %d (%s): %s is %r, frozen copy says %r"
                    % (got["rank"], got["slug"], field, got[field], want.get(field))
                )


def check_finding_prose(rows):
    """Assert the claims the hand-written Space finding block makes."""
    advice = (
        "the 'finding' block in the Space index.html states this in prose; it needs a "
        "human edit, this generator will not rewrite it"
    )
    for row in rows:
        if not row["mean_coarse"] > row["mean_fine"]:
            raise BuildError(
                "%s: mean coarse %r is not above mean fine %r, so \"coarse beats fine\" "
                "no longer holds; %s" % (row["display"], row["mean_coarse"], row["mean_fine"], advice)
            )
    top = rows[0]
    top_coarse = f"{top['mean_coarse']:.2f}"
    top_fine = f"{top['mean_fine']:.2f}"
    if (top_coarse, top_fine) != (FINDING_TOP_COARSE_2DP, FINDING_TOP_FINE_2DP):
        raise BuildError(
            "rank 1 (%s) now scores %s coarse / %s fine, not %s / %s; %s"
            % (top["display"], top_coarse, top_fine,
               FINDING_TOP_COARSE_2DP, FINDING_TOP_FINE_2DP, advice)
        )
    best = f"{max(row['mean_hm_cf'] for row in rows):.2f}"
    if best != FINDING_BEST_HM_2DP:
        raise BuildError(
            "the best mean HM-CF is now %s, not %s; %s" % (best, FINDING_BEST_HM_2DP, advice)
        )


# --------------------------------------------------------------------------- #
# renderers
# --------------------------------------------------------------------------- #

def bench_intro_lines(ctx):
    return [
        "",
        "Final ranking of the %d VLMs on CLIP-CC-Bench: **Borda** = Borda count across "
        "the %d embedding judges; **Mean HM-CF** = average harmonic mean of coarse- and "
        "fine-grained similarity across judges. Full per-judge scores are in the "
        "[`%s`](%s) folder."
        % (ctx["V"], ctx["J"], AGGREGATED_RESULTS_REL, AGGREGATED_RESULTS_REL),
        "",
    ]


def bench_table_lines(ctx):
    lines = ["", "| Rank | VLM | Borda | Mean HM-CF |", "|-----:|-----|------:|-----------:|"]
    for row in ctx["rows"]:
        lines.append(
            "| %d | %s | %d | %s |"
            % (row["rank"], row["display"], row["borda"], row["hm_cf_2dp"])
        )
    lines.append("")
    return lines


def space_meta_lines(ctx):
    return [
        '<meta name="description" content="Leaderboard for CLIP-CC-Bench: %d '
        "video-language models ranked on paragraph-level descriptions of 90-second movie "
        'clips. Multimodal Intelligence Lab, South Dakota State University.">' % ctx["V"]
    ]


def space_lead_lines(ctx):
    return [
        '  <p class="lead">%s video&ndash;language models ranked on paragraph-level '
        "descriptions of" % number_word(ctx["V"]).capitalize(),
        "  90-second movie clips. <b>Borda</b> is the Borda count across the %s embedding "
        "judges;" % number_word(ctx["J"]),
        "  <b>Mean HM-CF</b> is the average harmonic mean of coarse- and fine-grained "
        "similarity across",
        "  those judges.</p>",
    ]


def space_caption_lines(ctx):
    return [
        '      <caption id="lb-caption">Final ranking of the %d VLMs on CLIP-CC-Bench. '
        "Borda max = %d&times;%d = %d.</caption>"
        % (ctx["V"], ctx["J"], ctx["V"] - 1, ctx["max_borda"])
    ]


def space_tbody_lines(ctx):
    lines = ["      <tbody>"]
    for row in ctx["rows"]:
        lead = ' class="lead-row"' if row["rank"] == 1 else ""
        dot = " &#9679;" if row["provenance"] == COMMUNITY else ""
        lines.append(
            '        <tr%s><th scope="row" class="num">%d</th><td>%s%s</td>'
            '<td class="num">%d</td><td class="num">%s</td></tr>'
            % (lead, row["rank"], row["display"], dot, row["borda"], row["hm_cf_2dp"])
        )
    lines.append("      </tbody>")
    return lines


def space_footnote_lines(ctx):
    if not ctx["has_community"]:
        return []
    return [
        '  <p class="note">Rows marked &#9679; were added after publication; the paper\'s '
        "frozen %d-model result (V=%d, max Borda %d) is preserved in the repository.</p>"
        % (ctx["paper_V"], ctx["paper_V"], ctx["paper_max_borda"])
    ]


def space_readme_short_description(ctx):
    return "short_description: %d VLMs ranked on long-form video description" % ctx["V"]


def space_readme_body_lines(ctx):
    # Line 1 describes the live pool, line 2 the frozen paper pool. Keep the "evaluated
    # in the paper" claim on the frozen count: it must not move when a model is added.
    return [
        "",
        "This Space is a static page showing the current ranking of the %d video–language "
        "models on" % ctx["V"],
        "CLIP-CC-Bench. Metrics are computed on 199 of the 200 clips (clip 126 is absent "
        "from the outputs",
        "of the %d models evaluated in the paper)." % ctx["paper_V"],
        "",
    ]


def site_stat_lines(ctx):
    return [
        '        <div class="stat"><div class="big">%d</div><div class="lab">'
        "video&ndash;language models <span>(benchmarked)</span></div></div>" % ctx["V"]
    ]


def site_prose_lines(ctx):
    return [
        '      <p class="prose" style="margin-top:1rem">Final ranking of the %d VLMs. '
        "<strong>Borda</strong> = Borda count across the %s embedding judges; "
        "<strong>Mean HM-CF</strong> = average harmonic mean of coarse- and fine-grained "
        "similarity across judges.</p>" % (ctx["V"], number_word(ctx["J"]))
    ]


def site_tbody_lines(ctx):
    lines = ["        <tbody>"]
    for row in ctx["rows"]:
        lines.append(
            '          <tr><td class="n">%d</td><td>%s</td><td class="n">%d</td>'
            '<td class="n">%s</td></tr>'
            % (row["rank"], row["display"], row["borda"], row["hm_cf_2dp"])
        )
    lines.append("        </tbody>")
    return lines


# --------------------------------------------------------------------------- #
# surfaces
# --------------------------------------------------------------------------- #

def bench_readme_text(text, ctx, where):
    text = replace_block(text, "INTRO", bench_intro_lines(ctx), where)
    return replace_block(text, "", bench_table_lines(ctx), where)


def space_index_text(text, ctx, where):
    text = replace_block(text, "META", space_meta_lines(ctx), where)
    text = replace_block(text, "LEAD", space_lead_lines(ctx), where)
    text = replace_block(text, "CAPTION", space_caption_lines(ctx), where)
    text = replace_block(text, "ROWS", space_tbody_lines(ctx), where)
    return replace_block(text, "FOOTNOTE", space_footnote_lines(ctx), where)


def space_readme_text(text, ctx, where):
    text = replace_line(
        text, "short_description:", space_readme_short_description(ctx), where
    )
    return replace_block(text, "INTRO", space_readme_body_lines(ctx), where)


def site_index_text(text, ctx, where):
    text = replace_block(text, "STAT", site_stat_lines(ctx), where)
    text = replace_block(text, "PROSE", site_prose_lines(ctx), where)
    return replace_block(text, "ROWS", site_tbody_lines(ctx), where)


def collect_surfaces(ctx, site_dir, space_dir, notices):
    """[(label, path, new_text)] for every surface present on this machine."""
    surfaces = [
        ("bench README.md", BENCH_README_PATH,
         bench_readme_text(read_text(BENCH_README_PATH), ctx, "README.md")),
        ("bench leaderboard.json", LEADERBOARD_JSON_PATH, json_text(ctx["document"])),
    ]

    if space_dir is None or not space_dir.is_dir():
        notices.append("skipping the Space: %s not found" % space_dir)
    else:
        index = space_dir / "index.html"
        readme = space_dir / "README.md"
        surfaces.append(
            ("space index.html", index, space_index_text(read_text(index), ctx, str(index)))
        )
        surfaces.append(
            ("space README.md", readme, space_readme_text(read_text(readme), ctx, str(readme)))
        )
        surfaces.append(
            ("space leaderboard.json", space_dir / "leaderboard.json", json_text(ctx["document"]))
        )

    if site_dir is None or not site_dir.is_dir():
        notices.append("skipping the project site: %s not found" % site_dir)
    else:
        index = site_dir / "index.html"
        surfaces.append(
            ("site index.html", index, site_index_text(read_text(index), ctx, str(index)))
        )
        surfaces.append(
            ("site leaderboard.json", site_dir / "leaderboard.json", json_text(ctx["document"]))
        )

    return surfaces


def print_pwc_block(ctx):
    """Paste-ready block for external leaderboards (Papers with Code and friends)."""
    print("# Paste block: display name <TAB> Mean HM-CF (2 dp).")
    print("# Borda is deliberately NOT entered there: it is a rank aggregate over this")
    print("# pool of %d models under %d judges, so it is meaningless outside CLIP-CC-Bench"
          % (ctx["V"], ctx["J"]))
    print("# and would change for every model already listed whenever a model is added.")
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    for row in ctx["rows"]:
        writer.writerow([row["display"], row["hm_cf_2dp"]])


def publish_space(space_dir, repo_id):
    """Upload the Space folder to the Hub. Never reached by --check or the tests."""
    from huggingface_hub import HfApi  # lazy: the rest of this tool is stdlib-only

    token_path = Path.home() / ".hf_token"
    if not token_path.exists():
        raise BuildError("%s not found; --publish reads the Hub token from there" % token_path)
    if space_dir is None or not space_dir.is_dir():
        raise BuildError("--publish needs a Space folder; %s not found" % space_dir)
    api = HfApi(token=read_text(token_path).strip())
    api.upload_folder(folder_path=str(space_dir), repo_id=repo_id, repo_type="space")
    print("published %s to https://huggingface.co/spaces/%s" % (space_dir, repo_id))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Generate the CLIP-CC-Bench leaderboard surfaces."
    )
    parser.add_argument(
        "--date", default=datetime.date.today().isoformat(),
        help="value for the 'generated' field of leaderboard.json (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--site-dir", default=str(ROOT.parent / "clipcc-site"),
        help="project-page checkout (skipped when absent)",
    )
    parser.add_argument(
        "--space-dir", default=str(ROOT.parent / "clipcc-space"),
        help="Hugging Face Space folder (skipped when absent)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="report drift instead of writing; exit 2 if any surface would change",
    )
    parser.add_argument(
        "--freeze-paper", action="store_true",
        help="write results/leaderboard/paper_v1.json and stop; refuses if it exists",
    )
    parser.add_argument(
        "--publish", action="store_true",
        help="after generating, upload the Space folder to the Hub",
    )
    parser.add_argument(
        "--space-repo", default=None,
        help="Hub repo id for --publish, e.g. ORG/CLIP-CC-Bench-Leaderboard",
    )
    return parser.parse_args(argv)


def run(args):
    try:
        datetime.date.fromisoformat(args.date)
    except ValueError:
        raise BuildError("--date must be YYYY-MM-DD, got '%s'" % args.date)
    if args.publish and not args.space_repo:
        raise BuildError("--publish also needs --space-repo ORG/NAME")

    registry = load_registry()
    paper_tree = load_tree(CANON_DIR)
    additions_tree = load_tree(ADDITIONS_DIR)
    pool = build_pool(paper_tree, additions_tree)
    validate(pool, registry, paper_tree, additions_tree)

    paper_rows = compute_rows(paper_subset(pool), registry)
    rows = compute_rows(pool, registry)
    judges = pool["judges"]

    if args.freeze_paper:
        if PAPER_V1_PATH.exists():
            raise BuildError(
                "%s already exists; the frozen paper snapshot is immutable — delete it "
                "manually only if you are certain" % PAPER_V1_PATH.name
            )
        document = paper_document(paper_rows, judges)
        write_atomic(PAPER_V1_PATH, json_text(document))
        print(
            "froze %d paper rows (V=%d, max Borda %d) to %s"
            % (document["V"], document["V"], document["max_borda"], PAPER_V1_PATH)
        )
        return 0

    check_paper_regression(paper_rows, judges, PAPER_V1_PATH)
    check_finding_prose(rows)

    frozen = json.loads(read_text(PAPER_V1_PATH))
    ctx = {
        "rows": rows,
        "V": len(rows),
        "J": len(judges),
        "max_borda": max_borda(len(judges), len(rows)),
        "has_community": any(row["provenance"] == COMMUNITY for row in rows),
        "paper_V": frozen["V"],
        "paper_max_borda": frozen["max_borda"],
        "document": leaderboard_document(rows, judges, args.date, registry),
    }

    notices = []
    surfaces = collect_surfaces(ctx, Path(args.site_dir), Path(args.space_dir), notices)
    for notice in notices:
        print(notice)

    drifted = []
    for label, path, new_text in surfaces:
        current = read_text(path) if path.exists() else None
        if current == new_text:
            if not args.check:
                print("unchanged  %s" % label)
            continue
        drifted.append(label)
        if not args.check:
            write_atomic(path, new_text)
            print("wrote      %s (%s)" % (label, path))

    if args.check:
        if drifted:
            print("drift detected in %d surface(s):" % len(drifted))
            for label in drifted:
                print("  - %s" % label)
            print("run 'python3 tools/build_leaderboard.py' to regenerate")
            return 2
        print("all %d surface(s) up to date" % len(surfaces))
        return 0

    print()
    print_pwc_block(ctx)

    if args.publish:
        publish_space(Path(args.space_dir), args.space_repo)
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except BuildError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
