#!/usr/bin/env python3
"""Tests for tools/build_leaderboard.py.

Run directly (no test runner needed):

    python3 tools/test_build_leaderboard.py

pytest also collects this file, but nothing here depends on it. Every test that
writes works on a temporary copy; the real surfaces are never touched.
"""

import contextlib
import copy
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import build_leaderboard as bl  # noqa: E402

SITE_DIR = ROOT.parent / "clipcc-site"
SPACE_DIR = ROOT.parent / "clipcc-space"

# Table 2 of the paper, transcribed by hand: (display, Borda, mean HM-CF at 2 dp)
# in rank order. Deliberately independent of anything under results/.
PUBLISHED = [
    ("VideoLLaMA3", 80, "0.67"),
    ("mPLUG-Owl3", 75, "0.66"),
    ("LLaVA-OneVision", 67, "0.64"),
    ("ViLAMP", 67, "0.64"),
    ("LongVU", 61, "0.63"),
    ("Qwen2.5-72B", 55, "0.62"),
    ("Qwen2.5-32B", 48, "0.61"),
    ("VideoChat-Flash", 42, "0.60"),
    ("MiniCPM-V", 42, "0.60"),
    ("Video-XL", 36, "0.59"),
    ("ShareGPT4Video", 29, "0.58"),
    ("InternVL2", 27, "0.58"),
    ("TimeChat", 20, "0.56"),
    ("LLaVA-NeXT-Video", 16, "0.55"),
    ("TS-LLaVA", 9, "0.53"),
    ("Oryx", 6, "0.52"),
    ("LongVA", 0, "0.48"),
]

NEWCOMER = "newcomer_x"


def load_everything():
    registry = bl.load_registry()
    paper_tree = bl.load_tree(bl.CANON_DIR)
    additions_tree = bl.load_tree(bl.ADDITIONS_DIR)
    pool = bl.build_pool(paper_tree, additions_tree)
    bl.validate(pool, registry, paper_tree, additions_tree)
    return pool, registry, paper_tree, additions_tree


def paper_rows():
    pool, registry, _, _ = load_everything()
    return bl.compute_rows(bl.paper_subset(pool), registry), pool, registry


# --------------------------------------------------------------------------- #
# 1. the frozen paper result still recomputes, row for row
# --------------------------------------------------------------------------- #

def test_paper_v1_regression():
    rows, pool, _ = paper_rows()
    # raises BuildError on any drift in value, 2 dp string or order
    bl.check_paper_regression(rows, pool["judges"], bl.PAPER_V1_PATH)

    frozen = json.loads(bl.PAPER_V1_PATH.read_text(encoding="utf-8"))
    assert frozen["V"] == 17, frozen["V"]
    assert frozen["max_borda"] == 80, frozen["max_borda"]
    assert len(frozen["rows"]) == 17, len(frozen["rows"])
    assert [row["rank"] for row in frozen["rows"]] == list(range(1, 18))

    by_slug = {row["slug"]: row for row in frozen["rows"]}

    # Both places where Borda ties and the full-precision mean decides the order.
    # Rounded to 2 dp these pairs are indistinguishable, which is exactly the bug
    # the ranking script used to have.
    for better, worse, borda, shown, ranks in (
        ("llava_one_vision", "vilamp", 67, "0.64", (3, 4)),
        ("videochatflash", "minicpm", 42, "0.60", (8, 9)),
    ):
        top, bottom = by_slug[better], by_slug[worse]
        assert top["borda"] == bottom["borda"] == borda
        assert top["hm_cf_2dp"] == bottom["hm_cf_2dp"] == shown
        assert top["mean_hm_cf"] > bottom["mean_hm_cf"], (better, worse)
        assert (top["rank"], bottom["rank"]) == ranks


# --------------------------------------------------------------------------- #
# 2. the recomputed paper pool equals the published table
# --------------------------------------------------------------------------- #

def test_published_table():
    rows, _, _ = paper_rows()
    got = [(row["display"], row["borda"], row["hm_cf_2dp"]) for row in rows]
    assert got == PUBLISHED, got


# --------------------------------------------------------------------------- #
# 3. adding a model can only cost each existing model the duels it loses
# --------------------------------------------------------------------------- #

def add_newcomer(pool, pick):
    """Copy of `pool` with an 18th model scoring pick(sorted per-judge values)."""
    grown = copy.deepcopy(pool)
    per_judge = {}
    for judge in grown["judges"]:
        values = sorted(
            model["per_judge"][judge]["hm"] for model in grown["models"].values()
        )
        score = pick(values)
        assert score not in values, "synthetic score must not tie an existing one"
        per_judge[judge] = {
            "hm": score,
            "coarse": 0.70,
            "fine": 0.60,
            "total_evaluations": bl.EXPECTED_EVALUATIONS,
            "error_rate": 0.0,
        }
    grown["models"][NEWCOMER] = {"provenance": bl.COMMUNITY, "per_judge": per_judge}
    return grown


def test_pairwise_win_invariant():
    _, pool, registry = paper_rows()
    paper_pool = bl.paper_subset(pool)
    before = {row["slug"]: row["borda"] for row in bl.compute_rows(paper_pool, registry)}

    grown_registry = copy.deepcopy(registry)
    grown_registry["models"][NEWCOMER] = {"display": "Newcomer-X", "params": "7B"}

    cases = {
        "beats everyone": lambda values: values[-1] + 0.01,
        "loses to everyone": lambda values: values[0] - 0.01,
        "lands mid-table": lambda values: (values[8] + values[9]) / 2,
    }
    for label, pick in cases.items():
        grown = add_newcomer(paper_pool, pick)
        rows = bl.compute_rows(grown, grown_registry)
        assert len(rows) == 18, label
        by_slug = {row["slug"]: row for row in rows}
        newcomer = grown["models"][NEWCOMER]["per_judge"]

        deltas = []
        for slug, old_borda in before.items():
            wins = sum(
                1
                for judge in grown["judges"]
                if grown["models"][slug]["per_judge"][judge]["hm"] > newcomer[judge]["hm"]
            )
            delta = by_slug[slug]["borda"] - old_borda
            assert delta == wins, (label, slug, delta, wins)
            deltas.append(delta)

        if label == "beats everyone":
            assert by_slug[NEWCOMER]["borda"] == 85, by_slug[NEWCOMER]["borda"]
            assert by_slug[NEWCOMER]["rank"] == 1
            assert set(deltas) == {0}, deltas
        if label == "loses to everyone":
            assert by_slug[NEWCOMER]["borda"] == 0, by_slug[NEWCOMER]["borda"]
            assert by_slug[NEWCOMER]["rank"] == 18
            assert set(deltas) == {5}, deltas


# --------------------------------------------------------------------------- #
# 4. an exact per-judge tie is a hard failure that names both models
# --------------------------------------------------------------------------- #

def test_exact_tie_hard_fails():
    _, registry, paper_tree, _ = load_everything()
    tied = copy.deepcopy(paper_tree)
    judge = sorted(tied)[0]
    tied[judge]["vilamp"]["hm"] = tied[judge]["llava_one_vision"]["hm"]
    pool = bl.build_pool(tied, {})
    try:
        bl.validate(pool, registry, tied, {})
    except (bl.BuildError, SystemExit) as exc:
        message = str(exc)
        assert "llava_one_vision" in message, message
        assert "vilamp" in message, message
        assert judge in message, message
    else:
        raise AssertionError("an exact per-judge tie must be a hard failure")


# --------------------------------------------------------------------------- #
# 5. two runs with the same --date produce byte-identical trees
# --------------------------------------------------------------------------- #

def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def test_idempotent_generation():
    ignore = shutil.ignore_patterns("__pycache__")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        bench = tmp / "bench"
        bench.mkdir()
        shutil.copy2(ROOT / "README.md", bench / "README.md")
        shutil.copytree(ROOT / "tools", bench / "tools", ignore=ignore)
        shutil.copytree(ROOT / "results", bench / "results", ignore=ignore)

        site = tmp / "clipcc-site"
        site.mkdir()
        shutil.copy2(SITE_DIR / "index.html", site / "index.html")
        space = tmp / "clipcc-space"
        space.mkdir()
        shutil.copy2(SPACE_DIR / "index.html", space / "index.html")
        shutil.copy2(SPACE_DIR / "README.md", space / "README.md")

        command = [
            sys.executable,
            str(bench / "tools" / "build_leaderboard.py"),
            "--date", "2026-01-01",
            "--site-dir", str(site),
            "--space-dir", str(space),
        ]
        first = subprocess.run(command, capture_output=True, text=True)
        assert first.returncode == 0, first.stderr
        after_first = snapshot(tmp)

        second = subprocess.run(command, capture_output=True, text=True)
        assert second.returncode == 0, second.stderr
        after_second = snapshot(tmp)

        assert after_first == after_second, sorted(
            name for name in after_first
            if after_first[name] != after_second.get(name)
        )
        # the second run should recognise every surface as already current
        assert "wrote" not in second.stdout, second.stdout


# --------------------------------------------------------------------------- #
# 6. a missing or duplicated marker fails without touching the file
# --------------------------------------------------------------------------- #

def test_marker_misuse_hard_fails():
    begin, end = bl.markers("ROWS")
    good = "before\n%s\nold row\n%s\nafter\n" % (begin, end)
    assert bl.replace_block(good, "ROWS", ["new row"], "ok.html") == (
        "before\n%s\nnew row\n%s\nafter\n" % (begin, end)
    )

    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "surface.html"
        for label, text in (
            ("missing", "before\nno markers here\nafter\n"),
            ("duplicated", good + good),
        ):
            path.write_text(text, encoding="utf-8")
            before_bytes = path.read_bytes()
            try:
                replaced = bl.replace_block(
                    path.read_text(encoding="utf-8"), "ROWS", ["new row"], str(path)
                )
                bl.write_atomic(path, replaced)
            except (bl.BuildError, SystemExit) as exc:
                assert "marker" in str(exc), (label, exc)
            else:
                raise AssertionError("%s marker must be a hard failure" % label)
            assert path.read_bytes() == before_bytes, label
            assert list(path.parent.iterdir()) == [path], "temp file left behind"


# --------------------------------------------------------------------------- #
# 7. the repo's own ranking script agrees (skipped when pandas is absent)
# --------------------------------------------------------------------------- #

def test_rank_vlms_agrees():
    try:
        import pandas  # noqa: F401
    except ImportError:
        print("SKIP  test_rank_vlms_agrees (pandas is not installed)")
        return

    rows, pool, _ = paper_rows()
    script = ROOT / "src" / "scripts" / "rank_vlms.py"
    spec = importlib.util.spec_from_file_location("rank_vlms", script)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
        results = module.load_evaluation_results(bl.CANON_DIR)
        borda, means, stds = module.compute_borda_scores(results)
        table = module.generate_overall_vlm_ranking_table(borda, means, stds)

    assert {row["slug"]: row["borda"] for row in rows} == {
        slug: int(points) for slug, points in borda.items()
    }
    assert [row["slug"] for row in rows] == list(table["VLM"])
    displays = {row["slug"]: row["display"] for row in rows}
    assert [displays[slug] for slug in table["VLM"]] == [name for name, _, _ in PUBLISHED]
    assert list(table["Rank"]) == list(range(1, 18))


# --------------------------------------------------------------------------- #
# 8. --freeze-paper refuses to overwrite an existing snapshot
# --------------------------------------------------------------------------- #

def scratch_tree(tmp):
    """Copies of the bench repo, the site and the Space under `tmp`."""
    ignore = shutil.ignore_patterns("__pycache__")
    bench = tmp / "bench"
    bench.mkdir()
    shutil.copy2(ROOT / "README.md", bench / "README.md")
    shutil.copytree(ROOT / "tools", bench / "tools", ignore=ignore)
    shutil.copytree(ROOT / "results", bench / "results", ignore=ignore)

    site = tmp / "clipcc-site"
    site.mkdir()
    shutil.copy2(SITE_DIR / "index.html", site / "index.html")
    space = tmp / "clipcc-space"
    space.mkdir()
    shutil.copy2(SPACE_DIR / "index.html", space / "index.html")
    shutil.copy2(SPACE_DIR / "README.md", space / "README.md")
    return bench, site, space


def test_freeze_paper_refuses_to_overwrite():
    with tempfile.TemporaryDirectory() as raw:
        bench, _, _ = scratch_tree(Path(raw))
        frozen = bench / "results" / "leaderboard" / "paper_v1.json"
        before = frozen.read_bytes()

        result = subprocess.run(
            [sys.executable, str(bench / "tools" / "build_leaderboard.py"), "--freeze-paper"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, result.stdout
        assert "already exists" in result.stderr, result.stderr
        assert "immutable" in result.stderr, result.stderr
        assert frozen.read_bytes() == before, "the frozen snapshot was modified"
        assert list(frozen.parent.glob("*.tmp")) == [], "temp file left behind"


# --------------------------------------------------------------------------- #
# 9. the paper's count never follows the live pool
# --------------------------------------------------------------------------- #

PAPER_PHRASE = "evaluated in the paper"


def write_newcomer(bench, pick):
    """Add the synthetic 18th model to a scratch copy as a community submission."""
    pool, _, _, _ = load_everything()
    grown = add_newcomer(bl.paper_subset(pool), pick)
    leaderboard = bench / "results" / "leaderboard"

    for judge, metrics in grown["models"][NEWCOMER]["per_judge"].items():
        path = leaderboard / "additions" / judge / (NEWCOMER + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "hybrid_stats": {"hm_cf": {"mean": metrics["hm"]}},
                    "coarse_grained_stats": {"mean": metrics["coarse"]},
                    "fine_grained_stats": {"f1": {"mean": metrics["fine"]}},
                    "total_evaluations": metrics["total_evaluations"],
                    "error_rate": metrics["error_rate"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    registry_path = leaderboard / "models.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["models"][NEWCOMER] = {
        "display": "Newcomer-X",
        "params": "7B",
        "raw_outputs_url": "https://example.invalid/newcomer-x",
        "added": "2026-01-01",
    }
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


def test_paper_count_is_decoupled_from_live_count():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        bench, site, space = scratch_tree(tmp)
        write_newcomer(bench, lambda values: (values[8] + values[9]) / 2)

        result = subprocess.run(
            [
                sys.executable,
                str(bench / "tools" / "build_leaderboard.py"),
                "--date", "2026-01-01",
                "--site-dir", str(site),
                "--space-dir", str(space),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

        rendered = {
            "bench README.md": bench / "README.md",
            "bench leaderboard.json": bench / "results" / "leaderboard" / "leaderboard.json",
            "space index.html": space / "index.html",
            "space README.md": space / "README.md",
            "space leaderboard.json": space / "leaderboard.json",
            "site index.html": site / "index.html",
            "site leaderboard.json": site / "leaderboard.json",
        }
        rendered = {
            label: path.read_text(encoding="utf-8") for label, path in rendered.items()
        }

        # (a) the paper claim stays on the frozen 17 wherever it is made. Surfaces are
        # scanned with whitespace collapsed, so a claim wrapped over a line break is
        # still found (that wrap is how the pre-fix wording hid this one).
        hits = 0
        for label, text in rendered.items():
            flat = " ".join(text.split())
            at = flat.find(PAPER_PHRASE)
            while at >= 0:
                hits += 1
                window = flat[max(0, at - 80):at + len(PAPER_PHRASE) + 40]
                assert "17" in window, (label, window)
                assert "18" not in window, (label, window)
                at = flat.find(PAPER_PHRASE, at + len(PAPER_PHRASE))
        assert hits >= 2, hits  # Space README body and Space index note

        # (b) every live-scoped count did move to 18
        for label, needles in {
            "bench README.md": ["Final ranking of the 18 VLMs on CLIP-CC-Bench"],
            "space index.html": [
                "CLIP-CC-Bench: 18 video-language models",
                "Eighteen video&ndash;language models ranked",
                "Final ranking of the 18 VLMs on CLIP-CC-Bench. Borda max = 5&times;17 = 85.",
                # ... while the community footnote quotes the frozen pool
                "frozen 17-model result (V=17, max Borda 80)",
            ],
            "space README.md": [
                "short_description: 18 VLMs ranked on long-form video description",
                "current ranking of the 18 video–language models on",
            ],
            "site index.html": [
                '<div class="big">18</div>',
                "Final ranking of the 18 VLMs.",
            ],
            "bench leaderboard.json": ['"V": 18', '"max_borda": 85'],
        }.items():
            for needle in needles:
                assert needle in rendered[label], (label, needle)


def main():
    tests = [
        (name, value)
        for name, value in list(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failed = []
    for name, test in tests:
        try:
            test()
        except Exception:
            failed.append(name)
            print("FAIL  %s" % name)
            traceback.print_exc()
        else:
            print("PASS  %s" % name)
    print()
    print("%d passed, %d failed" % (len(tests) - len(failed), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
