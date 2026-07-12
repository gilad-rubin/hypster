import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"
SUMMARY = DOCS / "SUMMARY.md"
MIGRATION_PAGES = (
    DOCS / "migration" / "upgrade-0.7-to-0.8.md",
    DOCS / "migration" / "upgrade-0.8-to-0.9.md",
)
NEW_VISIBLE_PAGES = (
    *MIGRATION_PAGES,
    DOCS / "reference" / "currently-tested-environments.md",
    DOCS / "contributing" / "releasing.md",
)
MARKED_PYTHON = re.compile(r"<!-- test: exec -->\s*```python\n(.*?)\n```", re.DOTALL)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def test_issue_89_pages_are_in_gitbook_summary() -> None:
    summary = SUMMARY.read_text()

    for page in NEW_VISIBLE_PAGES:
        relative = page.relative_to(DOCS).as_posix()
        assert f"]({relative})" in summary, f"{relative} is missing from docs/SUMMARY.md"


@pytest.mark.parametrize("page", NEW_VISIBLE_PAGES, ids=lambda path: path.name)
def test_issue_89_pages_have_no_broken_local_links(page: Path) -> None:
    for target in MARKDOWN_LINK.findall(page.read_text()):
        if "://" in target or target.startswith(("#", "mailto:")):
            continue
        path_text = target.split("#", 1)[0]
        if not path_text:
            continue
        resolved = (page.parent / path_text).resolve()
        assert resolved.exists(), f"{page.relative_to(ROOT)} links to missing {target}"


@pytest.mark.parametrize("page", MIGRATION_PAGES, ids=lambda path: path.name)
def test_migration_pages_have_runnable_current_api_examples(page: Path) -> None:
    snippets = MARKED_PYTHON.findall(page.read_text())
    assert snippets, f"{page.relative_to(ROOT)} has no marked runnable Python examples"

    for index, snippet in enumerate(snippets, start=1):
        namespace = {"__name__": f"docs_example_{page.stem}_{index}"}
        exec(compile(snippet, f"{page}::snippet-{index}", "exec"), namespace)
