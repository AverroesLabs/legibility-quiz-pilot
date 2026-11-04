import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/quiz"))
import quiz


def test_sections_split_front_matter():
    s = quiz.sections("Title\n\nAbs\n\n## 1 Intro\nx\n### 1.1 Sub\ny\n## 2 Method\nz\n")
    assert [x["id"] for x in s] == ["s0", "s1", "s2"]
    assert "Sub" in s[1]["text"] and s[2]["title"] == "2 Method"


def test_parse_answers_variants():
    c = "preamble\nANSWER Q1: yes because\nit is\n**ANSWER Q2:** no\n\nanswer q4: maybe\nANSWER Q9: out of range\n"
    got = quiz.parse_answers(c, 4)
    assert got == {1: "yes because\nit is", 2: "no", 4: "maybe"}


def test_scramble_keeps_paragraphs():
    t = "a\n\nb\n\nc\n\nd"
    assert sorted(quiz.paragraphs(quiz.scramble(t, 1))) == ["a", "b", "c", "d"]


def test_retrieve_picks_overlap():
    t = "cats sleep often\n\nthe gradient norm explodes\n\ndogs bark"
    assert "gradient" in quiz.retrieve(t, "why does the gradient norm grow", k=1)


def test_extract_json_fenced():
    assert quiz.extract_json('```json\n{"a": 1}\n```') == {"a": 1}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")


import versions


def test_numbers_ignore_citation_years():
    t = "We use 128 tokens (Ouyang et al., 2022) and Bai et al. (2022) at p=0.25, 1,024 steps and 3.5% gain."
    assert versions.numbers(t) == {"128", "0.25", "1024", "3.5"}


def test_missing_numbers():
    assert versions.missing_numbers("a 12 b 0.5 c 7", "x 12 y 7") == ["0.5"]


def test_split_front_keeps_authors():
    s = {"id": "s0", "text": "# T\n\nAuthor A\n\n###### Abstract\n\nWe do x.\n"}
    head, body = versions.split_front(s)
    assert "Author A" in head and body == "We do x."
