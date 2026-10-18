"""Download 14 CFR parts from the public eCFR API and parse them into sections."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

ECFR_FULL_URL = "https://www.ecfr.gov/api/versioner/v1/full/{day}/title-14.xml"

_SECTION_NUMBER = re.compile(r"§+\s*(\d+\.\d+[A-Za-z0-9-]*)")

@dataclass(frozen=True)
class Section:
    """One regulatory section, e.g. 14 CFR § 25.1309."""

    part: str
    number: str  # "25.1309"
    heading: str
    subpart: str
    text: str

    @property
    def citation(self) -> str:
        return f"14 CFR § {self.number}"

ECFR_TITLES_URL = "https://www.ecfr.gov/api/versioner/v1/titles.json"

def latest_issue_date(titles: dict, title: int = 14) -> date:
    """Latest date eCFR can serve for a title (requests for later dates return 404)."""
    for t in titles["titles"]:
        if t["number"] == title:
            return date.fromisoformat(t["latest_issue_date"])
    raise ValueError(f"Title {title} not found in eCFR titles listing")

def download_part(part: str, dest_dir: Path, day: date | None = None) -> Path:
    """Download one part of Title 14 as XML into dest_dir and return the file path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"title14_part{part}.xml"
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        if day is None:
            resp = client.get(ECFR_TITLES_URL)
            resp.raise_for_status()
            day = latest_issue_date(resp.json())
        resp = client.get(ECFR_FULL_URL.format(day=day.isoformat()), params={"part": part})
        resp.raise_for_status()
        out.write_bytes(resp.content)
    return out

def _clean(text: str) -> str:
    return " ".join(text.split())

def _element_text(el: ET.Element) -> str:
    return _clean("".join(el.itertext()))

def parse_sections(xml_text: str | bytes) -> list[Section]:
    """Parse eCFR XML into a list of sections (appendices are not included)."""
    root = ET.fromstring(xml_text)
    parents = {child: parent for parent in root.iter() for child in parent}

    sections: list[Section] = []
    for el in root.iter("DIV8"):
        if el.get("TYPE") != "SECTION":
            continue

        head_el = el.find("HEAD")
        head_text = _element_text(head_el) if head_el is not None else ""

        match = _SECTION_NUMBER.search(el.get("N", "")) or _SECTION_NUMBER.search(head_text)
        if not match:
            continue
        number = match.group(1)
        heading = _clean(_SECTION_NUMBER.sub("", head_text, count=1)).strip(" .—-")

        subpart = ""
        node = parents.get(el)
        while node is not None:
            if node.tag == "DIV6":
                sub_head = node.find("HEAD")
                subpart = _element_text(sub_head) if sub_head is not None else ""
                break
            node = parents.get(node)

        paragraphs = [
            _element_text(child) for child in el if child.tag != "HEAD" and _element_text(child)
        ]
        if not paragraphs:
            continue  # reserved sections such as "[Reserved]" with no body

        sections.append(
            Section(
                part=number.split(".")[0],
                number=number,
                heading=heading,
                subpart=subpart,
                text="\n".join(paragraphs),
            )
        )
    return sections
