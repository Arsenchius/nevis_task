from __future__ import annotations

import re
from pathlib import Path
from typing import Any


SPEAKER_LINE_PATTERN = re.compile(r"^(ADVISOR|CLIENT1|CLIENT2): \[\d{2}:\d{2}:\d{2}\]")
SPEAKER_WITHOUT_TIMESTAMP_PATTERN = re.compile(r"^(ADVISOR|CLIENT1|CLIENT2):\s+(?!\[\d{2}:\d{2}:\d{2}\])")
NON_US_TERM_PATTERNS = {
    "ISA": re.compile(r"\bISAs?\b", re.IGNORECASE),
    "SIPP": re.compile(r"\bSIPPs?\b", re.IGNORECASE),
    "tax-free lump sum": re.compile(r"\btax-free lump sum\b", re.IGNORECASE),
    "pension commencement lump sum": re.compile(
        r"\bpension commencement lump sum\b",
        re.IGNORECASE,
    ),
}


def audit_transcript_format(transcript: str) -> dict[str, Any]:
    lines = transcript.splitlines()
    speaker_lines = [
        line for line in lines if line.startswith(("ADVISOR:", "CLIENT1:", "CLIENT2:"))
    ]
    speaker_lines_without_timestamps = [
        line for line in speaker_lines if not SPEAKER_LINE_PATTERN.match(line)
    ]
    non_us_terms = [
        term for term, pattern in NON_US_TERM_PATTERNS.items() if pattern.search(transcript)
    ]

    return {
        "has_metadata_header": transcript.lstrip().startswith("---"),
        "speaker_line_count": len(speaker_lines),
        "speaker_lines_without_timestamps": speaker_lines_without_timestamps[:10],
        "speaker_lines_without_timestamps_count": len(speaker_lines_without_timestamps),
        "non_us_terms": non_us_terms,
    }


def audit_transcript_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    return {"path": str(file_path), **audit_transcript_format(file_path.read_text())}
