from datetime import date

from aeroreg.ingest import latest_issue_date


def test_latest_issue_date_picks_title_14():
    titles = {
        "titles": [
            {"number": 1, "latest_issue_date": "2026-01-01"},
            {"number": 14, "latest_issue_date": "2026-09-30"},
        ]
    }
    assert latest_issue_date(titles) == date(2026, 9, 30)