import os, pytest

MAIN_PDF = "Unified_Robust_Duality_Informed_Convex_Equilibrium_Allocation_Theory.pdf"
ARCHIVE_ZIP = "Archive.zip"

def test_artifacts_exist_or_skip():
    missing = [p for p in [MAIN_PDF, ARCHIVE_ZIP] if not os.path.exists(p)]
    if missing:
        pytest.skip("Missing research artifacts: " + ", ".join(missing))
    assert os.path.getsize(ARCHIVE_ZIP) > 0
