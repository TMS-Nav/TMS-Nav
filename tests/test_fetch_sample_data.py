# the sample data fetcher, offline: a small fake archive stands in for the
# simnibs download, and every path points into a temporary folder
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from efield import config  # noqa: E402
from scripts import fetch_sample_data as fetch  # noqa: E402


def test_unpacks_and_copies_the_t1_from_a_local_archive():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        archive = tmp / "simnibs4_examples.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("simnibs4_examples/m2m_ernie/T1.nii.gz", b"not really a scan")
            z.writestr("simnibs4_examples/m2m_ernie/ernie.msh", b"")
        saves = tmp / "saves"
        saves.mkdir()
        old = fetch.SAVES, fetch.SAMPLE_DATA
        fetch.SAVES, fetch.SAMPLE_DATA = saves, saves / "sample_data"
        try:
            fetch.fetch("examples", from_zip=archive)
            assert (saves / "Ernie.nii.gz").read_bytes() == b"not really a scan"
            # a second run finds it unpacked and leaves the copy alone
            (saves / "Ernie.nii.gz").write_bytes(b"kept")
            fetch.fetch("examples", from_zip=tmp / "no_such.zip")
            assert (saves / "Ernie.nii.gz").read_bytes() == b"kept"
        finally:
            fetch.SAVES, fetch.SAMPLE_DATA = old
        # and the e field config finds the head model where it landed
        roots = config.sample_m2m_roots([saves / "sample_data"])
        assert roots == [saves / "sample_data" / "examples" / "simnibs4_examples"]
