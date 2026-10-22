#!/usr/bin/env python3
"""Subject-specific T1w reconstruction CLI; no fMRI or MNI download required."""
import argparse

from src.subject_ingest import ingest_subject


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', help='Unique subject ID; dedicated default input: saves/subjects/ID/T1w.nii.gz')
    parser.add_argument('anatomical', nargs='?', help='Native T1w NIfTI; omitted to use the dedicated subject folder')
    parser.add_argument('--corrections', help='Reviewed native RAS mm correction JSON (also auto-read from subject folder)')
    parser.add_argument('--transform', help='Explicit physical native↔MNI affine JSON; see docs/subject-reconstruction.md')
    parser.add_argument('--register-to', help='Explicit full-head MNI T1w reference NIfTI; reproducible affine registration')
    parser.add_argument('--skull-mask', help='Validated binary skull mask on the native anatomical grid')
    parser.add_argument('--marker-threshold', type=float, help='Override bright marker threshold in original intensity units')
    parser.add_argument('--force', action='store_true', help='Reprocess even if content hashes and options match')
    args = parser.parse_args()
    try:
        output, cached = ingest_subject(**vars(args))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'Ingestion failed: {exc}\n')
    print(f'{"Cached" if cached else "Reconstructed"}: {output}')
    print('Review qc.json, qc_projection.png and qc_fid-*.png before confirming landmarks.')
    print('Viewer updated. Run: cd viewer && npm run dev')


if __name__ == '__main__':
    main()
