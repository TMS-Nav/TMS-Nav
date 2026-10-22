#!/usr/bin/env python3
"""Generate an explicitly synthetic subject and run the complete ingestion path."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.subject_ingest import INPUT_ROOT, ingest_subject, validate_id
from src.synthetic_subject import make_phantom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', default='synthetic-demo')
    parser.add_argument('--five', action='store_true', help='Five cap markers with unequal intensities')
    args = parser.parse_args()
    validate_id(args.subject)
    folder = INPUT_ROOT / args.subject
    if (folder / 'T1w.nii.gz').exists():
        parser.error('Demo input exists; choose a new --subject rather than overwrite input')
    path, truth = make_phantom(folder, five=args.five)
    corrections = dict(subject=args.subject, axes='RAS', units='mm', source_sha256=truth['source_sha256'],
                       coordinate_frame=dict(name='subject_RAS'), fiducials=[],
                       landmarks=[dict(name=name, label=label, position=truth['landmarks_ras_mm'][label])
                                  for name, label in [('nasion', 'Nz'), ('inion', 'Iz'), ('LPA', 'LPA'), ('RPA', 'RPA')]])
    (folder / 'synthetic_anchor_corrections.json').write_text(json.dumps(corrections, indent=2))
    output, _ = ingest_subject(args.subject, path,
                 corrections=folder / 'synthetic_anchor_corrections.json',
                 skull_mask=folder / 'synthetic_skull_mask.nii.gz')
    payload = json.loads((output / 'markers.json').read_text())
    print(f'SYNTHETIC ONLY: {output}')
    print(f"Detected {len(payload['fiducials'])} marker candidates; {len(payload['electrodes'])} scalp electrode estimates")
    print('Ground-truth anchors were supplied explicitly. They were not detected from human anatomy.')


if __name__ == '__main__':
    main()
