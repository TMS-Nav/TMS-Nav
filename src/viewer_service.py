"""Loopback-only API for local MRI inspection, candidate review and EEG previews."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import unquote, urlsplit

import nibabel as nib
import numpy as np
import pyvista as pv

from src.fiducials import detect_markers, detection_signature
from src.mri_inputs import TEMPLATE_DIR
from src.subject_coordinates import apply_affine, file_hash, load_native
from src.subject_ingest import (DERIVATIVE_ROOT, INPUT_ROOT, VIEWER_ROOT, apply_corrections,
                                infer_electrodes, ingest_subject, validate_id, write_json)


class Workspace:
    def __init__(self, derivatives=DERIVATIVE_ROOT, inputs=INPUT_ROOT, viewer=VIEWER_ROOT):
        self.derivatives, self.inputs = Path(derivatives), Path(inputs)
        self.viewer = Path(viewer) if viewer is not None else None
        self.cache = {}
        self.lock = threading.RLock()  # Serialize VTK operations and saves.

    def load(self, subject):
        template = subject == 'MNI152'
        if not template: validate_id(subject)
        output = self.derivatives / subject
        if template:
            paths = list(TEMPLATE_DIR.rglob('mni_icbm152_t1_tal_nlin_sym_09a.nii.gz'))
            if not paths: raise ValueError('Generate the MNI template first')
            source = paths[0]
            manifest = None
        else:
            manifest = json.loads((output / 'manifest.json').read_text())
            source = Path(manifest['input_paths']['anatomical'])
        stamp = (source.stat().st_mtime_ns, source.stat().st_size,
                 (output / 'manifest.json').stat().st_mtime_ns if manifest else 0)
        if subject in self.cache and self.cache[subject]['stamp'] == stamp:
            return self.cache[subject]
        if manifest and file_hash(source) != manifest['fingerprint']['inputs']['anatomical']:
            raise ValueError('Native input changed; rerun ingestion before reviewing it')
        volume, affine, metadata = load_native(source, allow_template=template)
        entry = dict(stamp=stamp, volume=volume, affine=affine, metadata=metadata, source=source, manifest=manifest)
        if not template:
            entry.update(mask=nib.load(output / 'head_mask.nii.gz').get_fdata() > 0,
                         surface=pv.read(output / 'scalp.vtp'),
                         markers=json.loads((output / 'markers.json').read_text()))
        # Keep only two active volumes; original inputs are never modified.
        if len(self.cache) >= 2: self.cache.pop(next(iter(self.cache)))
        self.cache[subject] = entry
        return entry

    def points(self, entry, payload):
        data = deepcopy(payload)
        for category in ('fiducials', 'landmarks'):
            names = set()
            for marker in data.get(category, []):
                xyz = np.asarray(marker['position'], dtype=float)
                if xyz.shape != (3,) or not np.isfinite(xyz).all(): raise ValueError('Invalid point coordinate')
                ijk = apply_affine(np.linalg.inv(entry['affine']), xyz)
                if ((ijk < -.5) | (ijk > np.array(entry['volume'].shape) - .5)).any():
                    raise ValueError('Point is outside the native image')
                if marker['name'] in names: raise ValueError('Duplicate point names')
                names.add(marker['name'])
        labels = [m['label'] for m in data.get('landmarks', [])]
        if len(set(labels)) != len(labels): raise ValueError('Duplicate anatomical landmark labels')
        return data

    def operation(self, subject, action, body):
        with self.lock:
            if subject == 'MNI152': raise ValueError('Subject review is unavailable on the MNI demo')
            entry = self.load(subject)
            if body.get('source_sha256') != entry['markers']['source_sha256']:
                raise ValueError('Stale subject image identity; reload this subject')
            if action == 'detect':
                options = body.get('options', {})
                markers, qc = detect_markers(entry['volume'], entry['affine'], entry['mask'], body.get('threshold'), **options)
                return dict(fiducials=markers, detection_qc=qc, detection_signature=detection_signature(markers, qc))
            data = self.points(entry, body)
            data['electrodes'], data['eeg_arcs'], data['eeg_qc'] = [], [], {}
            infer_electrodes(entry['surface'], data, preview=True)
            if action == 'preview':
                return {k: data[k] for k in ('electrodes', 'eeg_arcs', 'eeg_qc')}
            if action != 'save': raise ValueError('Unknown action')
            options, threshold = body.get('detector_options', {}), body.get('marker_threshold')
            detected, qc = detect_markers(entry['volume'], entry['affine'], entry['mask'], threshold, **options)
            signature = detection_signature(detected, qc)
            if body.get('detection_signature') != signature:
                raise ValueError('Detector settings or candidate identity changed; rerun detection and review')
            corrections = dict(subject=subject, axes='RAS', units='mm', coordinate_frame=dict(name='subject_RAS'),
                               source_sha256=body['source_sha256'], detection_signature=signature,
                               landmarks=[m for m in data['landmarks'] if m['status']=='manually_confirmed'],
                               fiducials=[m for m in data['fiducials'] if m['status']=='manually_confirmed'])
            corrections['fiducials'] += [dict(name=name, exclude=True) for name in body.get('excluded_fiducials', [])]
            # Validate the correction contract before writing a reviewed input.
            folder = self.inputs / subject
            folder.mkdir(parents=True, exist_ok=True)
            correction_path = folder / 'corrections.json'
            detector_path = folder / 'detector.json'
            previous = {p: p.read_bytes() if p.exists() else None for p in (correction_path, detector_path)}
            try:
                write_json(correction_path, corrections)
                check = dict(subject=subject, detection_signature=signature, fiducials=deepcopy(detected),
                             landmarks=deepcopy(entry['markers']['landmarks']))
                apply_corrections(check, correction_path, body['source_sha256'])
                write_json(detector_path, dict(options=options, threshold=threshold))
                old = entry['manifest']['input_paths']
                transform, reference = old.get('transform'), old.get('reference')
                if reference and not transform:
                    # Save/review must not rerun an existing MNI registration.
                    reuse = folder / '.review-registration.json'
                    previous[reuse] = reuse.read_bytes() if reuse.exists() else None
                    write_json(reuse, json.loads((self.derivatives/subject/'transforms.json').read_text()))
                    transform, reference = reuse, None
                output, _ = ingest_subject(subject, entry['source'], input_root=self.inputs,
                    derivative_root=self.derivatives,
                    viewer_root=self.viewer if body.get('export_viewer', True) else None, corrections=correction_path,
                    marker_threshold=threshold, detector_options=options,
                    skull_mask=old.get('skull_mask'), transform=transform, register_to=reference)
            except BaseException:
                for path, value in previous.items():
                    if value is None: path.unlink(missing_ok=True)
                    else: path.write_bytes(value)
                raise
            self.cache.pop(subject, None)
            return dict(saved=True, markers=json.loads((output/'markers.json').read_text()))


def make_server(port=8765, workspace=None):
    workspace = workspace or Workspace()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass  # Do not log subject IDs or coordinates.

        def send(self, status, payload, content_type='application/json'):
            raw = json.dumps(payload, allow_nan=False).encode() if content_type=='application/json' else payload
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers(); self.wfile.write(raw)

        def dispatch(self):
            host = self.headers.get('Host', '').split(':')[0]
            if host not in ('127.0.0.1', 'localhost'): raise PermissionError('Loopback Host required')
            origin = self.headers.get('Origin')
            if origin and origin not in {f'http://{h}:{p}' for h in ('localhost','127.0.0.1') for p in (1234,self.server.server_port)}:
                raise PermissionError('Untrusted request origin')
            path = urlsplit(self.path).path
            if path == '/api/health': return self.send(200, dict(ready=True))
            parts = path.split('/')
            if len(parts)!=5 or parts[1:3]!=['api','subjects']: raise ValueError('Unknown API route')
            subject, action = unquote(parts[3]), parts[4]
            with workspace.lock:
                entry = workspace.load(subject)
                if self.command=='GET':
                    if action=='meta':
                        return self.send(200, dict(**entry['metadata'],
                            intensity_range=np.percentile(entry['volume'], [1,99.8]).tolist(),
                            markers=entry.get('markers'), service=True))
                    if action=='volume.nii':
                        image=nib.Nifti1Image(entry['volume'],entry['affine']); image.header.set_xyzt_units('mm')
                        return self.send(200,image.to_bytes(),'application/octet-stream')
                    raise ValueError('Unknown resource')
            if self.headers.get('Content-Type','').split(';')[0]!='application/json': raise ValueError('JSON required')
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=2_000_000: raise ValueError('Request exceeds size limit')
            body=json.loads(self.rfile.read(size))
            return self.send(200,workspace.operation(subject,action,body))

        def handle_request(self):
            try: self.dispatch()
            except (BrokenPipeError, ConnectionResetError): pass
            except PermissionError as exc: self.send(403,dict(error=str(exc)))
            except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
                # Avoid exposing source paths in file-system errors.
                self.send(400,dict(error='Local subject data unavailable; ingest this subject first' if isinstance(exc,OSError) else str(exc)))
        do_GET=handle_request
        do_POST=handle_request
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)
