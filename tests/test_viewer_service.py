"""Real loopback API tests, physical MRI alignment and interactive EEG correctness."""
from copy import deepcopy
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import nibabel as nib
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from src.fiducials import detect_markers
from src.subject_coordinates import file_hash, load_native
from src.subject_ingest import ingest_subject
from src.synthetic_subject import make_phantom
from src.viewer_service import Workspace, make_server


@pytest.fixture(scope='module')
def workspace(tmp_path_factory):
    root=tmp_path_factory.mktemp('review')
    path,truth=make_phantom(root/'inputs'/'review-test',five=True,oblique=True,reflected=True)
    ingest_subject('review-test',path,derivative_root=root/'outputs',viewer_root=None)
    work=Workspace(root/'outputs',root/'inputs',None)
    return work,truth,path


def payload(work,truth,confirmed=False):
    data=deepcopy(work.load('review-test')['markers'])
    for m in data['landmarks']:
        m['position']=truth['landmarks_ras_mm'][m['label']]
        m['status']='manually_confirmed' if confirmed else 'underdetermined'
    return data


def test_five_unequal_markers_and_rejection_diagnostics(workspace):
    work,truth,_=workspace
    entry=work.load('review-test')
    markers,qc=detect_markers(entry['volume'],entry['affine'],entry['mask'])
    assert len(markers)==5
    assert qc['count_matches_expectation']
    assert cdist(truth['fiducials_ras_mm'],[m['position'] for m in markers]).min(axis=1).max()<2
    rejected,qc=detect_markers(entry['volume'],entry['affine'],entry['mask'],max_distance=0,max_depth=0)
    assert len(rejected)==0
    assert sum(qc['rejection_counts'].values())>=5
    assert all(c['rejection_reasons'] for c in qc['components'])


def test_local_contrast_and_count_is_not_a_quota(workspace):
    work,_,_=workspace;entry=work.load('review-test')
    markers,qc=detect_markers(entry['volume'],entry['affine'],entry['mask'],mode='local',max_depth=8,max_distance=25,max_elongation=6)
    assert len(markers)>=5
    assert qc['threshold_units']=='local_contrast'
    none,qc=detect_markers(entry['volume'],entry['affine'],entry['mask'],threshold=10000)
    assert none==[] and not qc['count_matches_expectation']


def test_preview_changes_with_anchor_and_uses_cached_surface(workspace):
    work,truth,path=workspace;before=file_hash(path)
    surface=work.load('review-test')['surface']
    data=payload(work,truth)
    first=work.operation('review-test','preview',data)
    assert len(first['electrodes'])==22 and len(first['eeg_arcs'])==8
    assert first['eeg_qc']['status']=='provisional_unconfirmed_anchors'
    data['landmarks'][0]['position'][0]+=2
    second=work.operation('review-test','preview',data)
    assert first['electrodes']!=second['electrodes']
    assert work.load('review-test')['surface'] is surface
    assert file_hash(path)==before
    data['landmarks'][0]['position']=[10000,0,0]
    with pytest.raises(ValueError,match='outside'):work.operation('review-test','preview',data)


def test_http_volume_identity_and_restricted_access(workspace):
    work,_,_=workspace;server=make_server(0,work)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        raw=urlopen(base+'/api/subjects/review-test/volume.nii').read()
        image=nib.Nifti1Image.from_bytes(raw)
        entry=work.load('review-test')
        assert image.header.get_xyzt_units()[0]=='mm'
        assert np.allclose(image.affine,entry['affine'])
        assert np.array_equal(image.get_fdata(),entry['volume'])
        for headers in ({'Origin':'https://evil.example'},{'Host':'evil.example'}):
            with pytest.raises(HTTPError) as error:urlopen(Request(base+'/api/health',headers=headers))
            assert error.value.code==403
        with pytest.raises(HTTPError):urlopen(base+'/api/subjects/%2e%2e/meta')
        body=json.dumps({'source_sha256':'wrong'}).encode()
        with pytest.raises(HTTPError):urlopen(Request(base+'/api/subjects/review-test/preview',data=body,headers={'Content-Type':'application/json'}))
    finally:server.shutdown();server.server_close();thread.join()


def test_save_reviewed_corrections_and_reopen(workspace):
    work,truth,path=workspace;before=file_hash(path)
    data=payload(work,truth,True)
    for marker in data['fiducials']:marker['status']='manually_confirmed'
    result=work.operation('review-test','save',data)
    assert result['saved'] and len(result['markers']['electrodes'])==22
    assert all(m['status']=='manually_confirmed' for m in result['markers']['fiducials'])
    reopened=work.load('review-test')['markers']
    assert reopened==result['markers']
    assert (work.inputs/'review-test'/'corrections.json').exists()
    assert (work.inputs/'review-test'/'detector.json').exists()
    assert file_hash(path)==before


def test_faint_planar_voxel_sampling_keeps_fifth_sphere(tmp_path):
    from src.skull_boundary import head_mask
    path,truth=make_phantom(tmp_path,five=True)
    volume,affine,_=load_native(path);mask,_=head_mask(volume,affine)
    markers,qc=detect_markers(volume,affine,mask)
    assert len(markers)==5
    assert cdist(truth['fiducials_ras_mm'],[m['position'] for m in markers]).min(axis=1).max()<1


def test_optional_split_of_touching_bright_peaks():
    index=np.indices((40,40,40)).transpose(1,2,3,0)
    mask=np.linalg.norm(index-[20,20,20],axis=-1)<10
    volume=np.where(mask,100.,1.).astype(np.float32)
    for center in ([32,18,20],[32,24,20]):
        volume=np.maximum(volume,1000*np.exp(-np.sum((index-center)**2,axis=-1)/8))
    merged,_=detect_markers(volume,np.eye(4),mask,threshold=300,max_volume=2000)
    split,qc=detect_markers(volume,np.eye(4),mask,threshold=300,max_volume=2000,split_touching=True)
    assert len(merged)==1
    assert len(split)==2
    assert qc['parameters']['split_touching']


def test_browser_integer_options_preserve_detection_identity(workspace):
    from src.fiducials import detection_signature
    work,_,_=workspace;entry=work.load('review-test')
    default,qc=detect_markers(entry['volume'],entry['affine'],entry['mask'])
    explicit,other=detect_markers(entry['volume'],entry['affine'],entry['mask'],
        threshold=qc['threshold'],min_volume=8,max_volume=800,max_elongation=3,max_depth=4,max_distance=15)
    assert detection_signature(default,qc)==detection_signature(explicit,other)


def test_save_reuses_existing_registration_without_optimizing(workspace,tmp_path,monkeypatch):
    import shutil
    import src.subject_registration
    work,truth,source=workspace
    output=tmp_path/'out'/'review-test'
    shutil.copytree(work.derivatives/'review-test',output)
    manifest=json.loads((output/'manifest.json').read_text())
    manifest['input_paths']['reference']=str(source)
    (output/'manifest.json').write_text(json.dumps(manifest))
    matrix=np.eye(4);matrix[:3,3]=[3,4,5]
    (output/'transforms.json').write_text(json.dumps(dict(source_sha256=file_hash(source),
        target_sha256=file_hash(source),target_name='synthetic-reference',axes='RAS',units='mm',
        direction='native_to_MNI',matrix=matrix.tolist())))
    isolated=Workspace(tmp_path/'out',tmp_path/'inputs',None)
    def forbidden(*args,**kwargs):raise AssertionError('Existing registration must be reused')
    monkeypatch.setattr(src.subject_registration,'register_affine',forbidden)
    result=isolated.operation('review-test','save',payload(isolated,truth,True))
    for marker in result['markers']['landmarks']:
        assert np.allclose(marker['mni_position'],np.array(marker['position'])+[3,4,5])
