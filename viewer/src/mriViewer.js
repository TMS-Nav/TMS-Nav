import {Niivue, NVImage, SLICE_TYPE, SHOW_RENDER} from '@niivue/niivue';

export function createMriViewer({onPlace, onAdd}) {
  let nv=null, key='', loadedKey='', token=0, overlays=null, markers=null, location=[0,0,0], metadata=null, pendingFocus=null, loadQueue=Promise.resolve(), loading=false;
  const el=id=>document.getElementById(id);
  const status=text=>{el('mri-status').textContent=text;};
  const layout={multi:SLICE_TYPE.MULTIPLANAR, axial:SLICE_TYPE.AXIAL, coronal:SLICE_TYPE.CORONAL, sagittal:SLICE_TYPE.SAGITTAL};
  const active=()=>!el('mri-panel').hidden;
  function focus(position) {
    pendingFocus=position;
    if(!nv || loadedKey!==key) return;
    nv.scene.crosshairPos=nv.mm2frac(position); nv.drawScene(); syncLocation();
  }
  function syncLocation() {
    if(!nv?.volumes.length || loadedKey!==key) return;
    location=Array.from(nv.frac2mm(nv.scene.crosshairPos)).slice(0,3);
    ['x','y','z'].forEach((axis,i)=>{el('slice-'+axis).value=nv.scene.crosshairPos[i]*1000;});
    const voxel=Array.from(nv.volumes[0].mm2vox(location));
    const intensity=nv.volumes[0].getValue(...voxel);
    status(`${key==='MNI152'?'MNI':'Native'} RAS mm: ${location.map(x=>x.toFixed(2)).join(', ')} · intensity ${Number(intensity).toFixed(2)} · scroll slices; right-drag window/level`);
  }
  function updateOverlays(data) {
    markers=data;
    if(!nv || loadedKey!==key || !metadata) return;
    if(overlays) nv.removeVolume(overlays);
    overlays=null;
    if(!el('mri-overlays').checked) return;
    const image=NVImage.zerosLike(nv.volumes[0],'uint8');
    image.setColormapLabel({I:[0,1,2],R:[0,239,242],G:[0,117,193],B:[0,198,78],A:[0,255,255],labels:['','MRI fiducial','Anatomical landmark']});
    const [nx,ny,nz]=metadata.shape, spacing=metadata.voxel_to_native_ras_mm.slice(0,3).map((_,i)=>Math.hypot(...metadata.voxel_to_native_ras_mm.slice(0,3).map(row=>row[i])));
    for(const [category,value] of [['fiducials',1],['landmarks',2]]) for(const marker of data?.[category]||[]) {
      const center=Array.from(nv.volumes[0].mm2vox(marker.position,true));
      const radius=spacing.map(s=>Math.ceil(3/s));
      for(let z=Math.max(0,Math.floor(center[2])-radius[2]);z<=Math.min(nz-1,Math.ceil(center[2])+radius[2]);z++)
        for(let y=Math.max(0,Math.floor(center[1])-radius[1]);y<=Math.min(ny-1,Math.ceil(center[1])+radius[1]);y++)
          for(let x=Math.max(0,Math.floor(center[0])-radius[0]);x<=Math.min(nx-1,Math.ceil(center[0])+radius[0]);x++)
            if([x,y,z].reduce((s,v,i)=>s+((v-center[i])*spacing[i])**2,0)<=9) image.img[x+nx*(y+ny*z)]=value;
    }
    overlays=image; nv.addVolume(image);
  }
  async function open() {
    el('mri-panel').hidden=false; el('stage').classList.add('mri-active');
    el('tab-mri').setAttribute('aria-selected','true');el('tab-head').setAttribute('aria-selected','false');
    if(loadedKey===key && !loading) {nv?.resizeListener();return;}
    const request=++token, selected=key;
    status('Loading native anatomical T1w…');
    el('mri-canvas').style.visibility='hidden';
    // Serialize volume uploads: an older load must never replace the latest image.
    loadQueue=loadQueue.catch(()=>{}).then(async()=>{
    if(request!==token)return;
    loading=true;
    try {
      const response=await fetch(`/api/subjects/${encodeURIComponent(key)}/meta`);
      if(!response.ok) throw new Error('Start the local viewer with: micromamba run -n tms312 python run_viewer.py');
      const meta=await response.json();
      if(request!==token) return;
      if(!nv) {
        nv=new Niivue({isResizeCanvas:true,multiplanarShowRender:SHOW_RENDER.NEVER,isColorbar:false,isRadiologicalConvention:false});
        // Retain complete frames for QC screenshots; 2D slices do not need MSAA.
        const gl=el('mri-canvas').getContext('webgl2',{alpha:false,antialias:false,preserveDrawingBuffer:true});
        if(!gl)throw new Error('Anatomical MRI requires WebGL 2.');
        await nv.attachToCanvas(el('mri-canvas'),false);
        nv.onLocationChange=syncLocation;
      }
      await nv.loadVolumes([{url:`/api/subjects/${encodeURIComponent(selected)}/volume.nii`,name:'native-T1w.nii',colormap:'gray'}]);
      if(request!==token) {loadedKey='';return;}
      metadata=meta;loadedKey=selected;
      nv.setSliceType(layout[el('mri-layout').value]);
      el('mri-black').value=meta.intensity_range[0];el('mri-white').value=meta.intensity_range[1];
      contrast();
      nv.resizeListener();updateOverlays(markers);
      if(pendingFocus) focus(pendingFocus); else syncLocation();
      el('mri-panel').dataset.subject=key;
      el('mri-canvas').style.visibility='visible';
    } catch(error) {if(request===token)status(error.message);}
    finally {loading=false;}
    });
    return loadQueue;
  }
  function contrast() {
    if(!nv?.volumes.length)return;
    const low=Number(el('mri-black').value), high=Number(el('mri-white').value);
    if(!Number.isFinite(low)||!Number.isFinite(high)||low>=high)return;
    nv.volumes[0].cal_min=low;nv.volumes[0].cal_max=high;nv.updateGLVolume();
  }
  el('tab-mri').onclick=open;
  el('tab-head').onclick=()=>{el('mri-panel').hidden=true;el('stage').classList.remove('mri-active');el('tab-head').setAttribute('aria-selected','true');el('tab-mri').setAttribute('aria-selected','false');};
  el('mri-layout').onchange=()=>nv?.setSliceType(layout[el('mri-layout').value]);
  el('mri-radiological').onchange=()=>nv?.setRadiologicalConvention(el('mri-radiological').checked);
  el('mri-overlays').onchange=()=>updateOverlays(markers);
  ['black','white'].forEach(id=>el('mri-'+id).onchange=contrast);
  el('mri-zoom').oninput=()=>{
    if(!nv?.volumes.length)return;
    const pan=Array.from(nv.scene.pan2Dxyzmm);pan[3]=Number(el('mri-zoom').value)/100;nv.setPan2Dxyzmm(pan);
  };
  ['x','y','z'].forEach((axis,i)=>el('slice-'+axis).oninput=()=>{
    if(!nv?.volumes.length||loadedKey!==key)return;
    nv.scene.crosshairPos[i]=Number(el('slice-'+axis).value)/1000;nv.drawScene();syncLocation();
  });
  el('mri-reset').onclick=()=>{if(nv){el('mri-zoom').value=100;nv.scene.crosshairPos=[.5,.5,.5];nv.setPan2Dxyzmm([0,0,0,1]);nv.drawScene();syncLocation();}};
  el('mri-place').onclick=()=>{if(loadedKey===key)onPlace(location);};
  el('mri-add').onclick=()=>{if(loadedKey===key)onAdd(location);};
  return {focus,open,updateOverlays,load(subject,data){key=subject;markers=data;pendingFocus=null;loadedKey='';token++;el('mri-canvas').style.visibility='hidden';el('mri-panel').removeAttribute('data-subject');if(active())open();}};
}
