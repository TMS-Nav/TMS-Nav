import {createMriViewer} from './mriViewer.js';

const el=id=>document.getElementById(id);
async function api(key,action,body) {
  const response=await fetch(`/api/subjects/${encodeURIComponent(key)}/${action}`,body?{
    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});
  let result;
  try {result=await response.json();} catch {throw new Error('Start the local viewer with: micromamba run -n tms312 python run_viewer.py');}
  if(!response.ok)throw new Error(result.error||'Local review request failed');
  return result;
}

export function createSubjectReview(hooks) {
  let key='', template=true, available=false, revision=0, timer=null, detectorOptions={}, threshold=null,
      excluded=[], undo=null, loadRevision=0, editRevision=0;
  const status=text=>{el('review-status').textContent=text;};
  const mri=createMriViewer({onPlace:position=>hooks.place(position),onAdd:position=>add(position)});
  function options() {
    const result={mode:el('detect-mode').value,min_volume:Number(el('detect-min-volume').value),
      max_volume:Number(el('detect-max-volume').value),max_elongation:Number(el('detect-elongation').value),
      max_depth:Number(el('detect-depth').value),max_distance:Number(el('detect-distance').value),
      expected_count:Number(el('detect-expected').value),split_touching:el('detect-split').checked};
    return {options:result,threshold:el('detect-threshold').value.trim()===''?null:Number(el('detect-threshold').value)};
  }
  function setOptions(params={},cut=null) {
    const fields={mode:'mode',min_volume:'min-volume',max_volume:'max-volume',max_elongation:'elongation',
      max_depth:'depth',max_distance:'distance',expected_count:'expected'};
    const defaults={mode:'bright',min_volume:8,max_volume:800,max_elongation:3,max_depth:4,max_distance:15,expected_count:5};
    for(const [k,suffix] of Object.entries(fields))el('detect-'+suffix).value=params[k]??defaults[k];
    el('detect-split').checked=!!params.split_touching;el('detect-threshold').value=cut??'';
  }
  el('detect-profile').onchange=()=>{
    if(el('detect-profile').value==='liquimark') {
      setOptions({mode:'local',min_volume:8,max_volume:800,max_elongation:8,max_depth:8,max_distance:25,expected_count:5},null);
      status('Starting filters for an 8 mm round marker, not a validated detector. Preview, inspect each center, then confirm.');
    }
  };
  const body=()=>{
    const data=hooks.getData();
    return {source_sha256:data.source_sha256,detection_signature:data.detection_signature,
      landmarks:data.landmarks,fiducials:data.fiducials,detector_options:detectorOptions,
      marker_threshold:threshold,excluded_fiducials:excluded,export_viewer:false};
  };
  function replace(fiducials,patch={}) {editRevision++;hooks.replaceFiducials(fiducials,patch);render();}
  function add(position) {
    if(template)return;
    const data=hooks.getData();let i=1;while(data.fiducials.some(m=>m.name===`manual-${i}`))i++;
    const marker={name:`manual-${i}`,label:`manual-${i}`,position:[...position],normal:[0,0,1],category:'fiducial',
      color:'#ef75c6',status:'candidate_requires_review',confidence:0,confidence_type:'manual_candidate_requires_confirmation',method:'manual_crosshair'};
    replace([...data.fiducials,marker]);hooks.select(marker.name);status('Manual center added. Inspect all planes, then Confirm.');
  }
  function render() {
    if(template)return;
    const data=hooks.getData(), count=data.fiducials.filter(m=>m.status==='manually_confirmed').length;
    const expected=detectorOptions.expected_count??5;
    el('review-count').textContent=`${count}/${expected} confirmed · ${data.fiducials.length} active candidates`;
    const tbody=el('candidate-table').querySelector('tbody');tbody.replaceChildren();
    function row(marker, rejected=false) {
      const tr=document.createElement('tr'), identity=document.createElement('td'), measures=document.createElement('td'), actions=document.createElement('td');
      const button=(text,callback)=>{const b=document.createElement('button');b.textContent=text;b.onclick=callback;return b;};
      identity.appendChild(button(marker.name||`component ${marker.component}`,()=>{mri.focus(marker.position);mri.open();if(marker.name)hooks.select(marker.name);}));
      const note=document.createElement('div');note.textContent=rejected?marker.rejection_reasons.join(', ').replaceAll('_',' '):marker.status.replaceAll('_',' ');identity.appendChild(note);
      measures.textContent=`${marker.volume_mm3??'manual'} / ${marker.elongation??'—'}`;
      if(rejected) actions.appendChild(button('Include',()=>{
        const data=hooks.getData(), name='review-'+marker.component.replace('.','-');
        if(!data.fiducials.some(m=>m.name===name))replace([...data.fiducials,{...marker,name,label:name,category:'fiducial',normal:[0,0,1],color:'#ef75c6',status:'candidate_requires_review',confidence:0,method:'reviewed_rejected_component'}]);
      }));
      else {
        actions.appendChild(button('Confirm',()=>{
          const updated=hooks.getData().fiducials.map(m=>m.name===marker.name?{...m,status:'manually_confirmed',confidence:1,confidence_type:'human_confirmation_not_probability'}:m);
          replace(updated);status('Marker confirmed from MRI review; save to persist derivatives.');
        }));
        actions.appendChild(button('Exclude',()=>{excluded.push(marker.name);replace(hooks.getData().fiducials.filter(m=>m.name!==marker.name));}));
      }
      tr.append(identity,measures,actions);tbody.appendChild(tr);
    }
    data.fiducials.forEach(m=>row(m));
    for(const candidate of data.detection_qc?.components||[])if(!candidate.accepted)row(candidate,true);
    mri.updateOverlays(data);
  }
  async function preview(request,subject) {
    try {
      if(!available)throw new Error('Start run_viewer.py for live EEG previews; existing exports still work offline.');
      const result=await api(subject,'preview',body());
      if(request!==revision||subject!==key)return;
      hooks.updateEeg(result);
    } catch(error){if(request===revision&&subject===key)hooks.previewError(error.message);}
  }
  function changed(marker) {
    editRevision++;
    render();mri.focus(marker.position);
    if(template||marker.category!=='landmark')return;
    clearTimeout(timer);const request=++revision,subject=key;
    hooks.previewPending();timer=setTimeout(()=>preview(request,subject),300);
  }
  el('detect-run').onclick=async()=>{
    const subject=key,request=++revision, edits=editRevision;clearTimeout(timer);
    status('Detecting candidates…');
    try {
      const settings=options();
      const result=await api(subject,'detect',{source_sha256:hooks.getData().source_sha256,...settings});
      if(request!==revision||subject!==key)return;
      if(edits!==editRevision){status('New point edits were kept. Preview detection again to replace candidates.');return;}
      undo={data:structuredClone(hooks.getData()),options:detectorOptions,threshold,excluded:[...excluded]};
      detectorOptions=settings.options;threshold=settings.threshold;excluded=[];
      replace(result.fiducials,result);
      status(`Threshold ${result.detection_qc.threshold.toFixed(2)} (${result.detection_qc.threshold_units}); rejections: ${JSON.stringify(result.detection_qc.rejection_counts)}. Review each center.`);
      el('detect-undo').hidden=false;
    }catch(error){if(request===revision)status(error.message);}
  };
  const undoButton=document.createElement('button');undoButton.id='detect-undo';undoButton.textContent='Undo detection preview';undoButton.hidden=true;
  el('detect-run').after(undoButton);
  undoButton.onclick=()=>{if(!undo)return;revision++;detectorOptions=undo.options;threshold=undo.threshold;excluded=undo.excluded;replace(undo.data.fiducials,undo.data);setOptions(detectorOptions,threshold);undo=null;undoButton.hidden=true;status('Restored previous candidates and review edits.');};
  el('review-save').onclick=async()=>{
    const subject=key, edits=editRevision;status('Saving reviewed corrections and rebuilding derivatives…');el('review-save').disabled=true;
    try {
      const result=await api(subject,'save',body());
      if(subject!==key)return;
      if(edits!==editRevision){status('Saved the submitted snapshot; newer browser edits remain unsaved.');return;}
      excluded=[];hooks.replaceMarkers(result.markers);render();hooks.updateEeg(result.markers);
      undo=null;undoButton.hidden=true;status('Saved corrections and subject derivatives.');
    }catch(error){if(subject===key)status(error.message);}
    finally{el('review-save').disabled=false;}
  };
  return {changed,focus:mri.focus,
    async load(subject,isTemplate){
      const loadRequest=++loadRevision;
      const initialEdits=editRevision;
      key=subject;template=isTemplate;revision++;clearTimeout(timer);available=false;excluded=[];undo=null;undoButton.hidden=true;
      el('detect-run').disabled=true;el('review-save').disabled=true;
      el('review-panel').hidden=template;el('metrics-panel').hidden=!template;
      mri.load(subject,hooks.getData());
      if(template)return;
      const data=hooks.getData();detectorOptions=data.detection_qc?.parameters||{};
      threshold=data.detection_qc?.parameters?data.detection_qc.threshold:null;
      setOptions(detectorOptions,threshold);render();status('Connecting to local MRI/EEG service…');
      try {
        const meta=await api(subject,'meta');if(subject!==key||loadRequest!==loadRevision)return;available=true;
        if(meta.markers && editRevision===initialEdits && JSON.stringify(meta.markers)!==JSON.stringify(hooks.getBaseline())) {
          hooks.replaceMarkers(meta.markers);
          detectorOptions=meta.markers.detection_qc?.parameters||{};
          threshold=meta.markers.detection_qc?.threshold??null;
          setOptions(detectorOptions,threshold);render();
        }
        status('Select a candidate to inspect MRI slices; review filters when markers are missing.');
        el('detect-run').disabled=false;el('review-save').disabled=false;
        // Show a labeled provisional construction even before anchor confirmation.
        const request=++revision;await preview(request,subject);
      }catch(error){if(subject===key&&loadRequest===loadRevision)status(error.message);}
    }};
}
