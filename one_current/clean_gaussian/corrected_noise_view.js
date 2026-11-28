const revisionWidths=DB.corrected_noisy.completed_widths;
const revisionSlugs=revisionWidths.flatMap(w=>w===20?["fno","mlp"]:["fno_grid"+w,"conv_mlp_grid"+w]);
const revisionGridText=revisionWidths.map(w=>gridSpecs.find(g=>g[0]===w).join(" × ")).join(", ");
const revisionCorrected=()=>dom()==="gaussian_noisy";
const revisionDownstream=["fine-grid","fine-cv","cv-width","cv-budget","cv-300","derivatives"];

function visibleNoiseStudies(raw){
    const display={...raw},onlyClean=rows=>rows.filter(r=>r.domain==="clean");
    const cleanKeys=bank=>Object.fromEntries(Object.entries(bank).filter(([k])=>k.startsWith("clean__")));
    display.runs=Object.fromEntries(Object.entries(raw.runs).filter(([,r])=>r.domain==="clean"||r.meta.dataset_revision===raw.corrected_noisy.revision));
    display.seeds=Object.fromEntries(Object.entries(raw.seeds).filter(([k])=>k.startsWith("clean__")||display.runs[k]));
    for(const k of ["fine_cv","fine_cv_width","fine_cv_budget","fine_cv_300","fine_cv_w001"]){
        const study=raw[k];if(!study)continue;
        display[k]={...study,rows:onlyClean(study.rows),summary_csv:revisionCleanCsv(study.summary_csv),case_csv:revisionCleanCsv(study.case_csv)};
        if(study.cases)display[k].cases=cleanKeys(study.cases);
        if(study.reference)display[k].reference=onlyClean(study.reference);
    }
    if(raw.fine_df)display.fine_df=raw.fine_df;
    if(raw.derivatives)display.derivatives={...raw.derivatives,conditions:onlyClean(raw.derivatives.conditions),physics:onlyClean(raw.derivatives.physics||[])};
    display.equal_budget={...raw.equal_budget,rows:onlyClean(raw.equal_budget.rows)};
    if(raw.residuals)display.residuals=cleanKeys(raw.residuals);
    display.architectures=raw.architectures.map(a=>{
        const grid=raw.grid_comparison.rows.find(r=>r.domain==="gaussian_noisy" && (r.space_bins===20?r.family:
                    (r.family==="fno"?"fno_grid":"conv_mlp_grid")+r.space_bins)===a.slug);
        return grid?{...a,sources:[...a.sources,"Corrected Gaussian run records: corrected_noisy_results/grid"+grid.space_bins+"/seed{0,1,2}/"],
                notes:(a.notes||"")+" The corrected Gaussian models were retrained from scratch at each current count and seed."}:a;
    });
    return display;
}

function revisionCleanLabels(){
    const content=byId("content");
    if(tab==="architecture"){
        for(const p of content.querySelectorAll("p.sub"))if(p.textContent.startsWith("Settings apply"))
            p.textContent="The supervised FNO and MLP use the listed architectures for clean and corrected observations. The downstream settings document the clean refinement studies. Corrected noisy refinements are pending.";
    }
    if(tab==="grids"){
        for(const p of content.querySelectorAll("p")){
            if(p.textContent.startsWith("Each domain, current count"))
                p.innerHTML='The clean comparison uses 60 separately fitted checkpoints across five grids, two current counts, two network families and three seeds. Fits use 100 epochs, batch size 64 and AdamW with cosine decay from 0.001. Validation selects the lowest coefficient MSE. See <a href="#architecture">network architectures</a> or switch the observation selector to view the corrected Gaussian results.';
            if(p.textContent.startsWith("Building the reusable observations for both domains"))
                p.textContent=p.textContent.replace("Building the reusable observations for both domains","Building the original clean and legacy Gaussian grid cache");
        }
    }
    if(tab==="fine-grid"&&revisionCorrected()){
        for(const p of content.querySelectorAll("p")){
            if(p.textContent.startsWith("Each observed profile"))p.textContent="Each supplied noisy measurement has 290 spatial bins. A 580-cell finite-volume solver predicts each valid measured frame, up to 151 per current. The objective weights every observed bin and time point equally. Property errors use the material’s observed concentration range clipped to [0, 2].";
            if(p.textContent.startsWith("The center start"))p.textContent="Center sets all eight normalized coefficients to 0.5. FNO uses the seed-0 model trained directly on the corrected 290 × 151 observations for the matching current count. Both starts use forward-only refinement. The clean FNO-start solver uses the historical 20 × 13 FNO.";
        }
    }
    if(tab==="derivatives"){
        for(const h of content.querySelectorAll("h4"))if(h.textContent.includes("all four observation settings"))h.textContent="Same method across clean one- and three-current observations";
    }
    if(tab==="cv-300"){
        const buttons=[...content.querySelectorAll("button.export")];
        const labels={"Download original 24 summaries (CSV)":"Download 12 original clean summaries (CSV)","Download original 7,200 fits (CSV)":"Download 3,600 original clean fits (CSV)",
                      "Download all 28 summaries (CSV)":"Download all 16 clean summaries (CSV)","Download all 8,400 fits (CSV)":"Download all 4,800 clean fits (CSV)"};
        for(const b of buttons)if(labels[b.textContent])b.textContent=labels[b.textContent];
        for(const p of content.querySelectorAll("p"))if(p.textContent.startsWith("The combined study contains"))
            p.textContent="The clean comparison contains 4,800 fits across 16 conditions. Twelve original clean conditions use eight pilot materials plus 292 additional materials. Four clean correction conditions at weight 0.01 reuse the same update-375 base and initial correction parameters as weights 1 and 100, then run 125 updates.";
    }
}

function activeNoiseRun(slug,count){
    const r=runs[dom()+"__"+count+"__"+slug];
    return !revisionCorrected()||r?.meta.dataset_revision===DB.corrected_noisy.revision?r:undefined;
}

function revisionMethodMenu(){
    const selector=byId("method"),previous=selector.value;
    const choices=revisionCorrected()?revisionSlugs:order.filter(s=>runs["clean__1__"+s]);
    const code=choices.join("|");
    if(selector.dataset.choices===code)return;
    selector.dataset.choices=code;selector.innerHTML=choices.map(s=>'<option value="'+s+'">'+esc(names[s])+'</option>').join("");
    selector.value=choices.includes(previous)?previous:choices[0];
}

function revisionDownload(csv,name){
    const url=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"})),a=document.createElement("a");
    a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),500);
}

function revisionCleanCsv(csv){
    const lines=csv.trimEnd().split(/\r?\n/),header=lines.shift(),index=header.split(",").indexOf("domain");
    if(index<0)return csv;
    return [header,...lines.filter(r=>r.split(",")[index]?.replaceAll('"',"")==="clean")].join("\r\n")+"\r\n";
}

function revisionExportButton(){
    return '<button class="export" onclick="revisionDownload(DB.corrected_noisy.supervised_csv,\'corrected_supervised_accuracy_cost.csv\')">Download all corrected supervised seeds (CSV)</button>';
}

function revisionNotice(){
    const complete=revisionWidths.length===gridSpecs.length;
    const coverage=complete?'Corrected Gaussian supervised models are available on all five grids.':
        'Corrected Gaussian supervised models are available on '+revisionWidths.length+' of '+gridSpecs.length+' grids ('+revisionGridText+'). Additional resolutions will appear after all 12 fits at each grid finish.';
    const counts=DB.corrected_noisy.derivative_free_current_counts;
    const available=counts.length?' The 290 × 151 derivative-free solvers are available with center and corrected FNO starts for '+counts.join(' and ')+' current'+(counts.length===1&&counts[0]===1?'':'s')+'.':' The corrected derivative-free solver comparison is pending.';
    const warning=available+' Gaussian-noisy TTT, CNN, differentiable solver and PINN results used earlier observations or learned starts. Their corrected results are pending.';
    const jump=!revisionCorrected()&&tab!=="noise"?' <button type="button" class="export" id="jumpCorrected">View corrected Gaussian accuracy</button>':'';
    byId("revisionNotice").innerHTML='<strong>Corrected Gaussian observations.</strong> '+coverage+' '+warning+jump;
    if(byId("jumpCorrected"))byId("jumpCorrected").onclick=()=>{
        byId("domain").value="gaussian_noisy";byId("family").value="";refreshMaterials();
        location.hash="accuracy";tab="accuracy";render();
    };
}

function revisionTable(headers,rows){
    return wrap('<thead><tr>'+headers.map(x=>'<th>'+esc(x)+'</th>').join("")+'</tr></thead><tbody>'+
        rows.map(row=>'<tr>'+row.map(v=>'<td>'+esc(v)+'</td>').join("")+'</tr>').join("")+'</tbody>');
}

function revisionAccuracy(){
    const selected=byId("family").value,heading=title("Observed-range accuracy · corrected Gaussian inputs");
    const refinements=DB.corrected_noisy.derivative_free_current_counts.length?fineAccuracy():"";
    if(selected===groupNames[1]&&refinements)return heading+refinements+'<button class="export" onclick="downloadEqualBudget()">Download all Accuracy results (CSV)</button>';
    if(selected&&selected!==groupNames[0])return heading+'<p>The selected method family is awaiting evaluation on the corrected observations.</p>';
    const rows=revisionSlugs.map(s=>{
        const pair=[1,3].map(n=>run(s,n)),samples=pair.map(observations);
        return [names[s],samples[0].length,...samples.flatMap((v,i)=>{const [d,t,span]=metrics(v);return [num(d,3),num(t,5),num(span,4),ms(pair[i].per_s)];})];
    });
    return heading+'<p>Primary results use training seed 0. D is scored by relative L₂ error (%) and t₊ by dimensionless RMSE. The table shows the median across materials, with each fit scored over its own observed concentration interval. Time per material is mean batched inference time.</p>'+
        revisionTable(["Network and grid","Materials","One · D (%)","One · t₊","One · c span","One · time","Three · D (%)","Three · t₊","Three · c span","Three · time"],rows)+
        '<p><a href="#grids">Compare resolutions and seeds</a> · <a href="#noise">Compare paired clean and corrected observations</a></p>'+(selected?'':refinements)+
        '<button class="export" onclick="downloadEqualBudget()">Download all Accuracy results (CSV)</button>';
}

function revisionNoise(){
    const selected=byId("family").value;
    let h=title("Clean versus corrected Gaussian observations")+
        '<p>Each difference is corrected Gaussian minus clean for the same material, model, grid, seed 0 and current selection. Each prediction is scored on its own observed range. Positive values indicate a larger error on the corrected input. Where temporal sampling changes, the difference includes both noise and measurement timing.</p>';
    const counts=DB.corrected_noisy.derivative_free_current_counts,idx=fineCohortIndices();
    const refine=counts.length?'<h3>290 × 151 derivative-free fits</h3><p>These pairs use the same material and optimizer. Both center arms begin at 0.5. The clean FNO-start arm uses the earlier 20 × 13 network and the corrected arm uses the new 290 × 151 network, so the FNO-start difference also includes the change in initializer.</p>'+
        revisionTable(["Optimizer · start","Currents","Paired materials","Median Δ D (pp)","Median Δ t₊ RMSE","Median Δ concentration RMSE"],
          DB.fine_df.conditions.filter(r=>r.domain==="gaussian_noisy").map(r=>{
            const end=DB.fine_df.after[fineKey(r)],cl=DB.fine_df.after[fineKey({...r,domain:"clean"})];
            return [fineLabel(r),r.currents,idx.length,...[0,1,2].map((m,i)=>num(median(idx.map(j=>end[j][m]-cl[j][m])),[3,5,6][i]))];
          })):"";
    if(selected===groupNames[1]&&refine)return h+refine;
    if(selected&&selected!==groupNames[0])return h+'<p>Paired corrected results are pending for this method family.</p>';
    const rows=revisionSlugs.map(s=>{
        const pairs=[1,3].map(n=>[runs["clean__"+n+"__"+s],runs["gaussian_noisy__"+n+"__"+s]].map(observations));
        return [names[s],pairs[0][0].length,...pairs.flatMap(([cl,no])=>[1,2].map(i=>{
            const v=median(no.map((r,j)=>{if(r[0]!==cl[j][0])throw Error("Unpaired material IDs");return r[i]-cl[j][i];}));
            return (v>0?"+":"")+num(v,i===1?3:5);
        }))];
    });
    return h+revisionTable(["Network and grid","Materials","One · Δ D (pp)","One · Δ t₊ RMSE","Three · Δ D (pp)","Three · Δ t₊ RMSE"],rows)+revisionExportButton()+(selected?'':refine);
}

function revisionSolverPoints(points,idx){
    if(byId("family").value&&byId("family").value!==groupNames[1])return;
    const indices=fineCohortIndices();
    for(const r of fineRows()){
        const vals=DB.fine_df.after[fineKey(r)],err=median(indices.map(i=>vals[i][idx-1]));
        points.push({r:{name:fineLabel(r)+" · 290 × 151",family:groupNames[1],count:r.currents,per_s:r.pipeline_s_per_material,meta:{}},err,n:indices.length});
    }
}

function revisionSingleCountPlot(){
    const n=DB.corrected_noisy.derivative_free_current_counts[0],rows=fineRows();
    const metric=finePlotProperty==="D"?"refined_D_median_pct":"refined_t_median_rmse",label=finePlotProperty==="D"?"Median D relative L₂ (%)":"Median t₊ RMSE";
    const largest=Math.max(...rows.map(r=>r[metric]),1e-12),left=212,width=420;
    return '<svg viewBox="0 0 730 '+(rows.length*30+40)+'" style="width:100%;min-width:490px;max-width:800px" role="img" aria-label="'+esc(label)+'"><text x="'+left+'" y="16">'+esc(label+' · '+n+' current'+(n===1?'':'s'))+'</text>'+
        rows.map((r,i)=>{const y=i*30+40,w=width*r[metric]/largest;
            return '<text x="'+(left-10)+'" y="'+y+'" text-anchor="end">'+esc(fineLabel(r))+'</text><rect x="'+left+'" y="'+(y-10)+'" width="'+w+'" height="9" fill="'+(n===1?'#006b63':'#a74328')+'"/><text x="'+(left+w+9)+'" y="'+y+'">'+num(r[metric],finePlotProperty==="D"?2:4)+'</text>';
        }).join('')+'</svg>';
}

function revisionSeeds(){
    const table=revisionSlugs.flatMap(s=>[1,3].map(n=>{const rr=DB.seeds["gaussian_noisy__"+n+"__"+s];return [names[s],n,...["d","t"].flatMap(k=>rr.map(r=>num(r[k],k==="d"?3:5)))];}));
    return title("Training seeds · corrected Gaussian inputs")+'<p>Each cell is the median across the same 389 held-out materials. Each current count, network, resolution and seed was trained independently on the corrected observations.</p>'+
        revisionTable(["Network and grid","Currents","D (%) · seed 0","Seed 1","Seed 2","t₊ RMSE · seed 0","Seed 1","Seed 2"],table)+revisionExportButton();
}

function revisionCost(){
    const h=title("Training and inference cost · corrected Gaussian inputs");
    const refinements=DB.corrected_noisy.derivative_free_current_counts.length?'<h3>290 × 151 derivative-free refinement</h3>'+
        revisionTable(["Optimizer · start","Currents","Start per material","Worker fit per material","Total per material","One-time FNO training"],
            fineRows().map(r=>[fineLabel(r),r.currents,ms(r.initialization_s_per_material),ms(r.fit_s_per_material),ms(r.pipeline_s_per_material),ms(r.shared_prior_training_s)]))+
            '<button class="export" onclick="downloadFineCsv()">Download derivative-free accuracy and cost (CSV)</button>':"";
    if(byId("family").value===groupNames[1]&&refinements)return h+refinements;
    if(byId("family").value&&byId("family").value!==groupNames[0])return h+'<p>The selected method family is awaiting corrected evaluation.</p>';
    return h+'<p>Seed 0. Fit time includes validation and saving the epoch states. Scoring the frozen epoch states on train and test data is timed separately. Inference uses prepared inputs in batches of 128, averaged over ten full passes and divided by 389 materials.</p>'+
        revisionTable(["Network and grid","Currents","Device","Parameters","Fit time","Epoch diagnostics","Infer 389 materials","Inference per material","Selected epoch"],
            revisionSlugs.flatMap(s=>[1,3].map(n=>{const r=run(s,n);return [names[s],n,r.meta.device,r.meta.parameters.toLocaleString("en-US"),ms(r.training),ms(r.meta.epoch_evaluation_s),ms(r.meta.test_s),ms(r.per_s),r.meta.best_epoch];})))+revisionExportButton()+(byId("family").value?'':refinements);
}

function revisionGrids(){
    const db=DB.grid_comparison,corrected=db.rows.filter(r=>r.domain==="gaussian_noisy"),rows=corrected.filter(r=>r.seed===gridSeed);
    let h=title("Supervised grid comparison · corrected Gaussian inputs")+
        '<p>Direct grid observations supply every reported corrected run. Each completed grid contributes 12 models across two networks, two current counts and three seeds. The one-current measurement is the first of its three-current set.</p>'+
        '<div class="controls"><label>Training seed<select id="revisionSeed">'+[0,1,2].map(s=>'<option value="'+s+'"'+(s===gridSeed?" selected":"")+'>'+s+(s===0?" (primary)":"")+'</option>').join("")+'</select></label></div>'+
        '<h3>Each grid’s observed concentration interval</h3>'+gridErrors(rows,"own");
    if(DB.corrected_noisy.all_shared_ranges){
        h+='<h3>Same concentration interval for all five corrected grids</h3><p>For each material and one- or three-current selection, all five predictions are rescored on the intersection of their corrected measured ranges, clipped to [0, 2].</p>'+
            gridErrors(rows,"common")+'<div class="graphs">'+gridChart("D_median_pct","Median D error")+gridChart("t_median_rmse","Median t₊ RMSE")+'</div>';
    }else{
        h+='<p class="fineprint">The five-grid shared-interval scores and plots will appear when all five corrected grids are complete. Current scores use only the observations at their labelled resolution.</p>';
    }
    h+='<h3>Cost and held-out range</h3>'+revisionTable(["Grid","Network","Currents","Median c span","Epoch","Training","Inference per material"],
        rows.map(r=>[r.space_bins+" × "+r.max_frames,r.model,r.currents,num(r.own_median_c_span,5),r.best_epoch,ms(r.train_s),ms(r.mean_inference_s_per_material)]));
    return h+revisionExportButton()+' <button class="export" onclick="downloadGridCSV()">Download all current grids and seeds (CSV)</button>';
}

function revisionProtocol(){
    const rows=Object.values(DB.corrected_noisy.preparation).map(p=>[p.source_grid.join(" × "),p.source_folder,p.eligible,p.split.train,p.split.val,p.split.test,p.selected_frames_per_current.join("–")]);
    const inventory=DB.corrected_noisy.source_inventory;
    const support=inventory?'<h3>Held-out observation coverage</h3><p>Measured noise is the dimensionless RMS difference from clean observations averaged over identical spatial bins and sampled at the same times. “Only initial frame” counts materials for which every selected current contributes just time zero. “Same timestamps” compares the complete set of observation times against the clean benchmark sampling for that current selection.</p>'+
        revisionTable(["Grid","Currents","Measured noise RMS","Only initial frame / 389","Same timestamps / 389"],revisionWidths.flatMap(w=>[1,3].map(k=>{
            const item=inventory.grids[w].test[k];return [w+" × "+gridSpecs.find(g=>g[0]===w)[1],k,num(item.corrected_noise_rms_vs_native_clean,7),item.all_currents_single_frame_materials,item.materials_matching_legacy_observed_times];
        }))):"";
    return title("Clean and corrected Gaussian data and scoring")+
        '<p>The clean results use the existing <span class="code">Project/latest_synthetic_ml/data/clean</span> simulations with 1,160 native spatial cells. Each clean supervised grid uses its prior spatial averages and valid-time sampling. Every reported corrected Gaussian model uses its directly uploaded array under <span class="code">Project/latest_synthetic_ml/data/Corrected_noisy_data</span>.</p>'+
        '<p>The uploaded folders contain synthetic Gaussian perturbations drawn on their supplied grids. Their recorded standard deviation scales with concentration-dependent photon noise, with a reference standard deviation of 2.2 mM at 2.97 M. The selected concentration, current values and valid timestamps are read directly. Velocity is reconstructed from that grid’s concentration series. A single valid frame supplies no temporal difference. The clean benchmark retains its original sampling protocol.</p>'+
        revisionTable(["Grid","Source folder under data/","Eligible","Train","Validation","Test","Valid frames/current"],rows)+support+
        '<p>Each test material shares the clean split ID, transport coefficients and current indices. One-current models use the first selected current. Three-current models use all three. Each property is evaluated at 301 equally spaced concentrations on the observed range intersected with [0, 2].</p>'+
        '<h3>Supervised training</h3><p>Each network, current count, grid and seed has its own fit. All corrected runs use 100 epochs, AdamW, a starting learning rate of 0.001, weight decay of 0.01, a per-step cosine schedule and batches of 64. The eight normalized property coefficients supply the MSE target. The lowest validation MSE selects the checkpoint. All fits use float32 with TF32 disabled and one CPU thread. The seed-level CSV records exact checkpoint and epoch-trace SHA-256s along with the input hash.</p>'+revisionExportButton();
}

function renderNoiseRevision(){
    revisionNotice();
    const corrected=revisionCorrected(),noise=tab==="noise";
    if(!corrected&&!noise&&tab!=="protocol")return false;
    const selectedFamily=byId("family").value;
    let h;
    if(noise)h=revisionNoise();
    else if(tab==="fine-grid"&&DB.corrected_noisy.derivative_free_current_counts.length){
        // The original view binds its plot and per-material selectors after inserting the markup.
        return false;
    }
    else if(revisionDownstream.includes(tab)){
        byId("gradCountLabel").hidden=true;byId("gradStartLabel").hidden=true;byId("fineCountLabel").hidden=true;
        h=title("Corrected noisy refinement is pending")+'<p>This method used the earlier Gaussian observations or an initialization trained on them. It has not been rerun on the corrected source. The clean results remain available from the observation selector.</p><p><a href="#accuracy">View corrected supervised accuracy</a></p>';
    }else if(tab==="accuracy")h=revisionAccuracy();
    else if(tab==="grids")h=revisionGrids();
    else if(tab==="cost")h=revisionCost();
    else if(tab==="seeds")h=revisionSeeds();
    else if(tab==="trade")h=selectedFamily===groupNames[2]?title("Corrected accuracy versus cost")+'<p>The selected family awaits corrected evaluation.</p>':showTrade();
    else if(tab==="cases")h=showCases();
    else if(tab==="protocol")h=revisionProtocol();
    else return false;
    byId("content").innerHTML='<section class="panel">'+h+'</section>';
    if(byId("revisionSeed"))byId("revisionSeed").onchange=e=>{gridSeed=Number(e.target.value);render();};
    return true;
}
