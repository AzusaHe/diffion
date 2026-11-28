let cohort300Count=1,cohort300Metric="D_pct",cohort300Material=0,cohort300Groups;

function cv300Cases(){
    if(cohort300Groups) return cohort300Groups;
    const csv=DB.fine_cv_300.case_csv.trim().split(/\r?\n/), fields=csv.shift().split(",");
    const textFields=new Set(["domain","method","start","fit_origin","base_execution","final_execution","current_indices","dimensionless_currents","valid_frames_per_current"]);
    cohort300Groups=new Map();
    for(const line of csv){
        const cells=line.split(",");
        const row=Object.fromEntries(fields.map((k,i)=>[k,textFields.has(k)?cells[i]:Number(cells[i])]));
        const key=cvKey(row);
        if(!cohort300Groups.has(key))cohort300Groups.set(key,[]);
        cohort300Groups.get(key).push(row);
    }
    return cohort300Groups;
}

function cv300Accuracy(){
    const db=DB.fine_cv_300, selection=byId("cohort").value;
    const ids=selection==="6"?DB.equal_budget.shared_ids:selection==="all"?db.ids:db.ids.slice(0,Number(selection));
    const rows=db.rows.filter(r=>r.domain===dom());
    function metric(r,col){
        if(ids.length===db.ids.length)return r["median_"+col];
        return median(cv300Cases().get(cvKey(r)).filter(v=>ids.includes(v.material_id)).map(v=>v[col]));
    }
    const table=rows.filter(r=>r.currents===1).map(one=>{
        const three=rows.find(r=>r.currents===3&&r.method===one.method&&r.start===one.start);
        return [cvMethod[one.method],cvStart[one.start],ids.length,...[one,three].flatMap(r=>[num(metric(r,"D_pct_500"),3),num(metric(r,"t_rmse_500"),5),ms(r.pipeline_mean_s)])];
    });
    return '<h3>500-update c/v PINNs · 300 held-out materials</h3><p>Wider c and v networks fitted to the 290 × 151 observations. Errors are medians on the selected cohort. Times are means over 300 materials and include the coefficient start. <a href="#cv-300">Check all stages, field errors, costs and material fits</a>.</p>'+
        cvTable(["Method","Start","Materials","One · D error (%)","One · t₊ RMSE","One · time/material","Three · D error (%)","Three · t₊ RMSE","Three · time/material"],table);
}

function cv300Download(individual){
    const db=DB.fine_cv_300,url=URL.createObjectURL(new Blob([individual?db.case_csv:db.summary_csv],{type:"text/csv;charset=utf-8"}));
    const a=document.createElement("a");a.href=url;a.download=individual?"fine_cv_300_materials.csv":"fine_cv_300_accuracy_cost.csv";a.click();setTimeout(()=>URL.revokeObjectURL(url),500);
}

function cv300Update(){
    const db=DB.fine_cv_300,selected=r=>r.domain===dom()&&r.currents===cohort300Count, rows=db.rows.filter(selected);
    const [,label,digits]=cvProps.find(p=>p[0]===cohort300Metric), key=cohort300Metric;
    const triplets=(r,prefix)=>cvProps.map(([p,,d])=>[0,375,500].map(t=>num(r[prefix+p+"_"+t],d)).join(" → "));
    const sign=n=>(n>0?"+":"")+num(n,digits);
    byId("cv300Plot").innerHTML=pairedCvPlot(rows,r=>[r["median_"+key+"_375"],r["median_"+key+"_500"]],
        ["375 updates","500 updates"],label+" on 300 held-out materials",digits);
    byId("cv300Metrics").innerHTML=cvTable(["Method","Start",...cvProps.map(p=>p[1])],rows.map(r=>[cvMethod[r.method],cvStart[r.start],...triplets(r,"median_")]));
    byId("cv300ChangeTitle").textContent="Material-level changes in "+label;
    byId("cv300Change").innerHTML=cvTable(["Method","Start","Final median","Final IQR","Median Δ vs. start","Lower vs. start","Median Δ vs. 375","Lower vs. 375"],
        rows.map(r=>[cvMethod[r.method],cvStart[r.start],num(r["median_"+key+"_500"],digits),
            num(r["p25_"+key+"_500"],digits)+" – "+num(r["p75_"+key+"_500"],digits),
            sign(r["median_paired_delta_"+key+"_500_vs_0"]),r["n_lower_"+key+"_500_vs_0"]+"/300",
            sign(r["median_paired_delta_"+key+"_500_vs_375"]),r["n_lower_"+key+"_500_vs_375"]+"/300"]));
    byId("cv300Costs").innerHTML=cvTable(["Method","Start","Archived TTT updates","Start time","Updates 1–375","Updates 376–500","PINN time","Total incl. start"],
        rows.map(r=>[cvMethod[r.method],cvStart[r.start],r.archived_TTT_updates,ms(r.archived_start_mean_s),ms(r.stage1_mean_s),ms(r.stage2_mean_s),ms(r.PINN_mean_s),ms(r.pipeline_mean_s)]));
    byId("cv300Graph").innerHTML=cvTable(["Method","Start","Base · separate graph","Base · reuse enabled","Final · separate graph","Final · reuse enabled"],
        rows.map(r=>[cvMethod[r.method],cvStart[r.start],r.n_base_separate_graph,r.n_base_reused_graph,r.n_final_separate_graph,r.n_final_reused_graph]));
    const totals=(list,k)=>list.reduce((s,r)=>s+r[k],0);
    byId("cv300CostTotals").innerHTML=cvTable(["Scope","Previously recorded · 8","New training · 292","All 300"],[["Selected domain and current count",rows],["All 24 conditions",db.rows]].map(([name,list])=>
        [name,...["PINN_s_reused_counting_shared_once","PINN_s_new_counting_shared_once","PINN_s_all_counting_shared_once"].map(k=>ms(totals(list,k)))]));
    const cases=cv300Cases(), one=cases.get(cvKey(rows[0]))[cohort300Material];
    byId("cv300Case").innerHTML='<p>Material '+String(one.material_id).padStart(6,"0")+' · '+(one.fit_origin==="prior_eight"?"Reused eight-material fit":"New fit in the expanded cohort")+
        ' · Dimensionless current'+(cohort300Count===1?'':'s')+' ι = '+one.dimensionless_currents.split(" ").map(v=>num(+v,4)).join(", ")+
        ' · Valid frames per current '+one.valid_frames_per_current.split(" ").join(", ")+' · Measured c ['+num(one.c_min,5)+', '+num(one.c_max,5)+'] · Scored c ['+num(one.score_min,5)+', '+num(one.score_max,5)+'].</p>'+
        cvTable(["Method","Start",...cvProps.map(p=>p[1]),"PINN time","Base / final graph"],rows.map(r=>{const v=cases.get(cvKey(r))[cohort300Material];
            const how=m=>m==="separate_graph"?"separate":"reuse enabled";return [cvMethod[r.method],cvStart[r.start],...triplets(v,""),ms(v.PINN_s),how(v.base_execution)+" / "+how(v.final_execution)];}));
    if(db.reference.length){
        const methods={nm:"Nelder–Mead",cmaes:"CMA-ES",bo:"Bayesian optimization"};
        const numeric=db.reference.filter(selected).map(r=>[methods[r.method],cvStart[r.start],num(r.median_D_pct,3),num(r.median_t_rmse,5),ms(r.pipeline_mean_s)]);
        byId("cv300Numeric").innerHTML=cvTable(["Refinement","Start","D error (%)","t₊ RMSE","Time/material"],
            [...rows.map(r=>[cvMethod[r.method],cvStart[r.start],num(r.median_D_pct_500,3),num(r.median_t_rmse_500,5),ms(r.pipeline_mean_s)]),...numeric]);
    }
}

function cv300Bind(){
    for(const [id,val,apply] of [["cv300Count",cohort300Count,v=>cohort300Count=+v],["cv300Metric",cohort300Metric,v=>cohort300Metric=v],["cv300Material",cohort300Material,v=>cohort300Material=+v]]){
        byId(id).value=String(val);byId(id).onchange=e=>{apply(e.target.value);cv300Update();};
    }
    cv300Update();
}

function showCv300(){
    const db=DB.fine_cv_300,options=items=>items.map(([v,l])=>'<option value="'+v+'">'+l+'</option>').join("");
    return title("500-update c/v PINNs on 300 held-out materials")+
        '<p>The first 300 materials in the held-out solver cohort each contribute a one-current fit and a three-current fit that includes the same first current. Clean and Gaussian-noisy observations use 290 spatial bins and all valid native frames, up to 151 per current. Each material’s D and t₊ errors use the concentration range covered by that fit, intersected with [0, 2].</p>'+
        '<p>The study contains 7,200 fits across 24 conditions. Each condition reuses the measured fits for the first eight materials and adds 292 new material fits. D error is relative L₂ in percent. The t₊, concentration and reconstructed-velocity errors are absolute RMSEs. <a href="#cv-budget">View the eight-material 400 vs. 500 comparison</a>.</p>'+
        (db.native_audit?'<details><summary>Independent check against the native observations</summary><p>Independent CPU evaluation checked every fit at initialization, 375 updates and 500 updates against the original concentration data and the velocity reconstructed from them. Across the 7,200 fits, the largest absolute differences between the measured and independently recomputed RMSEs were '+db.native_audit.max_c_rmse_abs.toExponential(3)+' for dimensionless concentration and '+db.native_audit.max_v_rmse_abs.toExponential(3)+' for dimensionless velocity. Currents contributed '+db.native_audit.min_observed_frames_per_current+'–'+db.native_audit.max_observed_frames_per_current+' valid time frames. The validation record identifies all 24 scored archives by SHA-256.</p></details>':'')+
        '<details><summary>Architecture, starts and update counts</summary><p>Each c/v network has layers 3 → 44 → 47 → 46 → 1 with 4,546 parameters per network (176, 2,115, 2,208 and 47 in successive layers). The base networks use tanh activations and the additive corrections use sine. All fits use float32 Adam, one material per batch, 64 physics residual points per current and two concentration quadrature points per bin. Observed slope weight is zero.</p>'+
        '<p>Center sets the eight normalized transport coefficients to 0.5. FNO + TTT reuses their domain- and current-specific estimate from the 20 × 13 benchmark and carries 150 historical TTT updates outside the 500 PINN updates. All models fit concentration and velocity reconstructed from that concentration series.</p>'+
        cvTable(["Method","Field prefit","Joint","Additive correction","Adam updates per fit"],[["Standard joint",0,500,0,500],["Data-first + correction, w = 1 or 100",236,139,125,500]])+
        cvTable(["Stage","Field learning rate","Transport-logit learning rate","Observation weight","Each physics weight"],
            [["Standard joint","0.001","0.0001",1,100],["Data-first field prefit","0.001","Frozen","Field data only","—"],
             ["Data-first joint","0.0001","0.001",0.01,1],["Additive correction","0.001","0.001","1 or 100",1]])+
        '<p>The correction stage freezes the base c/v networks and fits the additive fields and eight transport coefficients.</p></details>'+
        '<div class="controls"><label>Currents per material<select id="cv300Count">'+options([[1,"One current"],[3,"Three currents"]])+'</select></label>'+
        '<label>Plotted error<select id="cv300Metric">'+options(cvProps.map(p=>[p[0],p[1]]))+'</select></label></div>'+
        '<div class="legend"><span class="one">375 updates</span><span class="three">500 updates</span></div><div class="scroll" id="cv300Plot"></div>'+
        '<h3>Accuracy at initialization → 375 → 500 updates</h3><p>Each number is the median of 300 material errors. Initial field errors describe the PINN’s initial c/v predictions.</p><div id="cv300Metrics"></div>'+
        '<h3 id="cv300ChangeTitle"></h3><p>The interval contains the middle 50% of final errors. Each Δ is the median of paired material-level changes. Negative values indicate improvement. Changes in D are percentage points.</p><div id="cv300Change"></div>'+
        '<h3>Measured cost per material</h3><p>PINN times are means of the recorded single-material fits, including graph setup when incurred. Total time adds the archived mean initialization time for the same domain and current count.</p><div id="cv300Costs"></div>'+
        '<h3>Recorded PINN stage time</h3><p>Each shared 375-update correction base and its setup is counted once. The totals exclude archived initialization and orchestration. Concurrent device load affects timing, including controlled execution comparisons for materials 001275 and 001355.</p><div id="cv300CostTotals"></div>'+
        '<details><summary>GPU graph execution in recorded times</summary><p>The original implementation builds a separate graph per fit. The later implementation enables reuse of the same computation graph for compatible single-material fits. Counts below identify the enabled execution mode, including construction on a cache miss. Each fit resets its observations, parameters and optimizer state. The two implementations produced bitwise-identical fields, transport logits and RMSEs at '+db.graph_timing.equal_checkpoints+' reference checkpoints, including '+db.graph_timing.equal_tensors+' compared parameter tensors.</p>'+
        '<p>An interleaved comparison on two materials included eight domain/current/start runs per execution mode, with three PINN methods in each run. The combined PINN stage time was '+num(db.graph_timing.separate_graph.seconds,3)+' s with separate graphs and '+num(db.graph_timing.reused_graph.seconds,3)+' s with reuse enabled, including setup. Primary timing rows retain the measured cost of each fit.</p><div id="cv300Graph"></div></details>'+
        (db.reference.length?'<details><summary>Compare with derivative-free refinement on the same 300 materials</summary><p>Both families use the same 290-bin observations, current indices and coefficient starts. The stored numerical coefficients are rescored on the exact PINN intervals. The solvers use CPU workers and the PINNs use the RTX 4090. Times are mean per-material pipeline costs from their respective campaigns.</p><div id="cv300Numeric"></div></details>':'')+
        '<h3>Individual material</h3><div class="controls"><label>Material ID<select id="cv300Material">'+options(db.ids.map((id,i)=>[i,String(id).padStart(6,"0")]))+'</select></label></div><div id="cv300Case"></div>'+
        '<div class="controls"><button class="export" onclick="cv300Download(false)">Download 300-material summaries (CSV)</button><button class="export" onclick="cv300Download(true)">Download all 7,200 fits (CSV)</button></div>';
}
