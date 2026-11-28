const cvMethod = {standard:"Standard joint", correction_w1:"Data-first + correction · w = 1", correction_w100:"Data-first + correction · w = 100"};
const cvStart = {center:"Center", fno_ttt:"FNO + TTT"};
const cvKey = r=>r.domain+"__"+r.currents+"__"+r.method+"_"+r.start;
const cvRows = ()=>DB.fine_cv.rows.filter(r=>r.domain===dom());
const cvCases = r=>DB.fine_cv.cases[cvKey(r)];
const cvProps = [["D_pct","D error (%)",3],["t_rmse","t₊ RMSE",5],["c_rmse","c RMSE",6],["v_rmse","v RMSE",6]];
let cvCurrent=1, cvMaterial=0, cvMetric="D_pct";

function cvTable(headers, lines){
    return wrap("<thead><tr>"+headers.map(x=>"<th>"+x+"</th>").join("")+"</tr></thead><tbody>"+
        lines.map(row=>"<tr>"+row.map(x=>"<td>"+x+"</td>").join("")+"</tr>").join("")+"</tbody>");
}

function cvAccuracy(){
    const selection=byId("cohort").value, db=DB.fine_cv;
    const ids=selection==="6"?DB.equal_budget.shared_ids:selection==="all"?db.ids:db.ids.slice(0,Number(selection));
    let h='<h3>290 × 151 c/v PINNs · 400 new updates</h3><p><a href="#fine-cv">Open the c/v PINN study</a> for the 300-update checkpoint, concentration and velocity errors, and stage timings.</p>';
    if(DB.fine_cv_width) h+='<p><a href="#cv-width">Compare the original and wider c/v networks</a> on the same eight materials and 400-update budget.</p>';
    if(DB.fine_cv_budget) h+='<p><a href="#cv-budget">400 vs. 500 PINN updates</a> compares the two training schedules on the wider 4,546-parameter field networks.</p>';
    if(selection!=="all" && +selection>db.ids.length) return h+'<p class="hint">This study covers the first eight held-out materials.</p>';
    const pairs=cvRows().filter(r=>r.currents===1).map(one=>{
        const three=cvRows().find(r=>r.currents===3&&r.method===one.method&&r.start===one.start);
        return [cvMethod[one.method],cvStart[one.start],ids.length,...[one,three].flatMap(r=>{
            const sample=cvCases(r).filter(v=>ids.includes(v.material_id));
            return [num(median(sample.map(v=>v.D_pct_400)),3),num(median(sample.map(v=>v.t_rmse_400)),5)];
        })];
    });
    return h+cvTable(["Method","Start","Materials","One · D error (%)","One · t₊ RMSE","Three · D error (%)","Three · t₊ RMSE"],pairs)+
        (DB.fine_cv_width?widthAccuracy(ids):"")+(DB.fine_cv_budget?budgetAccuracy(ids):"");
}

function cvDownload(individual){
    const url=URL.createObjectURL(new Blob([individual?DB.fine_cv.case_csv:DB.fine_cv.summary_csv],{type:"text/csv;charset=utf-8"}));
    const a=document.createElement("a");a.href=url;a.download=individual?"fine_cv_materials.csv":"fine_cv_accuracy_cost.csv";
    a.click();setTimeout(()=>URL.revokeObjectURL(url),500);
}

function cvPlot(rows){
    const [,label,precision]=cvProps.find(p=>p[0]===cvMetric);
    const lineRows=rows.map(r=>[r,[300,400].map(t=>r["median_"+cvMetric+"_"+t])]);
    const max=Math.max(...lineRows.flatMap(row=>row[1])), unit=10**Math.floor(Math.log10(max||1)), limit=Math.ceil((max||1)/unit)*unit;
    const left=325,right=740,x=v=>left+(right-left)*v/limit;
    let svg='<svg viewBox="0 0 760 211" style="width:100%;min-width:580px;max-width:810px" role="img" aria-label="'+label+' after 300 and 400 updates">';
    for(let i=0;i<=5;i++){
        const v=i*limit/5;svg+='<line x1="'+x(v)+'" x2="'+x(v)+'" y1="13" y2="174" stroke="#dbe4de"/><text x="'+x(v)+'" y="192" text-anchor="middle" font-size="12" fill="#345650">'+Number(v.toPrecision(3))+'</text>';
    }
    lineRows.forEach(([r,values],i)=>{
        const y=25+i*27;svg+='<text x="'+(left-10)+'" y="'+(y+4)+'" text-anchor="end" font-size="12" fill="#213633">'+cvMethod[r.method]+' · '+cvStart[r.start]+'</text>'+
            '<line x1="'+x(values[0])+'" x2="'+x(values[1])+'" y1="'+y+'" y2="'+y+'" stroke="#abbcb6" stroke-width="2"/>';
        for(const [j,v] of values.entries()) svg+='<circle cx="'+x(v)+'" cy="'+y+'" r="'+(j?3.5:5)+'" fill="'+(j?"#a74328":"#006b63")+'"><title>'+[300,400][j]+' updates · '+num(v,precision)+'</title></circle>';
    });
    return svg+'</svg>';
}

function cvUpdate(){
    const rows=cvRows().filter(r=>r.currents===cvCurrent);
    const stageNumbers=(get)=>cvProps.map(([k,,digits])=>[0,300,400].map(t=>num(get(k,t),digits)).join(" → "));
    byId("cvPlot").innerHTML=cvPlot(rows);
    byId("cvAccuracy").innerHTML=cvTable(["Method","Start",...cvProps.map(p=>p[1])],rows.map(r=>[
        cvMethod[r.method],cvStart[r.start],...stageNumbers((k,t)=>r["median_"+k+"_"+t])]));
    byId("cvCost").innerHTML=cvTable(["Method","Start","Archived TTT updates","Start/material","Updates 1–300","Updates 301–400","PINN/material","Total/material"],rows.map(r=>[
        cvMethod[r.method],cvStart[r.start],r.archived_TTT_updates,ms(r.archived_start_mean_s),ms(r.stage1_mean_s),ms(r.stage2_mean_s),ms(r.PINN_mean_s),ms(r.pipeline_mean_s)]));
    const uniqueFitTime=rows.reduce((s,r)=>s+r.n_test*(r.stage1_mean_s+r.stage2_mean_s-(r.method==="correction_w100"?r.stage1_mean_s+r.correction_setup_mean_s:0)),0);
    byId("cvCostNote").textContent="Across these six conditions, the measured PINN stage times total "+ms(uniqueFitTime)+" for eight materials after counting each shared 300-update base and correction setup once. Archived initialization is listed separately.";
    const detail=cvCases(rows[0])[cvMaterial];
    byId("cvCase").innerHTML='<p>Material '+String(detail.material_id).padStart(6,"0")+' · Dimensionless current'+(cvCurrent===1?"":"s")+' ι = '+
        detail.dimensionless_currents.split(" ").map(x=>num(Number(x),4)).join(", ")+' · Measured c range ['+num(detail.c_min,5)+', '+num(detail.c_max,5)+
        '] · Scored c range ['+num(detail.score_min,5)+', '+num(detail.score_max,5)+'].</p>'+
        cvTable(["Method","Start",...cvProps.map(p=>p[1]),"Updates 1–300","Updates 301–400"],rows.map(r=>{
            const v=cvCases(r)[cvMaterial];return [cvMethod[r.method],cvStart[r.start],...stageNumbers((k,t)=>v[k+"_"+t]),ms(v.stage1_s),ms(v.stage2_s)];
        }));
}

function cvBind(){
    for(const [id,val,apply] of [["cvCurrent",cvCurrent,v=>cvCurrent=Number(v)],["cvMetric",cvMetric,v=>cvMetric=v],["cvMaterial",cvMaterial,v=>cvMaterial=Number(v)]]){
        byId(id).value=String(val);byId(id).onchange=e=>{apply(e.target.value);cvUpdate();};
    }
    cvUpdate();
}

function showFineCv(){
    const options=entries=>entries.map(([value,label])=>'<option value="'+value+'">'+label+'</option>').join("");
    return title("290 × 151 c/v PINNs · 400 updates")+
        (DB.fine_cv_width?'<p><a href="#cv-width">Compare the two field-network widths</a> on field errors, transport recovery and measured cost at the same checkpoints.</p>':'')+
        (DB.fine_cv_width?'<p><a href="#cv-width">Compare with 4,546-parameter field networks</a>. The results on this tab use the original 2,273-parameter c and v networks.</p>':"")+
        '<p>Eight matched held-out materials, clean and Gaussian-noisy profiles, one or three currents, and two starts. Each fit uses 290 spatial averages and every valid native frame, up to 151 per current. D and t₊ are scored on 301 concentrations in that fit’s observed interval, intersected with [0, 2].</p>'+
        '<p>Standard training updates the fields and material coefficients jointly for all 400 steps, using data weight 1 and physics weight 100. Data-first spends 189 steps fitting c and reconstructed v, then 111 joint steps with data weight 0.01 and physics weight 1. Its final 100 steps freeze the base fields and fit additive c/v corrections and material coefficients. The correction-stage data weight is 1 or 100 and its physics weight is 1. The FNO + TTT start supplies the eight material coefficients from the earlier 20 × 13 benchmark. Center initializes all eight to 0.5.</p>'+
        '<details><summary>Field networks and optimizer</summary><p>Each material has separate 3 → 32 → 32 → 32 → 1 concentration and velocity networks. Adam uses single precision on one RTX 4090, with one material per batch and 64 physics residual points per current. Concentration predictions use two quadrature points per bin. Velocity predictions are evaluated at bin centers against velocity reconstructed separately from each observed profile series. The observed spatial-derivative loss weight is zero throughout.</p></details>'+
        '<div class="controls"><label>Currents per material<select id="cvCurrent">'+options([[1,"One current"],[3,"Three currents"]])+'</select></label>'+
        '<label>Plotted error<select id="cvMetric">'+options(cvProps.map(p=>[p[0],p[1]]))+'</select></label></div>'+
        '<div class="legend"><span class="one">300 updates</span><span class="three">400 updates</span></div><div class="scroll" id="cvPlot"></div>'+
        '<h3>Accuracy at initialization → 300 → 400 updates</h3><p>Each entry is a median across eight per-material errors. Concentration and reconstructed velocity RMSEs use valid observations and dimensionless fields. Initial c and v errors refer to the PINN’s initial field predictions. The starting FNO predicts transport coefficients.</p><div id="cvAccuracy"></div>'+
        '<h3>Measured cost per material</h3><p>Each fit takes 400 new Adam updates. FNO + TTT starts additionally use 150 archived test-time updates. Total time adds the archived start cost to the two measured PINN stages. Both correction weights reuse their 300-update base and correction setup for each domain, current count, material and start. Concurrent jobs shared the RTX 4090 during part of the sweep, so measured time differences also reflect changes in GPU load.</p><div id="cvCost"></div><p class="fineprint" id="cvCostNote"></p>'+
        '<h3>Individual material</h3><div class="controls"><label>Material ID<select id="cvMaterial">'+options(DB.fine_cv.ids.map((id,i)=>[i,String(id).padStart(6,"0")]))+'</select></label></div><div id="cvCase"></div>'+
        '<div class="controls"><button class="export" onclick="cvDownload(false)">Download c/v PINN summaries (CSV)</button><button class="export" onclick="cvDownload(true)">Download c/v PINN material fits (CSV)</button></div>';
}
