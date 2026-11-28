const fineMethods = {nm:"Nelder–Mead", cmaes:"CMA-ES", bo:"Bayesian optimization"};
const fineStarts = {center:"Center", fno:"FNO", fno_ttt:"FNO + TTT", mlp:"MLP", mlp_ttt:"MLP + TTT"};
const fineKey = r=>r.domain+"__"+r.currents+"__"+r.optimizer+"_"+r.start;
const fineStartKey = r=>r.domain+"__"+r.currents+"__"+r.start;
const fineCohortKey = r=>r.domain+"__"+r.currents;
const fineLabel = r=>fineMethods[r.optimizer]+" · "+fineStarts[r.start];
const fineRows = ()=>DB.fine_df?.conditions.filter(r=>r.domain===dom()) || [];
let fineMaterialIndex=0;
let finePlotProperty="D";

function finePlot(){
    const key=finePlotProperty==="D"?"refined_D_median_pct":"refined_t_median_rmse";
    const axis=finePlotProperty==="D"?"Median D relative L₂ error (%)":"Median t₊ RMSE";
    const rows=fineRows(), pairs=[];
    for(const method of Object.keys(fineMethods)) for(const start of Object.keys(fineStarts)){
        const values=[1,3].map(n=>rows.find(r=>r.currents===n&&r.optimizer===method&&r.start===start)?.[key]);
        if(values.every(Number.isFinite)) pairs.push({name:fineMethods[method]+" · "+fineStarts[start],values});
    }
    if(!pairs.length) return "";
    const left=217, right=742, step=23, bottom=33+pairs.length*step;
    const max=Math.max(...pairs.flatMap(p=>p.values)), unit=10**Math.floor(Math.log10(max||1));
    const limit=Math.max(unit,Math.ceil(max/unit)*unit), xpos=v=>left+(right-left)*v/limit;
    let svg='<svg viewBox="0 0 760 '+(bottom+40)+'" style="max-width:810px;min-width:560px;width:100%" role="img" aria-label="'+axis+' with one and three currents">'+
        '<text x="'+left+'" y="14" font-size="13" fill="#345650">'+axis+'</text>';
    for(let tick=0;tick<=5;tick++){
        const value=tick*limit/5,x=xpos(value);
        svg+='<line x1="'+x+'" x2="'+x+'" y1="22" y2="'+bottom+'" stroke="#dbe4de"/>'+
            '<text x="'+x+'" y="'+(bottom+19)+'" text-anchor="middle" font-size="12" fill="#46635b">'+Number(value.toPrecision(3))+'</text>';
    }
    pairs.forEach((p,i)=>{
        const y=37+i*step, [a,b]=p.values;
        svg+='<text x="'+(left-10)+'" y="'+(y+4)+'" text-anchor="end" font-size="12" fill="#213633">'+esc(p.name)+'</text>'+
            '<line x1="'+xpos(a)+'" x2="'+xpos(b)+'" y1="'+y+'" y2="'+y+'" stroke="#aebbb7" stroke-width="2"/>'+
            '<circle cx="'+xpos(a)+'" cy="'+y+'" r="5" fill="#006b63"><title>One current · '+a.toPrecision(4)+'</title></circle>'+
            '<circle cx="'+xpos(b)+'" cy="'+y+'" r="3.3" fill="#a74328"><title>Three currents · '+b.toPrecision(4)+'</title></circle>';
    });
    return svg+'</svg>';
}

function fineCohortIndices(){
    const choice = byId("cohort").value, ids = DB.fine_df.ids;
    if(choice==="6") return ids.flatMap((id,i)=>DB.equal_budget.shared_ids.includes(id)?[i]:[]);
    const n = choice==="all" ? ids.length : Math.min(ids.length,Number(choice));
    return Array.from({length:n},(_,i)=>i);
}

function fineOverview(rows,indices){
    const entries=[];
    for(const method of Object.keys(fineMethods)){
        const lines=[];
        for(const start of Object.keys(fineStarts)){
            const one=rows.find(r=>r.optimizer===method && r.start===start && r.currents===1);
            const three=rows.find(r=>r.optimizer===method && r.start===start && r.currents===3);
            if(!one && !three) continue;
            const cells=[one,three].map(r=>{
                if(!r) return "<td>—</td>".repeat(4);
                const scores=DB.fine_df.after[fineKey(r)], bounds=DB.fine_df.cohorts[fineCohortKey(r)];
                const med=f=>median(indices.map(i=>f(scores[i],bounds[i])));
                return "<td>"+num(med(v=>v[0]),2)+"</td><td>"+num(med(v=>v[1]),4)+
                    "</td><td>"+num(med((_,b)=>Math.min(2,b[1])-Math.max(0,b[0])),3)+"</td><td>"+ms(r.pipeline_s_per_material)+"</td>";
            }).join("");
            lines.push("<tr><td>"+fineStarts[start]+"</td>"+cells+"</tr>");
        }
        if(lines.length) entries.push('<tr class="group"><th colspan="9">'+fineMethods[method]+"</th></tr>"+lines.join(""));
    }
    const head='<thead><tr><th rowspan="2">Starting estimate</th><th colspan="4" class="num1">One current</th><th colspan="4" class="num3">Three currents</th></tr><tr>'+
        "<th>D error (%)</th><th>t₊ RMSE</th><th>c span</th><th>Time/material</th>".repeat(2)+"</tr></thead>";
    return wrap(head+"<tbody>"+entries.join("")+"</tbody>");
}

function fineAccuracy(){
    const rows=fineRows(), indices=fineCohortIndices();
    return '<h3>290 × 151 derivative-free refinement</h3><p>Median property errors over '+indices.length+
        ' held-out material'+(indices.length===1?'':'s')+'. The solver uses 290 spatial bin averages at every valid time. Times are mean costs over the full timed cohort, including the archived starting estimate. '+
        '<a href="#fine-grid">View starting errors, optimizer costs and individual materials</a>.</p>'+fineOverview(rows,indices);
}

function downloadFineCsv(individual=false){
    const fine=DB.fine_df;
    let csv=fine.csv;
    if(individual){
        const lines=["domain,currents,optimizer,start,material_id,current_indices,dimensionless_currents,observed_c_min,observed_c_max,scored_c_min,scored_c_max,initial_D_pct,refined_D_pct,initial_t_rmse,refined_t_rmse,initial_c_rmse,refined_c_rmse,initialization_mean_s,refinement_worker_s,pipeline_s_including_mean_init,evaluations,status"];
        for(const r of fine.conditions){
            const end=fine.after[fineKey(r)], begin=fine.before[fineStartKey(r)], bounds=fine.cohorts[fineCohortKey(r)];
            fine.ids.forEach((id,i)=>{
                const [d,t,c,s,e,stop]=end[i], [d0,t0,c0]=begin[i], [lo,hi]=bounds[i], m=materialMap[id];
                lines.push([r.domain,r.currents,r.optimizer,r.start,id,m.chosen.slice(0,r.currents).join(" "),
                    m.iota.slice(0,r.currents).join(" "),lo,hi,Math.max(0,lo),Math.min(2,hi),d0,d,t0,t,c0,c,
                    r.initialization_s_per_material,s,s+r.initialization_s_per_material,e,fine.stops[stop]].join(","));
            });
        }
        csv=lines.join("\r\n")+"\r\n";
    }
    const url=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"}));
    const a=document.createElement("a");
    a.href=url;a.download=individual?"fine_df_materials.csv":"fine_df_accuracy_cost.csv";
    a.click();setTimeout(()=>URL.revokeObjectURL(url),500);
}

function fineTable(headers,body){
    return wrap("<thead><tr>"+headers.map(x=>"<th>"+x+"</th>").join("")+"</tr></thead><tbody>"+
        body.map(row=>"<tr>"+row.map(x=>"<td>"+x+"</td>").join("")+"</tr>").join("")+"</tbody>");
}

function fineDetail(){
    const fine=DB.fine_df, count=Number(byId("fineCount").value);
    const rows=fineRows().filter(r=>r.currents===count);
    if(!rows.length) return "<p>No completed conditions for this selection.</p>";
    const both=(a,b,n)=>num(a,n)+" → "+num(b,n);
    const accuracy=fineTable(["Optimizer · start","D error (%)","t₊ RMSE","Concentration RMSE","Lower D error","Lower t₊ error","Lower c RMSE"],
        rows.map(r=>[fineLabel(r),both(r.initial_D_median_pct,r.refined_D_median_pct,3),
            both(r.initial_t_median_rmse,r.refined_t_median_rmse,5),both(r.initial_c_median_rmse,r.refined_c_median_rmse,6),
            r.D_improved_materials+"/"+r.n_test,r.t_improved_materials+"/"+r.n_test,r.c_improved_materials+"/"+r.n_test]));
    const cost=fineTable(["Optimizer · start","Start/material","Fit/material","Total/material","Median calls","Target reached","Call limit","Time limit","Converged"],
        rows.map(r=>[fineLabel(r),ms(r.initialization_s_per_material),ms(r.fit_s_per_material),ms(r.pipeline_s_per_material),
            num(r.median_evaluations,1),r.reached_target,r.used_budget,r.used_walltime,r.optimizer_converged]));
    const reference=fine.validation?.cases.find(v=>v.domain===dom()&&v.currents===count);
    const noise=fine.validation?.paired_observation_noise.find(v=>v.currents===count);
    const refText=reference?'<p class="summary">Using the generating D and t₊ in the same forward solver gives median concentration RMSE <b>'+
        num(reference.median,6)+'</b>. It meets the 0.0001 concentration target on '+reference.at_or_below_target_1e_4+
        ' of '+fine.validation.n_materials+' materials.'+(noise&&dom()==="gaussian_noisy"?' The pooled RMS difference between the paired clean and noisy observations is '+num(noise.pooled_rmse,6)+'.':'')+'</p>':'';
    const i=Number(byId("fineMaterial").value), m=materialMap[fine.ids[i]], b=fine.cohorts[fineCohortKey(rows[0])][i];
    const stopNames={reached_target:"Target reached",used_budget:"Call limit",used_walltime:"Time limit",optimizer_converged:"Optimizer converged"};
    const individual=fineTable(["Optimizer · start","D error (%)","t₊ RMSE","Concentration RMSE","Fit time","Calls","Stop reason"],
        rows.map(r=>{
            const [d,t,c,s,e,stop]=fine.after[fineKey(r)][i], [d0,t0,c0]=fine.before[fineStartKey(r)][i];
            return [fineLabel(r),both(d0,d,3),both(t0,t,5),both(c0,c,6),ms(s),e,stopNames[fine.stops[stop]]];
        }));
    byId("fineIndividual").innerHTML='<p>Material <b>'+String(m.id).padStart(6,"0")+
        '</b> · Dimensionless current'+(count===1?"":"s")+' ι = '+m.iota.slice(0,count).map(x=>num(x,4)).join(", ")+
        ' · Measured c range ['+num(b[0],5)+', '+num(b[1],5)+'] · Scored c range ['+num(Math.max(0,b[0]),5)+', '+num(Math.min(2,b[1]),5)+'].</p>'+individual;
    return '<h3>Starting estimate → refined fit</h3><p>Each arrow compares medians over the same '+rows[0].n_test+
        ' materials and observed intervals. The last three columns count materials whose individual error decreased strictly.</p>'+accuracy+refText+
        '<h3>Cost and stopping reasons</h3><p>Start and fit times are means. Each material contributes one stopping reason. The concentration target is 0.0001 in dimensionless units. The limits are 500 calls for Nelder–Mead and CMA-ES, 35 for Bayesian optimization and 300 seconds per fit.</p>'+cost+
        '<p class="fineprint">Start time is the saved mean per material from the seed-0 benchmark. Fit time is elapsed time in a CPU worker. Total time adds these two costs. Shared pretraining, validation, data preparation and compilation are separate. The summary CSV includes shared training and validation times. Concurrent worker times can exceed campaign wall time when summed.</p>';
}

function showFineGrid(){
    const fine=DB.fine_df, rows=fineRows();
    if(!rows.length) return title("290 × 151 derivative-free refinement")+"<p>No completed conditions for this domain.</p>";
    const all=fine.ids.map((_,i)=>i);
    const materialOptions=fine.ids.map((id,i)=>'<option value="'+i+'">'+String(id).padStart(6,"0")+"</option>").join("");
    return title("290 × 151 derivative-free refinement")+
        '<p class="sub">Nelder–Mead, CMA-ES and Bayesian optimization fit the same '+all.length+
        ' held-out materials in each condition. The single current is the first member of the corresponding three-current set.</p>'+
        '<p>Each observed profile averages four adjacent source cells into each of 290 bins. A 580-cell finite-volume solver predicts the profile at every valid native time, up to 151 per current. The objective weights each measured bin and time point equally. D and t₊ are scored over each material’s own observed concentration range, intersected with [0, 2].</p>'+
        '<p>The center start sets all eight normalized coefficients to 0.5. Learned starts reuse the previous FNO and MLP estimates from 20 spatial bins and up to 13 frames, with the checkpoint matched to domain and current count. The TTT starts include 150 earlier test-time updates. All five starts use the same fine-grid refinement data.</p>'+
        '<h3>Final accuracy at one and three currents</h3>'+
        '<div class="controls"><label>Plotted property<select id="finePlotProperty"><option value="D">D relative L₂ (%)</option><option value="t">t₊ RMSE</option></select></label></div>'+
        '<div class="legend"><span class="one">One current</span><span class="three">Three currents</span></div>'+
        '<div class="scroll" id="finePlot"></div><details><summary>View the values, observed spans and costs</summary>'+fineOverview(rows,all)+'</details>'+
        '<div id="fineDetails"></div><h3>Individual material</h3>'+
        '<div class="controls"><label>Material ID<select id="fineMaterial">'+materialOptions+'</select></label></div><div id="fineIndividual"></div>'+
        '<div class="controls"><button class="export" onclick="downloadFineCsv()">Download all fine-grid summaries (CSV)</button>'+
        '<button class="export" onclick="downloadFineCsv(true)">Download all fine-grid material fits (CSV)</button></div>';
}
