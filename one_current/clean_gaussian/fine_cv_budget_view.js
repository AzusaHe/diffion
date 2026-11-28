let budgetCount=1,budgetPhase="final",budgetMetric="D_pct",budgetMaterial=0;
const budgetSteps={initial:[0,0],stage1:[300,375],final:[400,500]};

function budgetAccuracy(ids){
    const db=DB.fine_cv_budget,rows=db.rows.filter(r=>r.domain===dom());
    const body=rows.filter(r=>r.currents===1).map(one=>{
        const three=rows.find(r=>r.currents===3&&r.method===one.method&&r.start===one.start);
        const cells=[one,three].flatMap(r=>{
            const cases=db.cases[cvKey(r)].filter(c=>ids.includes(c.material_id));
            return [["D_pct",3],["t_rmse",5]].map(([p,d])=>num(median(cases.map(v=>v["run400_"+p+"_400"])),d)+" → "+num(median(cases.map(v=>v["run500_"+p+"_500"])),d));
        });
        return [cvMethod[one.method],cvStart[one.start],...cells];
    });
    return '<details><summary>Wider c/v PINNs at 400 versus 500 updates</summary><p>Endpoint medians shown as 400 → 500 updates for '+ids.length+
        ' material'+(ids.length===1?'':'s')+'.</p>'+cvTable(["Method","Start","One · D error (%)","One · t₊ RMSE","Three · D error (%)","Three · t₊ RMSE"],body)+'</details>';
}

function budgetDownload(individual){
    const db=DB.fine_cv_budget,url=URL.createObjectURL(new Blob([individual?db.case_csv:db.summary_csv],{type:"text/csv;charset=utf-8"}));
    const a=document.createElement("a");a.href=url;a.download=individual?"fine_cv_budget_materials.csv":"fine_cv_budget_accuracy_cost.csv";
    a.click();setTimeout(()=>URL.revokeObjectURL(url),500);
}

function budgetUpdate(){
    const db=DB.fine_cv_budget,rows=db.rows.filter(r=>r.domain===dom()&&r.currents===budgetCount);
    const [a,b]=budgetSteps[budgetPhase],[,label,digits]=cvProps.find(p=>p[0]===budgetMetric);
    const v=(r,p,prefix="")=>[r[prefix+"run400_"+p+"_"+a],r[prefix+"run500_"+p+"_"+b]];
    const arrow=(pair,d)=>pair.map(x=>num(x,d)).join(" → ");
    const signed=(val,d)=>(val>0?"+":"")+num(val,d);
    const cost=(r,k)=>[r["mean_run400_"+k],r["mean_run500_"+k]].map(ms).join(" → ");
    byId("budgetPlot").innerHTML=pairedCvPlot(rows,r=>v(r,budgetMetric,"median_"),["400-step schedule","500-step schedule"],label+" at "+a+" versus "+b+" updates",digits);
    byId("budgetStageHeading").textContent="Median errors at "+a+" versus "+b+" updates";
    byId("budgetTable").innerHTML=cvTable(["Method","Start",...cvProps.map(p=>p[1])],rows.map(r=>[cvMethod[r.method],cvStart[r.start],...cvProps.map(([p,,d])=>arrow(v(r,p,"median_"),d))]));
    byId("budgetDeltaHeading").textContent="Paired change in "+label;
    byId("budgetDelta").innerHTML=cvTable(["Method","Start","400-step schedule","500-step schedule","Difference of medians","Median paired change","Materials with lower error"],
        rows.map(r=>[cvMethod[r.method],cvStart[r.start],...v(r,budgetMetric,"median_").map(x=>num(x,digits)),
            signed(r["delta_of_medians_"+budgetMetric+"_"+budgetPhase],digits),signed(r["median_delta_"+budgetMetric+"_"+budgetPhase],digits),
            r["n_lower_with_500_"+budgetMetric+"_"+budgetPhase]+"/"+r.n_test]));
    byId("budgetCost").innerHTML=cvTable(["Method","Start","First 300 / 375 steps","Final 100 / 125 steps","PINN/material","Time ratio","Total incl. archived start"],
        rows.map(r=>[cvMethod[r.method],cvStart[r.start],cost(r,"stage1_s"),cost(r,"stage2_s"),cost(r,"PINN_s"),
            num(r.PINN_time_ratio_500_over_400,2)+"×",cost(r,"pipeline_s")]));
    const unique=key=>rows.reduce((s,r)=>s+r.n_test*(r["mean_"+key+"_PINN_s"]-(r.method==="correction_w100"?r["mean_"+key+"_stage1_s"]+r["mean_"+key+"_correction_setup_s"]:0)),0);
    byId("budgetTimeNote").textContent="For these six conditions, measured PINN stage time totals "+ms(unique("run400"))+" for 400 updates and "+
        ms(unique("run500"))+" for 500. Each shared data-first base and correction setup is counted once.";
    const first=db.cases[cvKey(rows[0])][budgetMaterial];
    byId("budgetCases").innerHTML='<p>Material '+String(first.material_id).padStart(6,"0")+' · Dimensionless current'+(budgetCount===1?'':'s')+
        ' ι = '+first.dimensionless_currents.split(" ").map(x=>num(+x,4)).join(", ")+' · Scored concentration range ['+num(first.score_min,5)+', '+num(first.score_max,5)+'].</p>'+
        cvTable(["Method","Start",...cvProps.map(p=>p[1]),"Change in "+label,"PINN fit time"],rows.map(r=>{
            const c=db.cases[cvKey(r)][budgetMaterial];
            return [cvMethod[r.method],cvStart[r.start],...cvProps.map(([p,,d])=>arrow(v(c,p),d)),signed(c["delta_"+budgetMetric+"_"+budgetPhase],digits),
                [c.run400_PINN_s,c.run500_PINN_s].map(ms).join(" → ")];
        }));
}

function budgetBind(){
    for(const [id,val,apply] of [["budgetCount",budgetCount,v=>budgetCount=+v],["budgetPhase",budgetPhase,v=>budgetPhase=v],
                                 ["budgetMetric",budgetMetric,v=>budgetMetric=v],["budgetMaterial",budgetMaterial,v=>budgetMaterial=+v]]){
        byId(id).value=String(val);byId(id).onchange=e=>{apply(e.target.value);budgetUpdate();};
    }
    budgetUpdate();
}

function showCvBudget(){
    const options=items=>items.map(([v,l])=>'<option value="'+v+'">'+l+'</option>').join("");
    return title("400 vs. 500 updates on the wider c/v PINNs")+
        (DB.fine_cv_300?'<p><a href="#cv-300">See the 500-update results on 300 held-out materials</a>. This schedule comparison uses the original eight.</p>':'')+
        '<p>Two separately trained schedules use identical 3 → 44 → 47 → 46 → 1 field networks (4,546 parameters per field), observations, material starts and loss weights. All 192 paired fits use the 290 × 151 data. The eight held-out materials repeat across the 24 combinations of method, start, current count and observation domain. Each pair shares the same measured concentration interval for scoring D and t₊.</p>'+
        '<p>Standard training uses one continuous joint Adam run. Data-first fits the fields, jointly fits fields and transport, then freezes the base networks while training additive c/v corrections and transport. Both correction weights reuse their corresponding base. FNO + TTT starts use 150 archived updates in addition to each new PINN budget.</p>'+
        cvTable(["Schedule","Standard joint steps","Data-first field prefit","Data-first joint","Correction"],[["400 updates",400,189,111,100],["500 updates",500,236,139,125]])+
        '<div class="controls"><label>Currents per material<select id="budgetCount">'+options([[1,"One current"],[3,"Three currents"]])+'</select></label>'+
        '<label>Matched checkpoints<select id="budgetPhase">'+options([["initial","Initialization · 0 vs. 0"],["stage1","Intermediate · 300 vs. 375"],["final","Endpoint · 400 vs. 500"]])+'</select></label>'+
        '<label>Plotted error<select id="budgetMetric">'+options(cvProps.map(p=>[p[0],p[1]]))+'</select></label></div>'+
        '<div class="legend"><span class="one">400-step schedule</span><span class="three">500-step schedule</span></div><div class="scroll" id="budgetPlot"></div>'+
        '<h3 id="budgetStageHeading"></h3><p>Each arrow shows the shorter → longer schedule. D error is relative L₂ in percent. Transference number, concentration and reconstructed velocity use absolute RMSE.</p><div id="budgetTable"></div>'+
        '<h3 id="budgetDeltaHeading"></h3><p>Negative changes favor the longer schedule. The median paired change takes the within-material difference first. Changes in D error are percentage points.</p><div id="budgetDelta"></div>'+
        '<h3>Measured cost per material</h3><p>Times are mean elapsed times on one RTX 4090. The final column adds the historical initialization cost. GPU load differed across runs.</p><div id="budgetCost"></div><p id="budgetTimeNote" class="fineprint"></p>'+
        '<h3>Individual material</h3><div class="controls"><label>Material ID<select id="budgetMaterial">'+options(DB.fine_cv_budget.ids.map((m,i)=>[i,String(m).padStart(6,"0")]))+'</select></label></div><div id="budgetCases"></div>'+
        '<div class="controls"><button class="export" onclick="budgetDownload(false)">Download budget summaries (CSV)</button><button class="export" onclick="budgetDownload(true)">Download paired budget fits (CSV)</button></div>';
}
