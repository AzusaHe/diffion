let widthCount=1, widthStage=400, widthProp="D_pct", widthMaterial=0;

function widthAccuracy(ids){
    const db=DB.fine_cv_width,rows=db.rows.filter(r=>r.domain===dom()),one=rows.filter(r=>r.currents===1);
    const scores=(r,k)=>db.cases[cvKey(r)].filter(v=>ids.includes(v.material_id)).map(v=>v[k]);
    const body=one.map(r=>[cvMethod[r.method],cvStart[r.start],...[r,rows.find(t=>t.currents===3&&t.method===r.method&&t.start===r.start)].flatMap(t=>
        [["D_pct",3],["t_rmse",5]].map(([k,d])=>num(median(scores(t,"base_"+k+"_400")),d)+" → "+num(median(scores(t,"wide_"+k+"_400")),d)))]);
    return '<details><summary>Paired field-network widths at 400 updates</summary><p>Original → wider on the same '+ids.length+' material'+(ids.length===1?'':'s')+
        '. <a href="#cv-width">Compare all errors, costs and individual fits</a>.</p>'+cvTable(["Method","Start","One · D error (%)","One · t₊ RMSE","Three · D error (%)","Three · t₊ RMSE"],body)+'</details>';
}

function widthDownload(individual){
    const data=DB.fine_cv_width, url=URL.createObjectURL(new Blob([individual?data.case_csv:data.summary_csv],{type:"text/csv;charset=utf-8"}));
    const a=document.createElement("a");a.href=url;a.download=individual?"fine_cv_width_materials.csv":"fine_cv_width_accuracy_cost.csv";
    a.click();setTimeout(()=>URL.revokeObjectURL(url),500);
}

function pairedCvPlot(rows,values,names,description,digits){
    const max=Math.max(...rows.flatMap(values)),unit=10**Math.floor(Math.log10(max||1)),limit=Math.ceil((max||1)/unit)*unit;
    const left=322,right=730,x=v=>left+(right-left)*v/limit;
    let svg='<svg viewBox="0 0 760 206" style="width:100%;min-width:565px;max-width:810px" role="img" aria-label="'+description+'">';
    for(let i=0;i<=5;i++){
        const tick=i*limit/5;svg+='<line x1="'+x(tick)+'" x2="'+x(tick)+'" y1="12" y2="173" stroke="#dbe4de"/><text x="'+x(tick)+'" y="192" text-anchor="middle" font-size="12" fill="#345650">'+Number(tick.toPrecision(3))+'</text>';
    }
    rows.forEach((r,i)=>{
        const y=26+27*i, [a,b]=values(r);
        svg+='<text x="'+(left-12)+'" y="'+(y+4)+'" text-anchor="end" font-size="12" fill="#213633">'+cvMethod[r.method]+' · '+cvStart[r.start]+'</text>'+
            '<line x1="'+x(a)+'" x2="'+x(b)+'" y1="'+y+'" y2="'+y+'" stroke="#acbab4" stroke-width="2"/>'+
            '<circle cx="'+x(a)+'" cy="'+y+'" r="5" fill="#006b63"><title>'+names[0]+' · '+num(a,digits)+'</title></circle>'+
            '<circle cx="'+x(b)+'" cy="'+y+'" r="3" fill="#a74328"><title>'+names[1]+' · '+num(b,digits)+'</title></circle>';
    });
    return svg+'</svg>';
}

function widthUpdate(){
    const db=DB.fine_cv_width, rows=db.rows.filter(r=>r.domain===dom()&&r.currents===widthCount);
    const [,label,digits]=cvProps.find(row=>row[0]===widthProp), k=widthProp+"_"+widthStage;
    const paired=(a,b,d)=>num(a,d)+" → "+num(b,d);
    const delta=(v,d)=>(v>0?"+":"")+num(v,d);
    const costs=(a,b)=>ms(a)+" → "+ms(b);
    const propCells=r=>cvProps.map(([p,,d])=>paired(r["median_base_"+p+"_"+widthStage],r["median_wide_"+p+"_"+widthStage],d));
    byId("widthPlot").innerHTML=pairedCvPlot(rows,r=>[r["median_base_"+k],r["median_wide_"+k]],["Original","Wider"],label+" by field-network width at "+widthStage+" updates",digits);
    byId("widthGroupHeading").textContent="Median errors after "+widthStage+" updates";
    byId("widthMedians").innerHTML=cvTable(["Method","Start",...cvProps.map(p=>p[1])],rows.map(r=>[cvMethod[r.method],cvStart[r.start],...propCells(r)]));
    byId("widthDeltaHeading").textContent="Paired change in "+label;
    byId("widthDeltas").innerHTML=cvTable(["Method","Start","Original median","Wider median","Difference of medians","Median paired change","Materials with lower error"],
        rows.map(r=>[cvMethod[r.method],cvStart[r.start],num(r["median_base_"+k],digits),num(r["median_wide_"+k],digits),
            delta(r["delta_of_medians_"+k],digits),delta(r["median_delta_"+k],digits),r["wider_lower_n_"+k]+"/"+r.n_test]));
    byId("widthCost").innerHTML=cvTable(["Method","Start","Updates 1–300","Updates 301–400","PINN/material","PINN time ratio","Total incl. start"],
        rows.map(r=>[cvMethod[r.method],cvStart[r.start],costs(r.mean_base_stage1_s,r.mean_wide_stage1_s),costs(r.mean_base_stage2_s,r.mean_wide_stage2_s),
            costs(r.mean_base_PINN_s,r.mean_wide_PINN_s),num(r.wide_over_base_PINN_time,2)+"×",costs(r.mean_base_pipeline_s,r.mean_wide_pipeline_s)]));
    const time=kind=>rows.reduce((s,r)=>s+r.n_test*(r["mean_"+kind+"_PINN_s"]-(r.method==="correction_w100"?r["mean_"+kind+"_stage1_s"]+r["mean_"+kind+"_correction_setup_s"]:0)),0);
    byId("widthTiming").textContent="Summing the eight materials in this selection and counting each shared base and correction setup once gives "+ms(time("base"))+" of PINN time at the original width and "+ms(time("wide"))+" at the wider setting.";
    const item=r=>db.cases[cvKey(r)][widthMaterial], example=item(rows[0]);
    byId("widthCase").innerHTML='<p>Material '+String(example.material_id).padStart(6,"0")+' · Dimensionless current'+(widthCount===1?"":"s")+' ι = '+
        example.dimensionless_currents.split(" ").map(v=>num(+v,4)).join(", ")+' · Scored concentration range ['+num(example.score_min,5)+", "+num(example.score_max,5)+'].</p>'+
        cvTable(["Method","Start",...cvProps.map(p=>p[1]),"Change in "+label,"PINN time"],rows.map(r=>{
            const m=item(r);return [cvMethod[r.method],cvStart[r.start],...cvProps.map(([p,,d])=>paired(m["base_"+p+"_"+widthStage],m["wide_"+p+"_"+widthStage],d)),
                delta(m["delta_"+k],digits),costs(m.base_PINN_s,m.wide_PINN_s)];
        }));
}

function widthBind(){
    for(const [id,val,apply] of [["widthCount",widthCount,v=>widthCount=+v],["widthStage",widthStage,v=>widthStage=+v],
                                 ["widthProp",widthProp,v=>widthProp=v],["widthMaterial",widthMaterial,v=>widthMaterial=+v]]){
        byId(id).value=String(val);byId(id).onchange=e=>{apply(e.target.value);widthUpdate();};
    }
    widthUpdate();
}

function showCvWidth(){
    const options=items=>items.map(([v,l])=>'<option value="'+v+'">'+l+'</option>').join("");
    return title("Does doubling the c/v network size improve recovery?")+
        (DB.fine_cv_budget?'<p><a href="#cv-budget">Compare the 400- and 500-update schedules</a> for the wider field networks.</p>':'')+
        '<p>Every original fit is paired with a wider fit on the same material, observations, coefficient start and 400-update schedule. The 24 conditions cover clean and Gaussian-noisy data, one or three currents, three methods and two starts. Each condition uses the same eight held-out materials. A pair shares its observed concentration range for scoring D and t₊. Initial field predictions can differ because the randomly initialized networks have different layer sizes.</p>'+
        '<div class="arch-protocols"><section class="arch-protocol"><h3>Original field network</h3><p class="arch-count">2,273 trainable parameters per field</p><p>3 → 32 → 32 → 32 → 1</p></section>'+
        '<section class="arch-protocol three"><h3>Wider field network</h3><p class="arch-count">4,546 trainable parameters per field</p><p>3 → 44 → 47 → 46 → 1</p></section></div>'+
        '<p>The widths apply to the tanh c/v base networks and the sine correction networks. During the final correction stage, the two base networks are frozen and the two correction networks and eight transport coefficients are trainable. The FNO + TTT estimates come from the same archived 20 × 13 model in both runs. <a href="#fine-cv">Original c/v PINN schedules and results</a>.</p>'+
        '<details><summary>Parameter counts by layer and method</summary>'+cvTable(["Parameters","Original","Wider"],[
            ["Per field · input → hidden 1","128","176"],["Per field · hidden 1 → hidden 2","1,056","2,115"],["Per field · hidden 2 → hidden 3","1,056","2,208"],["Per field · hidden 3 → output","33","47"],
            ["Standard · both field networks","4,546","9,092"],["Correction · both base and both correction networks","9,092","18,184"],["Final trainable fields + 8 material logits · either method","4,554","9,100"]])+'</details>'+
        '<div class="controls"><label>Currents per material<select id="widthCount">'+options([[1,"One current"],[3,"Three currents"]])+'</select></label>'+
        '<label>Checkpoint<select id="widthStage">'+options([[0,"Initialization"],[300,"300 updates"],[400,"400 updates"]])+'</select></label>'+
        '<label>Plotted error<select id="widthProp">'+options(cvProps.map(p=>[p[0],p[1]]))+'</select></label></div>'+
        '<div class="legend"><span class="one">Original (2,273 parameters/field)</span><span class="three">Wider (4,546 parameters/field)</span></div><div class="scroll" id="widthPlot"></div>'+
        '<h3 id="widthGroupHeading"></h3><p>Each arrow shows original → wider. D error is relative L₂ in percent. The t₊, concentration and reconstructed-velocity errors are absolute RMSEs.</p><div id="widthMedians"></div>'+
        '<h3 id="widthDeltaHeading"></h3><p>Negative changes indicate lower error at the wider setting. The paired change subtracts original from wider for each material before taking the median. The difference of medians subtracts the two group medians. Changes in D error are percentage points.</p><div id="widthDeltas"></div>'+
        '<h3>Mean fit cost</h3><p>Each arrow shows original → wider. The time ratio is wider PINN time divided by original PINN time. Both sweeps used one RTX 4090, and concurrent GPU load can affect this comparison. The FNO + TTT start includes 150 earlier updates outside the 400 PINN updates.</p><div id="widthCost"></div><p id="widthTiming" class="fineprint"></p>'+
        '<h3>Individual material</h3><div class="controls"><label>Material ID<select id="widthMaterial">'+options(DB.fine_cv_width.ids.map((v,i)=>[i,String(v).padStart(6,"0")]))+'</select></label></div><div id="widthCase"></div>'+
        '<div class="controls"><button class="export" onclick="widthDownload(false)">Download width summaries (CSV)</button><button class="export" onclick="widthDownload(true)">Download paired material results (CSV)</button></div>';
}
