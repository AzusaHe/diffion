const gridSpecs = DB.grid_comparison?.grids ?? Array.from(new Map(
    (DB.grid_comparison?.rows??[]).map(r=>[r.space_bins,r.max_frames])
)).sort((a,b)=>a[0]-b[0]);
const gridSizes = gridSpecs.map(([w])=>w);
const gridLabels = gridSpecs.map(([w,t])=>w+" × "+t);
const fineNetworkSlugs = gridSizes.filter(w=>w>20).flatMap(w=>["conv_mlp_grid"+w,"fno_grid"+w]);
const gridFamilies = {fno:"FNO", mlp:"MLP → Conv-MLP"};
let gridSeed = 0;

function gridHead(left, groups, labels) {
    return '<thead><tr>'+left.map(t=>'<th rowspan="2">'+t+'</th>').join("")+
        groups.map(t=>'<th colspan="'+labels.length+'">'+t+'</th>').join("")+'</tr><tr>'+
        groups.map(()=>labels.map(t=>'<th>'+t+'</th>').join("")).join("")+'</tr></thead>';
}

function gridTable(rows, makeCells) {
    const body = Object.entries(gridFamilies).map(([family,name])=>{
        const bodyRows=gridSizes.map((w,i)=>{
            const pair=[1,3].map(n=>rows.find(r=>r.family===family&&r.currents===n&&r.space_bins===w));
            return pair.every(Boolean)?'<tr><td>'+gridLabels[i]+'</td>'+pair.map(makeCells).join("")+'</tr>':"";
        }).join("");
        return bodyRows?'<tbody><tr class="group"><th colspan="5" scope="rowgroup">'+name+'</th></tr>'+bodyRows+'</tbody>':"";
    }).join("");
    return wrap(gridHead(["Input grid"],["One current","Three currents"],["D error (%)","t₊ RMSE"])+body);
}

function gridErrors(rows, scope) {
    const metrics=[["D_median_pct",2],["t_median_rmse",4]];
    return gridTable(rows,r=>{
        const family=rows.filter(q=>q.family===r.family&&q.currents===r.currents);
        return metrics.map(([metric,digits])=>{
            const key=scope+"_"+metric, value=num(r[key],digits);
            const lowest=family.length===gridSizes.length&&family.every(q=>q[key]>=r[key]);
            return '<td>'+(lowest?'<strong>'+value+'</strong>':value)+'</td>';
        }).join("");
    });
}

function gridChart(metric, label) {
    const key=[dom(),gridSeed,metric].join("__"), png=DB.grid_comparison.figures?.[key];
    if(!png)return "";
    const uri="data:image/png;base64,"+png;
    return '<div><img src="'+uri+'" alt="'+esc(label)+' by input resolution" style="width:100%;height:auto">'+
        '<a class="fineprint" href="'+uri+'" download="grid_'+key+'.png">Download '+esc(label)+' plot (PNG)</a></div>';
}

function showGrids() {
    const db=DB.grid_comparison;
    const all=db.rows.filter(r=>r.domain===dom());
    const rows=all.filter(r=>r.seed===gridSeed);
    let h=title("Supervised recovery across input grids")+
        '<p class="sub">Each grid uses the same 1,402 training, 156 validation and 389 test materials. The one-current measurement is the first member of each material’s three-current set. Grid dimensions are spatial bins × maximum observed frames per current.</p>'+
        '<p>The FNO keeps its architecture and parameter count at every resolution. The 20 × 13 MLP has a dense input. All finer Conv-MLPs reuse a 3 × 3 convolution with three output channels. The number of downsampling passes depends on grid size'+(gridSizes.includes(290)?', and the 290 × 151 encoder also uses mask-aware area averaging':'')+'. Conv-MLP comparisons therefore include changes in downsampling as well as resolution, with a fixed parameter count across the finer grids.</p>'+
        '<div class="controls"><label>Training seed<select id="gridSeed">'+[0,1,2].map(s=>'<option value="'+s+'"'+(s===gridSeed?' selected':'')+'>'+s+(s===0?" (primary)":"")+'</option>').join("")+'</select></label></div>';
    if (!rows.length) return h+'<p>No paired results are available for this selection.</p>';
    h+='<h3>Same concentration interval for all '+gridSizes.length+' grids</h3><p>For each material, intersect the observed concentration ranges of '+gridLabels.join(", ")+' for the selected domain and current count, then clip to [0, 2]. Score each prediction, including the archived predictions, at the same 301 concentrations on that interval. Entries are medians across the 389 materials. Bold identifies the lowest unrounded error across grids within each model family and current count.</p>'+
       gridErrors(rows,"common");
    if(rows.length===4*gridSizes.length&&rows.every(r=>r.n_common_bounds_equal_20===r.n_test))h+='<p class="fineprint">For this domain, the shared range exactly matches the 20 × 13 observed range on all 389 materials under both current protocols.</p>';
    h+='<div class="graphs">'+gridChart("D_median_pct","D error")+gridChart("t_median_rmse","t₊ RMSE")+'</div>';
    h+='<h3>Each grid’s own observed interval</h3><p>Each prediction is scored over its own observed range, clipped to [0, 2]. These are the values on the <a href="#accuracy">Accuracy tab</a> for seed 0. Three-current and one-current inputs have their own intervals throughout.</p>'+gridErrors(rows,"own");
    const spanRows=[...gridSizes,0].map((w,i)=>{
        const values=[1,3].map(n=>rows.find(r=>r.family==="fno"&&r.currents===n&&r.space_bins===(w||gridSizes[0])));
        if(!values.every(Boolean))return "";
        return '<tr><td>'+(w?gridLabels[i]:"Shared interval")+'</td>'+values.map(r=>'<td>'+num(w?r.own_median_c_span:r.common_median_c_span,4)+'</td>').join("")+'</tr>';
    }).join("");
    h+='<details><summary>Concentration span used for scoring</summary><p>Median of each material’s upper minus lower scoring bound, in dimensionless concentration.</p>'+
        wrap('<thead><tr><th>Scoring range</th><th>One-current span</th><th>Three-current span</th></tr></thead><tbody>'+spanRows+'</tbody>')+'</details>';
    h+='<h3>Training and inference cost</h3><p>Training time includes validation. Inference is the elapsed time for all 389 prepared inputs at batch size 128, averaged over ten passes and divided by 389. Timings are wall clock on a shared workstation where other CPU jobs ran during parts of the experiment. The original MLP ran on CPU and the Conv-MLPs ran on the RTX 4090, so their timings also reflect the hardware change.</p>';
    const costRows=rows.map(r=>'<tr><td>'+esc(r.model==="fno"?"FNO":r.model==="mlp"?"MLP":"Conv-MLP")+'</td><td>'+r.currents+'</td><td>'+r.space_bins+' × '+r.max_frames+'</td><td>'+num(r.parameters,0)+'</td><td>'+ms(r.train_s)+'</td><td>'+ms(r.mean_inference_s_per_material)+'</td><td>'+esc(r.device)+'</td></tr>').join("");
    h+=wrap('<thead><tr><th>Network</th><th>Currents</th><th>Grid</th><th>Trainable parameters</th><th>Fit time</th><th>Inference/material</th><th>Device</th></tr></thead><tbody>'+costRows+'</tbody>');
    const prep=db.preparation;
    const prepared=gridSpecs.filter(([w])=>prep?.[w]);
    if(prepared.length)h+='<p class="fineprint">Building the reusable observations for both domains took '+prepared.map(([w,t])=>ms(prep[w].prepare_wall_s)+' at '+w+' × '+t).join(", ")+'. Preprocessing and model training are outside the inference timing.</p>';
    if(prep?.[160]?.timing_definition&&prep?.[290]?.timing_definition)h+='<p class="fineprint">Both of the largest grids were prepared in one pass. Their times measure elapsed wall time from the same start until each grid was published.</p>';
    if(all.length===gridSizes.length*12){
        const cells=r=>{
            const reps=all.filter(q=>q.space_bins===r.space_bins&&q.family===r.family&&q.currents===r.currents);
            return ["common_D_median_pct","common_t_median_rmse"].map((k,i)=>{
                const values=reps.map(q=>q[k]), mean=values.reduce((a,b)=>a+b,0)/values.length;
                const sd=Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0)/(values.length-1));
                return '<td>'+num(mean,i?4:2)+' ± '+num(sd,i?4:2)+'</td>';
            }).join("");
        };
        h+='<details><summary>Variation across training seeds 0, 1 and 2</summary><p>Mean ± sample standard deviation across three per-seed medians on the shared concentration interval. The same 389 held-out materials appear in each seed.</p>'+
            gridTable(rows,cells)+'</details>';
    }
    return h+'<h3>Inputs and training</h3><p>The 1,160 native spatial cells are averaged into equal-width bins using the volume overlap with each native cell. Across the grids shown, a bin spans '+gridSizes.map(w=>1160/w).join(", ")+' native cells, respectively. Frames are evenly selected from valid measurements, including both endpoints. Shorter trajectories contribute their available frames and use a validity mask.'+(gridSizes.includes(290)?' The 290 × 151 input retains every valid recorded time and averages four native spatial cells per bin.':'')+'</p>'+
        '<p>Each domain, current count, model and training seed has a separate supervised checkpoint. The 20 × 13 fits come from the original benchmark. The '+(gridSizes.length-1)+' finer grids contribute '+((gridSizes.length-1)*24)+' fits across clean and Gaussian-noisy observations. All fits use 100 epochs, batch size 64 and AdamW at an initial learning rate of 0.001 with cosine decay. Validation selects the epoch with the lowest MSE on eight normalized Bernstein coefficients. The <a href="#architecture">NN architecture tab</a> lists layers, activations, trainable parameters and settings for each grid.</p>'+
        '<button class="export" onclick="downloadGridCSV()">Download all grids, seeds and domains (CSV)</button>';
}

function downloadGridCSV(){
    const url=URL.createObjectURL(new Blob([DB.grid_comparison.csv],{type:"text/csv;charset=utf-8"}));
    const a=document.createElement("a");a.href=url;a.download="grid_accuracy_cost.csv";a.click();
    setTimeout(()=>URL.revokeObjectURL(url),500);
}
