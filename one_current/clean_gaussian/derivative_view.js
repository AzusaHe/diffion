// Embedded in the portable HTML by report.py.
let gradientSelected = "pinn_data_first_fno_ttt";

function gradientRows() {
    return (DB.derivatives?.conditions || []).filter(r =>
        r.domain === dom() && r.currents === Number(byId("gradCount").value) && r.start === byId("gradStart").value);
}

function pairedChange(value, precision) {
    const label = (value > 0 ? "+" : "") + num(value, precision);
    return '<span class="' + (value < 0 ? "paired-improved" : value > 0 ? "paired-worse" : "") + '">' + label + '</span>';
}

function pairedChart(cases, key, metric, precision) {
    const deltas = cases.map(v => v[key+"1"] - v[key+"0"]);
    const scale = Math.max(...deltas.map(Math.abs)) || 1;
    const center = 218, span = 132, rowHeight = 19;
    const labels = cases.map((v,i) => {
        const diff = deltas[i], end = center + span * diff / scale, y = i * rowHeight + 19;
        return '<g><title>Material ' + v.id + ' · ' + num(diff,precision) + '</title>' +
            '<text x="6" y="'+(y+3)+'" font-size="11">'+String(v.id).padStart(6,"0")+'</text>' +
            '<rect x="'+Math.min(center,end)+'" y="'+(y-4)+'" width="'+Math.max(Math.abs(end-center),.4)+
            '" height="8" fill="'+(diff < 0 ? '#167662' : '#a95237')+'"/></g>';
    }).join("");
    const height = rowHeight * cases.length + 35;
    return '<figure class="paired-chart"><figcaption>'+metric+'</figcaption><svg role="img" aria-label="'+metric+
        ' by material. Exact values appear in the table below." viewBox="0 0 385 '+height+'">' +
        '<line x1="'+center+'" y1="8" x2="'+center+'" y2="'+(height-25)+'" stroke="#80978b"/>'+labels+
        [[center-span,"−"+num(scale,precision)],[center,"0"],[center+span,"+"+num(scale,precision)]].map(([x,t])=>
            '<text text-anchor="middle" x="'+x+'" y="'+(height-6)+'" font-size="11">'+t+'</text>').join("")+'</svg></figure>';
}

function derivativeCases(row) {
    const related=DB.derivatives.conditions.filter(r=>r.method===row.method);
    const allSettings='<h4>Same method across all four observation settings</h4><p class="hint">Start gives the error of the initial material coefficients, before any PINN update, on each material’s observed interval for that row.</p>'+
        wrap('<thead><tr><th rowspan="2">Observations</th><th rowspan="2">Currents</th><th colspan="3">D relative L₂ (%)</th><th colspan="3">t₊ RMSE</th><th colspan="2">Time per material</th></tr><tr>'+'<th>Start</th><th>Control</th><th>With slopes</th>'.repeat(2)+'<th>Control</th><th>With slopes</th></tr></thead><tbody>'+
            related.map(r=>'<tr><td>'+(r.domain==="clean"?"Clean":"Gaussian noise")+'</td><td>'+r.currents+'</td><td>'+
                num(r.D_initial_pct,2)+'</td><td>'+num(r.D_control_pct,2)+'</td><td>'+num(r.D_derivative_pct,2)+'</td><td>'+
                num(r.t_initial_rmse,4)+'</td><td>'+num(r.t_control_rmse,4)+'</td><td>'+num(r.t_derivative_rmse,4)+
                '</td><td>'+ms(r.control_s_per_material)+'</td><td>'+ms(r.derivative_s_per_material)+'</td></tr>').join("")+'</tbody>');
    const table = '<thead><tr><th>Material</th><th>Scored c interval</th><th>D control (%)</th><th>D with slopes (%)</th><th>Δ D (pp)</th><th>t₊ control</th><th>t₊ with slopes</th><th>Δ t₊</th></tr></thead><tbody>'+
        row.cases.map(c => '<tr><td>'+String(c.id).padStart(6,"0")+'</td><td>'+num(Math.max(0,c.lo),3)+' – '+num(Math.min(2,c.hi),3)+
            '</td><td>'+num(c.d0,2)+'</td><td>'+num(c.d1,2)+'</td><td>'+pairedChange(c.d1-c.d0,3)+
            '</td><td>'+num(c.t0,4)+'</td><td>'+num(c.t1,4)+'</td><td>'+pairedChange(c.t1-c.t0,5)+'</td></tr>').join("")+'</tbody>';
    return allSettings+'<h4>Individual materials in the selected setting</h4><div class="graphs">'+pairedChart(row.cases,"d","Change in D error (percentage points)",2)+
        pairedChart(row.cases,"t","Change in t₊ RMSE",4)+'</div><div id="gradientCaseTable">'+wrap(table)+'</div>';
}

function stageComparison(rows) {
    const base=rows.find(r=>r.method.includes("_data_first_"));
    const variants=rows.filter(r=>r.method.includes("_frozen_")||r.method.includes("_data_correction_"));
    if(!base || !variants.length) return "";
    const body=variants.map(r=>{
        const changes=key=>r.cases.map((v,i)=>v[key]-base.cases[i][key]);
        const d=changes("d1"), t=changes("t1");
        return '<tr><td>'+esc(r.name)+'</td><td>'+pairedChange(median(d),3)+'</td><td>'+d.filter(x=>x<0).length+'/'+r.n+
            '</td><td>'+pairedChange(median(t),5)+'</td><td>'+t.filter(x=>x<0).length+'/'+r.n+
            '</td><td>'+num(r.derivative_s_per_material/base.derivative_s_per_material,2)+'×</td></tr>';
    }).join("");
    return '<h3>What changed when the fields were frozen or corrected?</h3><p class="hint">All fits in this comparison use the slope loss. The reference is data-first training, which refines c, v and the material properties jointly after field pretraining. The frozen-field variant uses the same budget. The data-seeded correction adds a 500-step Adam and 16-iteration L-BFGS stage. Negative changes indicate lower material error than the reference.</p>'+
        wrap('<thead><tr><th>Method vs. data-first</th><th>Paired Δ D (pp)</th><th>D wins</th><th>Paired Δ t₊</th><th>t₊ wins</th><th>Time / reference</th></tr></thead><tbody>'+body+'</tbody>');
}

function derivativePhysics() {
    const cases = (DB.derivatives?.physics || []).filter(r=>r.domain===dom() && r.currents===Number(byId("gradCount").value));
    if(!cases.length) return "";
    const body = [0,1].map(w=>{
        const selected = cases.filter(r=>r.derivative_weight===w);
        return '<tr><td>'+(w ? 'Value + slope' : 'Value only')+'</td><td>'+num(median(selected.map(r=>r.fitted)),3)+
            '</td><td>'+num(median(selected.map(r=>r.truth)),3)+'</td><td>'+selected.filter(r=>r.fitted<r.truth).length+' / '+selected.length+'</td></tr>';
    }).join("");
    return '<h3>Physics residual at fitted and true material properties</h3><p class="paired-notes">For each data-first fit initialized with FNO + TTT, keep the learned c and v fields fixed. Evaluate the sum of the salt, volume and wall residual losses twice, substituting either the fitted or the true material coefficients. Both evaluations use the same 2,048 fresh interior points and 256 wall times per current. The table shows median residuals and the number of materials with a lower residual at the fitted coefficients.</p>'+
        wrap('<thead><tr><th>Field training</th><th>Fitted coefficients</th><th>True coefficients</th><th>Fitted lower</th></tr></thead><tbody>'+body+'</tbody>');
}

function derivativeSchedule() {
    const configs = [
        ["Standard · c and v", "500 joint Adam steps. The control fits concentration values."],
        ["Staged · c and v", "350 Adam steps fitting observed data, 100 Adam steps with physics and fixed material properties, then 500 joint Adam steps and 16 L-BFGS iterations. The control fits concentration values. The two variants use physics weight 1 or 100."],
        ["Data-first / fine grid", "850 Adam steps and 30 L-BFGS iterations fitting c and reconstructed v with fixed properties. Then 500 Adam steps and 16 L-BFGS iterations updating both fields and properties."],
        ["Two-stage · frozen fields", "Same budget as data-first. After fitting the fields, freeze both networks and optimize only the eight material logits in stage two."],
        ["Data-first + correction", "Complete the data-first schedule, freeze its field networks, then fit two residual networks and the material logits with 500 Adam steps and 16 L-BFGS iterations."],
        ["Solver + correction", "Compute a forward solution at the FNO + TTT estimate. Fit a shared field to that solution for 850 Adam steps and 30 L-BFGS iterations. Freeze the base fields, then update the residual networks and properties on observations for 500 Adam steps and 16 L-BFGS iterations. The two variants use data weight 0.01 or 1."],
    ];
    return '<details><summary>Loss, networks and optimizer schedules</summary><div class="paired-notes"><p>For a field q, take the difference of neighboring predicted values and divide by bin spacing h, then compare it with the same difference of neighboring target values. Use only adjacent valid bins at the same current and measured time. The concentration prediction is a two-point quadrature average within each bin. Velocity uses the bin-center prediction.</p>'+
        '<p>The added term is the mean squared spatial slope error divided by (√2 σ<sub>q</sub>/h)², separately for c and v. Both terms have multiplier 1 inside the data loss. The control uses 0. The parent data weight still applies at each stage. The benchmark uses σ<sub>c</sub> = 0.001 × mean observed c and σ<sub>v</sub> = 0.001 × the material’s velocity scale as reference scales. Reconstructed velocity depends on the concentration observations, so its errors can be correlated. Time derivatives enter velocity reconstruction and the physical equations; this experiment supervises spatial slopes.</p>'+
        '<p>Each material has two 3 → 32 → 32 → 32 → 1 tanh networks with 2,273 parameters apiece and eight material logits. Corrections add two 2,273-parameter residual networks with a sine first hidden layer, tanh hidden layers and a zero-initialized output. All four domain/current conditions use the same field architectures and stage budgets. The usual grid contains up to 13 frames × 20 bins per current. The fine-grid variant uses up to 104 frames × 40 bins. See <a href="#architecture">NN architecture</a> for the optimizer parameters and learning rates for each original family.</p></div>'+
        '<ul>'+configs.map(([name,description])=>'<li><b>'+name+'</b> · '+description+'</li>').join("")+'</ul></details>';
}

function downloadDerivativeCases() {
    const keys = ["id","lo","hi","d0","d1","t0","t1","c0","c1","v0","v1","c_value0","c_value1","v_value0","v_value1"];
    const lines = [["domain","currents","method","initialization","material_id","observed_c_min","observed_c_max","D_control_pct","D_derivative_pct","t_control_rmse","t_derivative_rmse","c_slope_control_rmse","c_slope_derivative_rmse","v_slope_control_rmse","v_slope_derivative_rmse","c_value_control_relative_rmse","c_value_derivative_relative_rmse","v_value_control_relative_rmse","v_value_derivative_relative_rmse","mean_control_pipeline_seconds_per_material","mean_derivative_pipeline_seconds_per_material"]];
    for (const r of DB.derivatives.conditions) for (const c of r.cases)
        lines.push([r.domain,r.currents,r.method,r.start,...keys.map(k=>c[k]),r.control_s_per_material,r.derivative_s_per_material]);
    const url=URL.createObjectURL(new Blob([lines.map(line=>line.join(",")).join("\r\n")],{type:"text/csv"}));
    const anchor=document.createElement("a");
    anchor.href=url; anchor.download="diffion_derivative_material_pairs.csv"; anchor.click();
    setTimeout(()=>URL.revokeObjectURL(url),500);
}

function showDerivatives() {
    const rows = gradientRows();
    let html = title("Does fitting observed spatial slopes improve property recovery?");
    if(!rows.length) return html+"<p>Paired derivative-loss results are unavailable for this selection.</p>";
    const n=rows[0].n;
    html += '<p class="sub paired-notes">Paired fits on the same '+n+' held-out materials. Each pair uses the same concentration data, material initialization and step budget. The added targets are spatial slopes of measured concentration and of velocity reconstructed from those same concentration profiles, applied current and mass balance.</p>';
    const dLower=rows.filter(r=>r.D_derivative_pct<r.D_control_pct).length;
    const tLower=rows.filter(r=>r.t_derivative_rmse<r.t_control_rmse).length;
    html += '<div class="summary">In the selected setting, derivative supervision reduced median D error for <b>'+dLower+' of '+rows.length+'</b> methods and median t₊ RMSE for <b>'+tLower+' of '+rows.length+'</b>.</div>';
    const all=DB.derivatives.conditions, improved=(a,b)=>all.filter(r=>r[b]<r[a]).length;
    html += '<p class="paired-notes">Across all '+all.length+' combinations of method, initialization, noise and current count, the median slope RMSE fell in '+
        improved("c_slope_control_rmse","c_slope_derivative_rmse")+' combinations for c and '+improved("v_slope_control_rmse","v_slope_derivative_rmse")+
        ' for v. Median property error fell in '+improved("D_control_pct","D_derivative_pct")+' for D and '+improved("t_control_rmse","t_derivative_rmse")+
        ' for t₊. These descriptive counts reuse the same eight materials throughout.</p>';
    html += '<h3>Transport properties on each fit’s observed c range</h3><p class="hint">The control and slope columns are medians across materials. Δ is the median of the eight material-by-material changes (with slopes minus control). Lower values favor slope supervision. D changes are in percentage points. Wins count materials with a strict error reduction.</p>';
    const hd = '<thead><tr><th rowspan="2">Method</th><th colspan="4">D relative L₂ (%)</th><th colspan="4">t₊ RMSE</th></tr><tr>'+'<th>Control</th><th>With slopes</th><th>Paired Δ</th><th>Wins</th>'.repeat(2)+'</tr></thead>';
    html += wrap(hd+'<tbody>'+rows.map(r=>'<tr><td>'+esc(r.name)+'</td><td>'+num(r.D_control_pct,2)+'</td><td>'+num(r.D_derivative_pct,2)+
        '</td><td>'+pairedChange(r.D_paired_delta_pp,3)+'</td><td>'+r.D_improved+'/'+n+'</td><td>'+num(r.t_control_rmse,4)+
        '</td><td>'+num(r.t_derivative_rmse,4)+'</td><td>'+pairedChange(r.t_paired_delta_rmse,5)+'</td><td>'+r.t_improved+'/'+n+'</td></tr>').join("")+'</tbody>');
    html += stageComparison(rows);
    html += '<h3>Observed slope fit and computational cost</h3><p class="hint">Slope errors are per-material RMSEs in dimensionless space, summarized by their median. These use the fitted observations, so they measure agreement with the supplied slopes. Runtime is mean wall time per material from prepared observations and includes the FNO + TTT initialization where used, any required prefit or forward solve, optimization and final diagnostics. Reading raw files and making the binned inputs precede timing. Shared checkpoint training is accounted for in <a href="#cost">Compute cost</a>.</p>';
    html += wrap('<thead><tr><th rowspan="2">Method</th><th colspan="2">∂c/∂x RMSE</th><th colspan="2">∂v/∂x RMSE</th><th colspan="2">Time per material</th></tr><tr>'+'<th>Control</th><th>With slopes</th>'.repeat(3)+'</tr></thead><tbody>'+
        rows.map(r=>'<tr><td>'+esc(r.name)+'</td><td>'+num(r.c_slope_control_rmse,4)+'</td><td>'+num(r.c_slope_derivative_rmse,4)+'</td><td>'+
            num(r.v_slope_control_rmse,5)+'</td><td>'+num(r.v_slope_derivative_rmse,5)+'</td><td>'+ms(r.control_s_per_material)+'</td><td>'+ms(r.derivative_s_per_material)+'</td></tr>').join("")+'</tbody>');
    html += '<details><summary>Agreement with the observed c values and reconstructed v values</summary><p class="hint">Median relative RMSE, expressed in percent. Each material’s c RMSE is divided by its mean observed concentration and v RMSE by its velocity scale (peak absolute reconstructed v, floored at 10⁻⁵).</p>'+
        wrap('<thead><tr><th rowspan="2">Method</th><th colspan="2">c value relative RMSE (%)</th><th colspan="2">v value relative RMSE (%)</th></tr><tr>'+'<th>Control</th><th>With slopes</th>'.repeat(2)+'</tr></thead><tbody>'+
            rows.map(r=>'<tr><td>'+esc(r.name)+'</td>'+["c_value0_median","c_value1_median","v_value0_median","v_value1_median"].map(k=>'<td>'+num(100*r[k],3)+'</td>').join("")+'</tr>').join("")+'</tbody>')+'</details>';
    if (rows[0].start==="fno_ttt") html += derivativePhysics();
    const selected = rows.find(r=>r.method===gradientSelected) || rows.find(r=>r.method.includes("_data_first_")) || rows[0];
    gradientSelected=selected.method;
    html += '<h3>Inspect a method</h3><div class="controls"><label>Method<select id="derivativeMethod">'+
        rows.map(r=>'<option value="'+r.method+'"'+(r.method===selected.method?' selected':'')+'>'+esc(r.name)+'</option>').join("")+
        '</select></label></div><div id="gradientCases">'+derivativeCases(selected)+'</div>';
    html += derivativeSchedule();
    const archive=rows.map(r=>r.archive_diff).filter(Boolean);
    if (archive.length) html+='<p class="fineprint">The paired controls use the same schedules as the original benchmark. On the displayed methods with archived predictions, the largest absolute repeat-versus-archive differences for any material were '+num(Math.max(...archive.map(v=>v[0])),3)+' percentage points in D error and '+num(Math.max(...archive.map(v=>v[1])),5)+' in t₊ RMSE.</p>';
    return html+'<button class="export" onclick="downloadDerivativeCases()">Download all material pairs (CSV)</button>';
}
