"""Render the final paired benchmark from verified complete result tables."""
import csv
import json
from pathlib import Path
from statistics import median, quantiles
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, Table, TableStyle, SimpleDocTemplate, PageBreak, Spacer

CODE=Path(__file__).resolve().parent
ROOT=CODE/"random"
DEST=CODE/"reports"/"DiffIon_Random_One_Current_Results.pdf"
COLOR="#116563"
for face,file in (("UI","segoeui.ttf"),("UIB","segoeuib.ttf"),("Title","georgia.ttf")):
    pdfmetrics.registerFont(TTFont(face,str(Path("C:/Windows/Fonts")/file)))
pdfmetrics.registerFontFamily("UI",normal="UI",bold="UIB")


def load(path): return json.loads(path.read_text())
def load_csv(path):
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))
def fmt(value, places=2):
    x=float(value)
    return f"{x:.{places}f}" if abs(x)>=10**(-places) or x==0 else f"{x:.1e}"
def p(text,size=10.8,leading=15,after=7,face="UI",tint="#1c343d",right=False):
    return Paragraph(text,ParagraphStyle("body",fontName=face,fontSize=size,leading=leading,
                        textColor=colors.HexColor(tint),spaceAfter=after,alignment=TA_RIGHT if right else 0))
def title(text): return p(text,23,30,9,"Title")
def sub(text): return p(text,13,17,4,"UIB",COLOR)
def small(text): return p(text,9.4,13.4,9,"UI","#455d65")
def table(headers,rows,widths):
    def cells(row,is_head):
        return [p(escape(str(value)),8.7 if is_head else 9.8,11.6 if is_head else 13,0,
                 "UIB" if is_head or k==0 else "UI","#ffffff" if is_head else "#1c343d",k>0)
                for k,value in enumerate(row)]
    ts=Table([cells(headers,True)]+[cells(row,False) for row in rows],colWidths=widths,repeatRows=1,hAlign="LEFT")
    ts.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#163842")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#edf4f3"),colors.white]),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    return ts
def page_number(canvas,doc):
    w,h=doc.pagesize
    canvas.saveState();canvas.setFillColor(colors.HexColor(COLOR));canvas.rect(0,h-5,w,5,stroke=0,fill=1)
    canvas.setFont("UIB",9);canvas.drawString(47,h-37,"CHEME 236   /   DIFFION")
    canvas.setFont("UI",9);canvas.drawRightString(w-47,h-37,"ONE-CURRENT EXPERIMENT")
    canvas.setFillColor(colors.HexColor("#597079"));canvas.setFont("UI",8.5)
    canvas.drawString(47,23,"Random current per material  /  seed 236");canvas.drawRightString(w-47,23,str(doc.page))
    canvas.restoreState()


def main():
    data=ROOT/"report_data"
    rows=load_csv(data/"methods.csv"); lookup={r["method"]:r for r in rows}
    observed={r["method"]:r for r in load_csv(data/"observed_range_comparison.csv")}
    # The paired table uses native-grid concentration extrema for the two
    # native PINN fits, and binned observations for the other main methods.
    for method in observed.keys() & lookup.keys():
        lookup[method]["D_observed_pct"]=observed[method]["one_current_D_observed_pct"]
        lookup[method]["t_observed_rmse"]=observed[method]["one_current_t_observed_rmse"]
    def val(method,key,places=2): return fmt(lookup[method][key],places)
    stats=load_csv(ROOT/"solver_runs/results/summary.csv")
    if sum(int(r["completed"]) for r in stats)!=2805: raise ValueError("Numerical experiment incomplete")
    sv={(r["method"],r["start"]):r for r in stats}
    learned=load(ROOT/"learned/comparison.json")
    comparisons=learned["results"]
    splits=load(ROOT/"learned/frozen_splits.json")["splits"]
    A=comparisons["fno_ttt"]["D_relL2_pct"]
    material_groups={}
    for r in load_csv(data/"materials.csv"):
        material_groups.setdefault(r["method"],[]).append(r)
    def pilot_fno(key):
        return median(float(r[key]) for r in material_groups["fno_ttt_s0"][:8])
    def observed_p95(method):
        values=[float(r["D_observed_pct"]) for r in material_groups[method]]
        return fmt(quantiles(values,n=100,method="inclusive")[94])
    historical = {r["method"]: r for r in load_csv(data/"historical_refinements.csv")}
    def historical_d(method):
        if method=="fno_ttt_pilot":
            return fmt(historical[method]["D_common_pct"])
        return fmt(observed[method]["three_current_D_observed_pct"])
    def historical_t(method):
        if method=="fno_ttt_pilot":
            return fmt(historical[method]["t_common_rmse"],4)
        return fmt(observed[method]["three_current_t_observed_rmse"],4)
    solver_groups=[[float(r["D_observed_pct"]) for r in material_groups["lbfgs_fno_ttt"] if int(r["rank"])==band]
                   for band in range(3)]
    staged=("standard_cv","standard_c","data_first","data_correction","solver_correction","solver_correction_weight1",
            "fine","staged_cv_w100","staged_c_w100","staged_cv_w1","staged_c_w1")
    names=dict(standard_cv="Joint c and v",standard_c="Joint c only",data_first="Data-first joint",data_correction="Data-first + correction",
               solver_correction="Solver-seeded, w = 0.01",solver_correction_weight1="Solver-seeded, w = 1",fine="Native-grid data-first",
               staged_cv_w100="Staged c and v, w = 100",staged_c_w100="Staged c only, w = 100",
               staged_cv_w1="Staged c and v, w = 1",staged_c_w1="Staged c only, w = 1")
    stages=[(m,s) for m in staged for s in ("fno_ttt","center") if "pinn_"+m+"_"+s in lookup]
    expected_pinn={key for key in historical if key.startswith("pinn_")}
    missing_pinn=sorted(expected_pinn-lookup.keys())
    if len(expected_pinn)!=19 or missing_pinn:
        raise ValueError(f"Incomplete PINN comparison: {missing_pinn}")
    if len(lookup)!=42 or sum(len(group) for group in material_groups.values())!=6512:
        raise ValueError("Expected 12 learned, 11 numerical, and 19 PINN groups across 6,512 estimates")
    comparison_rows=[]
    for key,metrics in comparisons.items():
        d,t=metrics["D_relL2_pct"],metrics["tplus_rmse"]
        comparison_rows.append([key,300,3,d["one_current_observed_window"],d["three_current_original_window"],
                                t["one_current_observed_window"],t["three_current_original_window"],
                                d["one_current_original_window"],d["three_current_original_window"],
                                t["one_current_original_window"],t["three_current_original_window"]])
    for key,old in historical.items():
        if key=="fno_ttt_pilot":
            continue
        current=lookup[key]
        if int(current["n"])!=int(old["n"]):
            raise ValueError(f"Historical cohort size differs for {key}")
        comparison_rows.append([key,current["n"],1,current["D_observed_pct"],observed[key]["three_current_D_observed_pct"],
                                current["t_observed_rmse"],observed[key]["three_current_t_observed_rmse"],
                                current["D_common_pct"],old["D_common_pct"],current["t_common_rmse"],
                                old["t_common_rmse"]])
    with (data/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["method","n_materials","runs_per_material","one_current_D_observed_pct","three_current_D_observed_pct",
                         "one_current_t_observed_rmse","three_current_t_observed_rmse",
                         "one_current_D_common_pct","three_current_D_common_pct","one_current_t_common_rmse",
                         "three_current_t_common_rmse"])
        writer.writerows(comparison_rows)
    P="pinn_data_correction_fno_ttt"
    init_by_id={r["material_id"]:float(r["D_observed_pct"]) for r in material_groups["fno_ttt_s0"][:8]}
    improved=sum(float(r["D_observed_pct"])<init_by_id[r["material_id"]] for r in material_groups[P])
    body=[title("What changes with one current?"),p("Each material received one random constant current density. The same draw was used by every method, from supervised training to per-material physics refinement."),
          sub("Property recovery within each protocol’s measured range"),
          p(f"Across the same 300 test materials, FNO + TTT gave <b>{fmt(A['one_current_observed_window'])}%</b> median diffusivity error over the single-current observed range and <b>{fmt(A['three_current_original_window'])}%</b> over the three-current observed range. These are means of three seed-specific medians. Median concentration spans were <b>{fmt(learned['median_observed_span']['one'],3)}</b> and <b>{fmt(learned['median_observed_span']['three'],3)}</b>, respectively."),
          p(f"With one current and FNO + TTT initialization, the differentiable solver reached a median D error of <b>{val('lbfgs_fno_ttt','D_observed_pct')}%</b>. On the eight-case pilot, the data-first PINN correction moved the single-current median from {fmt(pilot_fno('D_observed_pct'))}% to <b>{val(P,'D_observed_pct')}%</b>. It improved {improved} of the eight individual cases."),
          Image(str(data/"diffusivity_comparison.png"),width=504,height=173.6), Spacer(1,8),
          sub("Paired experimental design"),
          p("The main generator assigned each retained material one of its original three current densities with equal probability. The numerical seed combines 236 with the original material ID. The three candidate bands are 0–0.625, 1.25–1.875 and 2.5–3.125 in dimensionless current. The 300 test assignments comprise 87 low, 112 middle and 101 high currents."),
          p("The original splits contain 10,336 training, 593 validation, 1,207 in-distribution test and 2,859 parameter-tail materials. Velocity was reconstructed from the selected concentration trajectory. The single-current input supplied additional valid frames in 121 of the 300 paired test cases."),
          sub("Which concentrations enter the error?"),
          p("Each fit is scored from its minimum to maximum valid measured concentration. The one-current bounds use that trajectory alone. The three-current bounds use all three measured trajectories. The ranges differ for 199/300 materials. Each curve is evaluated on 301 equally spaced points in its own interval. D error is relative L2 in percent, and transference-number error is absolute RMSE."),
          p("For predictive validation, the forward solver replayed the fitted coefficients at a held-out current of 1.0. Concentration errors are RMSE divided by mean true concentration, expressed as a percentage."),
          PageBreak(),
          title("Pretrained estimators and TTT"),p("MLP and FNO checkpoints were trained from scratch on single-current inputs. Each architecture used 100 epochs and three seeds. TTT reset weights per material and optimized the measured trajectory’s PDE residual for 150 AdamW steps."),
          table(["Method","1 current D (%)","3 currents D (%)","1 current t+ RMSE","3 currents t+ RMSE"],[
              [m.replace("_supervised","").replace("_ttt"," + TTT").upper(),fmt(v["D_relL2_pct"]["one_current_observed_window"]),
               fmt(v["D_relL2_pct"]["three_current_original_window"]),fmt(v["tplus_rmse"]["one_current_observed_window"],4),
               fmt(v["tplus_rmse"]["three_current_original_window"],4)] for m,v in comparisons.items()],[105,101,101,101,108]),
          Spacer(1,11),sub("How the random current changes the result"),
          table(["Method","Protocol","Low D (%)","Middle D (%)","High D (%)"],[
              [arch+" + TTT",label,*[fmt(comparisons[arch.lower()+"_ttt"]["D_relL2_pct"][key][str(k)]) for k in range(3)]]
              for arch in ("MLP","FNO") for label,key in (("1 current","one_current_observed_window_by_rank"),("3 currents","three_current_observed_window_by_rank"))],[114,100,98,102,102]),
          small("Materials are grouped by the single current drawn for each one. Both protocols use the same materials per group. Median single-current concentration spans are 0.193, 0.960 and 1.402 in the low, middle and high groups."),
          sub("Material uncertainty and held-out prediction"),
          p(f"The paired material-bootstrap 95% interval for one-current minus three-current FNO + TTT D error is {fmt(A['one_minus_three_observed_window_95pct_material_bootstrap'][0])} to {fmt(A['one_minus_three_observed_window_95pct_material_bootstrap'][1])} percentage points. It captures sampling uncertainty over materials for this current assignment. Each protocol’s measured concentration coverage is part of the comparison."),
          p("The input layer has fewer channels with one experiment. The width-40 MLP has 23,368 parameters versus 65,048 in its three-current version. The FNO keeps the same Fourier modes, width and depth. The comparison therefore also includes the change in MLP input-layer capacity."),
          table(["One-current seed-0 estimate","D, observed (%)","t+ RMSE, observed","Held-out c RMSE (%)"],[
              [s,val(k,"D_observed_pct"),val(k,"t_observed_rmse",4),val(k,"c_unseen1_pct",3)] for s,k in
              [("MLP","mlp_supervised_s0"),("MLP + TTT","mlp_ttt_s0"),("FNO","fno_supervised_s0"),("FNO + TTT","fno_ttt_s0")]],[130,115,132,139]),PageBreak(),
          title("Differentiable and gradient-free inversion"),p("The same eight bounded coefficients were fit to the one observed current. L-BFGS-B used adjoint gradients. Nelder–Mead and CMA-ES used 2,000-evaluation budgets on 300 materials. Bayesian optimization used 150 evaluations on 30 materials."),
          table(["Method and start","N","1 current D (%)","3 currents D (%)","1 current t+ RMSE","3 currents t+ RMSE","1 current c hits¹"],[
               [m+" / "+s,lookup[k]["n"],val(k,"D_observed_pct"),historical_d(k),val(k,"t_observed_rmse",4),historical_t(k),sv[(m0,s)]["reached"]]
               for m,m0,s,k in ([("L-BFGS-B","lbfgs",s,"lbfgs_"+s) for s in ("center","mlp","mlp_ttt","fno","fno_ttt")]
               +[(m,m,s,m+"_"+s) for m in ("NM","CMAES","BO") for s in ("center","fno_ttt")])],[133,35,71,71,71,66,69]),
          small("¹ Concentration RMSE at or below 0.01% of the mean on the fitting trajectory. An optimizer can reach this tolerance while its individual transport curves still have appreciable error. Warm-start labels refer to the fresh single-current seed-0 predictions."),
          sub("Solver evaluations at a common tolerance")]
    costs=[]
    for method,start in (("lbfgs","center"),("lbfgs","fno"),("lbfgs","fno_ttt"),("NM","fno_ttt"),("CMAES","fno_ttt"),("BO","fno_ttt")):
        path=ROOT/"solver_runs/results"/("lbfgs" if method=="lbfgs" else "gradfree")/start
        if method!="lbfgs": path=path/method
        rec=[json.loads(p.read_text())["to_target"]["0.0001"] for p in path.glob("m[0-9][0-9][0-9].json")]
        hit=[v for v in rec if v[0] is not None]
        costs.append([method.upper()+" / "+start,str(len(hit))+" / "+str(len(rec)),fmt(median(v[0] for v in hit),1) if hit else "—",
                      fmt(median(v[1] for v in hit)) if hit else "—"])
    body.extend([table(["Method and start","Cases reaching target","Evals to target","Seconds to target"],costs,[153,129,119,115]),
                 small("Times are medians among successful cases on the shared workstation under concurrent workloads. They exclude pretraining, TTT and warm-start construction. Compact-model outcomes appear on the final page."),
                 p(f"With FNO + TTT initialization, {sum(e>1 for e in solver_groups[0])} of the 87 low-current cases exceeded 1% D error on their one-current range after L-BFGS-B. Their median error was {fmt(median(solver_groups[0]),4)}%. Among the other 213 cases, {sum(e>1 for group in solver_groups[1:] for e in group)} exceeded 1%. The overall 95th percentile was {observed_p95('lbfgs_fno_ttt')}%."),PageBreak(),
                 title("PINN refinement on eight materials"),p("All PINN variants use the selected current of the same eight pilot materials. The learned start is seed-0 FNO + TTT. A center start sets every normalized coefficient to 0.5."),
                 table(["Variant","Start","1 current D (%)","3 currents D (%)","1 current t+ RMSE","3 currents t+ RMSE","1 current held-out c (%)"],[
                     ["FNO + TTT","Initial",fmt(pilot_fno("D_observed_pct")),historical_d("fno_ttt_pilot"),fmt(pilot_fno("t_observed_rmse"),4),historical_t("fno_ttt_pilot"),fmt(pilot_fno("c_unseen1_pct"),3)]]
                     +[[names[m],"Learned" if s=="fno_ttt" else "Center",val("pinn_"+m+"_"+s,"D_observed_pct"),historical_d("pinn_"+m+"_"+s),val("pinn_"+m+"_"+s,"t_observed_rmse",4),historical_t("pinn_"+m+"_"+s),val("pinn_"+m+"_"+s,"c_unseen1_pct",3)] for m,s in stages if not m.startswith("staged_")],[129,57,67,67,66,65,65]),
                 small("Each protocol uses its observed concentration interval for the same eight materials. Held-out concentration error uses forward predictions from the one-current fitted coefficients."),
                 sub("How well does fitting concentration recover diffusivity?"),
                 p(f"With one current, fitting the joint c and v PINN changed D error from {fmt(pilot_fno('D_observed_pct'))}% to {val('pinn_standard_cv_fno_ttt','D_observed_pct')}% over that current’s measured interval. Its median error on the fitted concentration field was 0.274%. Data-first joint fitting followed by correction gave {val(P,'D_observed_pct')}% D error with one current and {historical_d(P)}% with three currents over their respective intervals."),
                 PageBreak(),title("What staging and grid resolution change"),
                 table(["Staged variant","Start","1 current D (%)","3 currents D (%)","1 current t+ RMSE","3 currents t+ RMSE"],[
                     [names[m],"Learned" if s=="fno_ttt" else "Center",val("pinn_"+m+"_"+s,"D_observed_pct"),historical_d("pinn_"+m+"_"+s),val("pinn_"+m+"_"+s,"t_observed_rmse",4),historical_t("pinn_"+m+"_"+s)] for m,s in stages if m.startswith("staged_")],[157,68,73,73,72,73]),
                 small("The weight refers to the staged physics loss. Native-grid fits use concentration extrema across 40 cells and up to 104 times per current. Other main fits use 20 bins and up to 13 times."),
                 sub("Prediction at an independently held-out current"),
                 p("One-current fitted coefficients were passed through the forward solver at a new dimensionless current of 1.0. The table shows median prediction error there, the 95th percentile of D error over the fitted protocol’s observed range, and the number of admissible held-out trajectories."),
                 table(["One-current method","N","Held-out c RMSE (%)","95th percentile D, observed (%)","Admissible held-out replays"],[
                     [name,lookup[m]["n"],val(m,"c_unseen1_pct",4),observed_p95(m),lookup[m]["n_unseen_admissible"]]
                     for name,m in (("FNO + TTT, seed 0","fno_ttt_s0"),("L-BFGS-B, center","lbfgs_center"),("L-BFGS-B, learned","lbfgs_fno_ttt"),
                         ("Nelder–Mead, learned","NM_fno_ttt"),("CMA-ES, learned","CMAES_fno_ttt"),("Bayesian opt., learned","BO_fno_ttt"),
                         ("Data-first + correction",P))],[147,39,104,111,115]),Spacer(1,10),
                 sub("Frozen checkpoints outside the training distribution"),
                 table(["Cohort","N","MLP D (%)","FNO D (%)","MLP t+ RMSE","FNO t+ RMSE"],[
                     [label,splits[key]["n"],*[fmt(splits[key]["results"][arch]["mean_of_seed_medians"][metric],places)
                      for metric,places in (("D_relL2_pct",2),("tplus_rmse",4)) for arch in ("mlp","fno")]]
                     for key,label in (("id_test","In-distribution"),("ood_params","Parameter tails"))],[136,53,78,78,85,86]),
                 small("Both cohorts use each material’s one-current observed window. Results average the medians from three independently trained checkpoints before TTT."),PageBreak(),
                 title("Independent checks and earlier campaign"),sub("How much information does the current supply?"),
                 p("On the eight pilot materials, the median condition number of the concentration-versus-coefficient Jacobian was 4,507 with each material’s selected current and all its available frames. Holding the observation horizon fixed, it was 7,727 with one current and 1,185 with all three. A larger value means equal-sized changes in different coefficient directions have more unequal effects on concentration."),
                 sub("Compact optimizer control"),
                 table(["Target c RMSE","Nelder–Mead hits","CMA-ES hits","Bayesian opt. hits"],[
                     [lab,*[sv[("compact",key+"/"+m)]["reached"]+" / 5" for m in ("NM","CMAES","BO")]]
                     for key,lab in (("target_1pct","1%"),("target_0.1pct","0.1%"),("target_0.01pct","0.01%"))],[121,138,122,135]),
                 small("The compact solver has one reference material with five optimizer starts. Its random draw selected dimensionless current 0.6 from the original 0.3, 0.6, 0.9 set. Its simplified transport model uses a separate parameterization."),
                 sub("Earlier clean and X-ray-noisy CNN / PINN campaign")])
    legacy_observed=load(data/"legacy_observed_ranges.json")
    legacy=[]
    for d in ("clean","xray_noisy"):
        domain=legacy_observed[d]
        for label in ("one","three"):
            cnn,pinn=domain["cnn"][label],domain["pinn"][label]
            legacy.append([d,"1 current" if label=="one" else "3 currents",fmt(cnn["combined_normalized_rmse"],4),
                           pinn["status"].capitalize(),fmt(pinn["D_normalized_rmse"],4),fmt(pinn["t_normalized_rmse"],4)])
    body.extend([table(["Domain","Protocol","CNN combined RMSE","PINN status","PINN D RMSE","PINN t+ RMSE"],legacy,[95,83,85,95,77,81]),
                 small("CNN scores use the measured concentration range of each single input or triplet. The 389 test materials are equally weighted after averaging triplet squared errors within each material. The PINN comparison uses material 2 and the original five acceptance criteria. Rejected fits show the final candidate."),
                 sub("Reading the files"),
                 p("The random experiment directory contains the selected current rank, material IDs, trained checkpoints and per-fit outputs. report_data/comparison.csv gives both protocols’ observed-range errors for all 34 configurations. solver_runs/results/summary.csv records coverage and cost. Per-case PINN files retain field losses and timing."),
                 small("The main results use noiseless synthetic data and one randomized assignment. The two independent campaigns have different grids, data generators, and scoring scales. Absolute numbers across those campaigns have different interpretations.")])
    DEST.parent.mkdir(exist_ok=True,parents=True)
    doc=SimpleDocTemplate(str(DEST),pagesize=(8.5*inch,11*inch),rightMargin=47,leftMargin=47,topMargin=57,bottomMargin=40,
                          title="DiffIon | One random current per material")
    doc.build(body,onFirstPage=page_number,onLaterPages=page_number)
    print(DEST)


if __name__=="__main__":
    main()
