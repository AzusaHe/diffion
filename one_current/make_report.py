"""Render the final paired benchmark from verified complete result tables."""
import csv
import json
from pathlib import Path
from statistics import median
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
    def val(method,key,places=2): return fmt(lookup[method][key],places)
    stats=load_csv(ROOT/"solver_runs/results/summary.csv")
    if sum(int(r["completed"]) for r in stats)!=2805: raise ValueError("Numerical experiment incomplete")
    sv={(r["method"],r["start"]):r for r in stats}
    comparisons=load(ROOT/"learned/comparison.json")["results"]
    splits=load(ROOT/"learned/frozen_splits.json")["splits"]
    A=comparisons["fno_ttt"]["D_relL2_pct"]
    material_groups={}
    for r in load_csv(data/"materials.csv"):
        material_groups.setdefault(r["method"],[]).append(r)
    def pilot_fno(key):
        return median(float(r[key]) for r in material_groups["fno_ttt_s0"][:8])
    historical = {r["method"]: r for r in load_csv(data/"historical_refinements.csv")}
    def historical_d(method):
        return fmt(historical[method]["D_common_pct"]) if method in historical else "—"
    solver_groups=[[float(r["D_common_pct"]) for r in material_groups["lbfgs_fno_ttt"] if int(r["rank"])==band]
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
        comparison_rows.append([key,300,3,d["one_current_observed_window"],d["one_current_original_window"],
                                d["three_current_original_window"],t["one_current_observed_window"],
                                t["one_current_original_window"],t["three_current_original_window"]])
    for key,old in historical.items():
        if key=="fno_ttt_pilot":
            continue
        current=lookup[key]
        if int(current["n"])!=int(old["n"]):
            raise ValueError(f"Historical cohort size differs for {key}")
        comparison_rows.append([key,current["n"],1,current["D_observed_pct"],current["D_common_pct"],
                                old["D_common_pct"],current["t_observed_rmse"],current["t_common_rmse"],
                                old["t_common_rmse"]])
    with (data/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["method","n_materials","runs_per_material","one_current_D_observed_pct","one_current_D_common_pct",
                         "three_current_D_common_pct","one_current_t_observed_rmse","one_current_t_common_rmse",
                         "three_current_t_common_rmse"])
        writer.writerows(comparison_rows)
    P="pinn_data_correction_fno_ttt"
    init_by_id={r["material_id"]:float(r["D_common_pct"]) for r in material_groups["fno_ttt_s0"][:8]}
    improved=sum(float(r["D_common_pct"])<init_by_id[r["material_id"]] for r in material_groups[P])
    body=[title("What changes with one current?"),p("Each material received one random constant current density. The same draw was used by every method, from supervised training to per-material physics refinement."),
          sub("Property recovery on the original concentration window"),
          p(f"Across 300 test materials, FNO + TTT had <b>{fmt(A['one_current_original_window'])}%</b> median diffusivity error with a single random current, compared with <b>{fmt(A['three_current_original_window'])}%</b> with three. On its own observed interval, the same single-current estimator scored <b>{fmt(A['one_current_observed_window'])}%</b>. All three figures average seed-specific medians."),
          p(f"Starting from FNO + TTT, the differentiable solver reached a median common-window D error of <b>{val('lbfgs_fno_ttt','D_common_pct')}%</b> across 300 materials. On the eight-case PINN pilot, the data-first correction moved the median from {fmt(pilot_fno('D_common_pct'))}% to <b>{val(P,'D_common_pct')}%</b>. It improved {improved} of the eight individual cases."),
          Image(str(data/"diffusivity_comparison.png"),width=504,height=173.6), Spacer(1,8),
          sub("Paired experimental design"),
          p("The main generator assigned each retained material one of its original three current densities with equal probability. The numerical seed combines 236 with the original material ID. The three candidate bands are 0–0.625, 1.25–1.875 and 2.5–3.125 in dimensionless current. The 300 test assignments comprise 87 low, 112 middle and 101 high currents."),
          p("The dataset retains the original 10,336 training materials, 593 validation materials, 1,207 in-distribution test materials and 2,859 parameter-tail materials. Parameter fitting used the selected trajectory. Methods that use velocity reconstructed it from that trajectory. Recomputing the valid time prefix extended observations for 121/300 test cases. The regenerated and saved concentration observations agree exactly wherever both are available."),
          sub("Two ways to score the recovered curves"),
          p("The observed window is the concentration interval reached by the single input. The common window is the original three-current interval for the same material. It is identical across the methods being compared and includes extrapolation for many single-current fits. The two windows differ for 199/300 cases."),
          p("For predictive validation, each fitted set of eight coefficients was replayed in the same forward solver at a fourth, fixed dimensionless current of 1.0. This current was reserved for scoring. Concentration errors are RMSE divided by mean true concentration, expressed as a percentage."),
          PageBreak(),
          title("Pretrained estimators and TTT"),p("MLP and FNO checkpoints were trained from scratch on single-current inputs. Each architecture used 100 epochs and three seeds. TTT reset weights per material and optimized the measured trajectory’s PDE residual for 150 AdamW steps."),
          table(["Method","1 current D, observed (%)","1 current D, common (%)","3 currents D, common (%)","1 current t+, common RMSE"],[
              [m.replace("_supervised","").replace("_ttt"," + TTT").upper(),fmt(v["D_relL2_pct"]["one_current_observed_window"]),
               fmt(v["D_relL2_pct"]["one_current_original_window"]),fmt(v["D_relL2_pct"]["three_current_original_window"]),
               fmt(v["tplus_rmse"]["one_current_original_window"],4)] for m,v in comparisons.items()],[105,101,101,101,108]),
          Spacer(1,11),sub("How the random current changes the result"),
          table(["Method","Low current, D (%)","Middle current, D (%)","High current, D (%)"],[
              [arch+" + TTT",*[fmt(comparisons[arch.lower()+"_ttt"]["D_relL2_pct"]["one_current_original_window_by_rank"][str(k)]) for k in range(3)]] for arch in ("MLP","FNO")],[121,126,134,135]),
          small("These are common-window medians within each current stratum, averaged across the three seeds. Median single-current concentration spans are 0.193, 0.960 and 1.402 in the low, middle and high groups, respectively."),
          sub("Material uncertainty and held-out prediction"),
          p(f"The paired material-bootstrap 95% interval for the change in FNO + TTT common-window D error is {fmt(A['one_minus_three_original_window_95pct_material_bootstrap'][0])} to {fmt(A['one_minus_three_original_window_95pct_material_bootstrap'][1])} percentage points. It quantifies material sampling under this recorded current draw."),
          p("The input layer has fewer channels with one experiment. The width-40 MLP has 23,368 parameters versus 65,048 in its three-current version. The FNO keeps the same Fourier modes, width and depth. The comparison therefore also includes the change in MLP input-layer capacity."),
          table(["Seed-0 estimate","D, common (%)","t+, common RMSE","Held-out c RMSE (%)"],[
              [s,val(k,"D_common_pct"),val(k,"t_common_rmse",4),val(k,"c_unseen1_pct",3)] for s,k in
              [("MLP","mlp_supervised_s0"),("MLP + TTT","mlp_ttt_s0"),("FNO","fno_supervised_s0"),("FNO + TTT","fno_ttt_s0")]],[130,115,132,139]),PageBreak(),
          title("Differentiable and gradient-free inversion"),p("The same eight bounded coefficients were fit to the one observed current. L-BFGS-B used adjoint gradients. Nelder–Mead and CMA-ES used 2,000-evaluation budgets on 300 materials. Bayesian optimization used 150 evaluations on 30 materials."),
          table(["Method and start","N","1 current D, observed (%)","1 current D, common (%)","3 currents D, common (%)","1 current t+, common","c target hits¹"],[
               [m+" / "+s,lookup[k]["n"],val(k,"D_observed_pct"),val(k,"D_common_pct"),historical_d(k),val(k,"t_common_rmse",4),sv[(m0,s)]["reached"]]
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
                 p(f"With FNO + TTT initialization, {sum(e>1 for e in solver_groups[0])} of the 87 low-current cases exceeded 1% common-window D error after L-BFGS-B. Their median was {fmt(median(solver_groups[0]))}%. Only {sum(e>1 for group in solver_groups[1:] for e in group)} of the other 213 cases exceeded 1%. The overall 95th percentile was {val('lbfgs_fno_ttt','p95_D_common_pct')}%."),PageBreak(),
                 title("PINN refinement on eight materials"),p("All PINN variants use the selected current of the same eight pilot materials. The learned start is seed-0 FNO + TTT. A center start sets every normalized coefficient to 0.5."),
                 table(["Variant","Start","1 current D, observed (%)","1 current D, common (%)","3 currents D, common (%)","1 current t+, common","Held-out c (%)"],[
                     ["FNO + TTT","Initial",fmt(pilot_fno("D_observed_pct")),fmt(pilot_fno("D_common_pct")),historical_d("fno_ttt_pilot"),fmt(pilot_fno("t_common_rmse"),4),fmt(pilot_fno("c_unseen1_pct"),3)]]
                     +[[names[m],"Learned" if s=="fno_ttt" else "Center",val("pinn_"+m+"_"+s,"D_observed_pct"),val("pinn_"+m+"_"+s,"D_common_pct"),historical_d("pinn_"+m+"_"+s),val("pinn_"+m+"_"+s,"t_common_rmse",4),val("pinn_"+m+"_"+s,"c_unseen1_pct",3)] for m,s in stages if not m.startswith("staged_")],[129,57,67,67,66,65,65]),
                 small("Observed and common D errors are relative L2 percentages. The three-current comparison uses saved fits for the same eight materials. Held-out concentration error uses new forward predictions from the fitted coefficients."),
                 sub("How well does fitting concentration recover diffusivity?"),
                 p(f"With the learned start, fitting the joint c and v PINN changed observed-window D error from {fmt(pilot_fno('D_observed_pct'))}% to {val('pinn_standard_cv_fno_ttt','D_observed_pct')}%. Its median error on the fitted concentration field was 0.274%. Data-first joint fitting followed by the small correction networks produced {val(P,'D_observed_pct')}% observed-window D error and {val(P,'D_common_pct')}% common-window D error. The fitted field used by a PINN can differ from the forward-solver field implied by its inferred properties."),
                 PageBreak(),title("What staging and grid resolution change"),
                 table(["Staged variant","Start","D, observed (%)","D, common (%)","Held-out c (%)"],[
                     [names[m],"Learned" if s=="fno_ttt" else "Center",val("pinn_"+m+"_"+s,"D_observed_pct"),val("pinn_"+m+"_"+s,"D_common_pct"),val("pinn_"+m+"_"+s,"c_unseen1_pct",3)] for m,s in stages if m.startswith("staged_")],[156,66,97,97,100]),
                 small("Staging fits the fields before optimizing the transport properties. The weight refers to the physics terms. Native-grid data-first uses 40 cells and up to 104 times at the selected current. The other measured-data fits use 20 bins and up to 13 times. The native-grid pilot thus also tests a denser measurement."),
                 sub("Prediction at an independently held-out current"),
                 p("All final coefficient estimates were passed through the forward solver at dimensionless current 1.0. The table shows median error on that new trajectory and the 95th percentile of the diffusivity error on the common window. The last column counts predictions that remained inside the simulated concentration domain throughout the true held-out trajectory."),
                 table(["Method and start","N","Held-out c RMSE (%)","95th percentile D, common (%)","Admissible held-out replays"],[
                     [name,lookup[m]["n"],val(m,"c_unseen1_pct",4),val(m,"p95_D_common_pct"),lookup[m]["n_unseen_admissible"]]
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
    old=load(data/"legacy_three_current_matched.json")
    legacy=[]
    for d in ("clean","xray_noisy"):
        a=load(ROOT/f"legacy/cnn_{d}/summary.json"); b=load(ROOT/f"legacy/pinn_{d}/one_current_result.json")
        bscore=b["full_curve_scores"]
        legacy.append([d,fmt(a["combined_normalized_rmse"],4),fmt(old[d]["combined_normalized_rmse"],4),
                       b["status"].capitalize(),fmt(bscore["D_curve_rmse_01"],4),fmt(bscore["t_plus_curve_rmse_01"],4)])
    body.extend([table(["Domain","CNN 1 current¹","CNN 3 currents¹","1-current PINN","PINN D RMSE","PINN t+ RMSE"],legacy,[95,83,85,95,77,81]),
                 small("¹ Combined normalized full-curve RMSE on 389 test materials, equally weighted. The three-current control averages squared errors across usable triplets within material. Both new CNNs used one seeded random usable current per material and 50 epochs. The paired PINN used material 2, current index 3 (dimensionless 2.2285), and the original five acceptance gates. Scores for rejected fits describe the last candidate."),
                 sub("Reading the files"),
                 p("The random experiment directory contains the selected current rank, material IDs, trained checkpoints and per-fit outputs. report_data/materials.csv records common-window, observed-window and held-out scores for each main estimate. solver_runs/results/summary.csv records coverage and cost. Separate per-case PINN JSON files retain field losses and timing."),
                 small("The main results use noiseless synthetic data and one randomized assignment. The two independent campaigns have different grids, data generators, and scoring scales. Absolute numbers across those campaigns have different interpretations.")])
    DEST.parent.mkdir(exist_ok=True,parents=True)
    doc=SimpleDocTemplate(str(DEST),pagesize=(8.5*inch,11*inch),rightMargin=47,leftMargin=47,topMargin=57,bottomMargin=40,
                          title="DiffIon | One random current per material")
    doc.build(body,onFirstPage=page_number,onLaterPages=page_number)
    print(DEST)


if __name__=="__main__":
    main()
