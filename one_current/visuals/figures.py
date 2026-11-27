"""Render the paired benchmark figures from saved scores and fitted coefficients."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np


REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "one_current" / "random"
OUT = RUN / "visuals"
THREE, ONE = "#006E75", "#BF532B"
INK, MUTED, LINE = "#182F37", "#49636A", "#DDE6E5"
BAND = ["Low", "Middle", "High"]
LEARNED = ["mlp_supervised", "mlp_ttt", "fno_supervised", "fno_ttt"]
FREE = [f"{opt}_{start}" for opt in ("NM","CMAES","BO") for start in ("center","fno_ttt")]
DIRECT = ["lbfgs_"+s for s in ("center","mlp","mlp_ttt","fno","fno_ttt")]
VARIANTS = {"standard_c":"Joint PINN, c", "standard_cv":"Joint PINN, c + v",
            "data_first":"Data-first joint PINN", "data_correction":"Data-first + correction",
            "solver_correction":"Solver-seeded, w = 0.01", "solver_correction_weight1":"Solver-seeded, w = 1",
            "fine":"Native-grid data-first", "staged_c_w1":"Staged c, w = 1",
            "staged_c_w100":"Staged c, w = 100", "staged_cv_w1":"Staged c + v, w = 1",
            "staged_cv_w100":"Staged c + v, w = 100"}
PILOT = "pilot_start"
NAMES = dict(zip(LEARNED, ["MLP", "MLP + TTT", "FNO", "FNO + TTT"]))
NAMES[PILOT] = "FNO + TTT starting estimate"
for prefix, family in (("lbfgs","L-BFGS-B"),("NM","Nelder–Mead"),("CMAES","CMA-ES"),("BO","Bayesian optimization")):
    for suffix, start in (("center","center"),("fno","FNO"),("mlp","MLP"),("mlp_ttt","MLP + TTT"),("fno_ttt","FNO + TTT")):
        NAMES[f"{prefix}_{suffix}"] = f"{family} / {start}"
for variant, label in VARIANTS.items():
    for start in ("fno_ttt","center"):
        NAMES[f"pinn_{variant}_{start}"] = label
WARM = ["pinn_"+v+"_fno_ttt" for v in VARIANTS]
CENTER = ["pinn_"+v+"_center" for v in VARIANTS if v not in ("data_correction","solver_correction","solver_correction_weight1")]
D_KEYS = ["D_three_observed_pct","D_one_observed_pct"]
T_KEYS = ["t_three_observed_rmse","t_one_observed_rmse"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"text.color":INK,
                     "axes.labelcolor":MUTED,"xtick.color":MUTED,"ytick.color":MUTED,
                     "pdf.fonttype":42,"axes.titleweight":"bold"})


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def load_cases():
    groups = defaultdict(list)
    for row in rows(OUT/"paired_cases.csv"):
        groups[row["method"]].append(row)
    assert set(groups)==set(LEARNED+FREE+DIRECT+WARM+CENTER)
    pilot = {r["material_id"] for r in groups[WARM[0]]}
    groups[PILOT] = [r for r in groups["fno_ttt"] if r["seed"]=="0" and r["material_id"] in pilot]
    assert len(groups[PILOT])==8
    return groups


def statistic(group, key, quantile=.5):
    seeds = defaultdict(list)
    for r in group:
        seeds[r["seed"]].append(float(r[key]))
    return np.mean([np.quantile(values,quantile) for values in seeds.values()])


def number(v, metric):
    if v != 0 and v < .001:
        mantissa, exponent = f"{v:.1e}".split("e")
        return f"{mantissa}e{int(exponent)}"
    if metric=="D" and 0 < v < .01:
        return f"{v:.2g}"
    return f"{v:.2f}" if metric=="D" else f"{v:.3f}"


def page(title, subtitle):
    fig = plt.figure(figsize=(11.7,8.3),facecolor="white")
    fig.text(.045,.944,title,fontsize=22,fontweight="bold",va="top")
    fig.text(.046,.873,subtitle,fontsize=11,va="top",color=MUTED,linespacing=1.4)
    return fig


def key(fig, y=.801):
    artists=[Line2D([],[],marker="o",mfc=THREE,mec=THREE,ls="",ms=7,label="Three currents"),
             Line2D([],[],marker="D",mfc="white",mec=ONE,mew=1.7,ls="",ms=6,label="One current")]
    fig.legend(handles=artists,loc="center left",bbox_to_anchor=(.042,y),ncol=2,
               frameon=False,handletextpad=.25,columnspacing=1.6,borderaxespad=0)


def note(fig, text):
    fig.text(.046,.055,text,fontsize=9,color=MUTED,va="bottom",linespacing=1.4)


def pair_page(groups, methods, title, subtitle, footer, *, small=False, logarithmic=False):
    fig=page(title,subtitle)
    key(fig)
    y=np.arange(len(methods))
    low_y,high_y=(.41,.73) if small else (.17,.73)
    row_ax=fig.add_axes([.045,low_y,.27,high_y-low_y])
    row_ax.set(xlim=(0,1),ylim=(len(y)-.5,-.5));row_ax.axis("off")
    for i,m in enumerate(methods):
        row_ax.text(0,i,NAMES[m],va="center",fontsize=9.1)
        n=len({r["material_id"] for r in groups[m]})
        row_ax.text(.97,i,str(n),va="center",ha="right",fontsize=8.7,color=MUTED)
    fig.text(.045,.752,"Method / initialization",fontsize=9,fontweight="bold")
    fig.text(.307,.752,"n",fontsize=9,fontweight="bold",ha="right")
    for x,cols,metric,title in ((.34,D_KEYS,"D","Diffusivity error (%)"),(.666,T_KEYS,"t","Transference-number RMSE")):
        ax=fig.add_axes([x,low_y,.172,high_y-low_y])
        vals=np.array([[statistic(groups[m],k) for k in cols] for m in methods])
        ax.set_ylim(len(y)-.5,-.5)
        ax.hlines(y,vals[:,0],vals[:,1],color="#A6B7B7",lw=1.65)
        ax.scatter(vals[:,0],y,marker="o",s=36,c=THREE,zorder=3)
        ax.scatter(vals[:,1],y,marker="D",s=32,facecolors="none",edgecolors=ONE,lw=1.6,zorder=4)
        if logarithmic:
            ax.set_xscale("log")
            ax.set_xlim(vals.min()*.77,vals.max()*1.3)
            from matplotlib.ticker import NullFormatter, FixedLocator, FuncFormatter
            points=np.geomspace(vals.min(),vals.max(),3)
            ax.xaxis.set_major_locator(FixedLocator(points))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:number(v,metric)))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_xlim(-vals.max()*.04,vals.max()*1.07)
            ax.locator_params(axis="x",nbins=4)
        ax.set_yticks([]);ax.grid(axis="x",color=LINE);ax.set_axisbelow(True)
        ax.spines[["left","right","top"]].set_visible(False);ax.spines["bottom"].set_color(LINE)
        ax.tick_params(axis="x",labelsize=7.5,length=0,pad=5)
        fig.text(x,.752,title,fontsize=9,fontweight="bold")
        label_ax=fig.add_axes([x+.177,low_y,.123,high_y-low_y])
        label_ax.set(xlim=(0,1),ylim=(len(y)-.5,-.5));label_ax.axis("off")
        for i,(a,b) in enumerate(vals):
            label_ax.text(.43,i,number(a,metric),ha="right",va="center",fontsize=8.5,color=THREE)
            label_ax.text(.98,i,number(b,metric),ha="right",va="center",fontsize=8.5,color=ONE)
        fig.text(x+.230,.752,"3",ha="right",color=THREE,weight="bold")
        fig.text(x+.297,.752,"1",ha="right",color=ONE,weight="bold")
    note(fig,footer)
    return fig


def heat_page(groups, methods, title, subtitle):
    fig=page(title,subtitle)
    palette=colors.LinearSegmentedColormap.from_list("change",["#9FCECB","#F5F7F4","#EAAC8C"])
    axes=[fig.add_axes([.32,.16,.305,.59]),fig.add_axes([.646,.16,.305,.59])]
    ax_left=fig.add_axes([.045,.16,.268,.59])
    ax_left.set(xlim=(0,1),ylim=(len(methods)-.5,-.5));ax_left.axis("off")
    for i,m in enumerate(methods):
        ax_left.text(0,i,NAMES[m],va="center",fontsize=8.9)
    for ax,keys,metric,heading in zip(axes,[D_KEYS,T_KEYS],["D","t"],["Diffusivity error (%)","Transference-number RMSE"]):
        a=np.array([[[statistic([r for r in groups[m] if int(r["rank"])==j],key) for key in keys]
                      for j in range(3)] for m in methods])
        effect=np.log2(a[:,:,1]/a[:,:,0])
        ax.imshow(effect,cmap=palette,vmin=-2,vmax=2,aspect="auto",interpolation="none")
        for i in range(len(methods)):
            for j in range(3):
                x,z=a[i,j]
                ax.text(j,i,number(x,metric)+" → "+number(z,metric),ha="center",va="center",fontsize=7.35)
        ax.set_xticks(range(3),BAND);ax.xaxis.tick_top()
        ax.tick_params(length=0,pad=4);ax.set_yticks([])
        ax.set_xticks(np.arange(-.5,3), minor=True);ax.set_yticks(np.arange(-.5,len(methods)),minor=True)
        ax.tick_params(which="minor",length=0);ax.grid(which="minor",color="white",lw=1.1)
        ax.spines[:].set_visible(False)
        ax.set_title(heading,fontsize=11,pad=25)
    ax_left.text(0,1.025,"Method / initialization",transform=ax_left.transAxes,fontsize=9,fontweight="bold")
    note(fig,"Each cell reads three-current median → one-current median on the same materials. Each fit is scored over its own observed concentration interval.\n"
         "Orange means higher error with one current. Teal means lower error. Shading saturates at a fourfold difference.")
    return fig


def current_focus(groups):
    fig=page("How the assigned current changes recovery",
             "Each panel groups the same materials by the single current retained. Each protocol is scored across the concentrations it observed.")
    key(fig,.795)
    methods=["fno_ttt","NM_fno_ttt","CMAES_fno_ttt","lbfgs_fno_ttt"]
    for i,method in enumerate(methods):
        ax=fig.add_axes([.072+.24*i,.30,.183,.36])
        vals=np.asarray([[statistic([r for r in groups[method] if int(r["rank"])==j],k) for k in D_KEYS] for j in range(3)])
        # Connect paired protocols vertically at the same assigned-current band.
        ax.vlines(range(3),vals[:,0],vals[:,1],color="#A6B7B7",lw=2)
        ax.scatter(range(3),vals[:,0],c=THREE,s=42,marker="o",zorder=3)
        ax.scatter(range(3),vals[:,1],edgecolors=ONE,facecolors="none",lw=1.8,s=40,marker="D",zorder=4)
        ax.set_title(["FNO + TTT","Nelder–Mead","CMA-ES","L-BFGS-B"][i],fontsize=13,loc="left",pad=21)
        ax.set_xticks(range(3),BAND);ax.set_xlim(-.45,2.45)
        if method.startswith("lbfgs"):
            ax.set_yscale("log");ax.set_ylim(1e-9,1);ax.set_yticks([1e-8,1e-6,1e-4,1e-2,1])
            ax.text(0,1.012,"logarithmic scale",transform=ax.transAxes,fontsize=8,color=MUTED)
        else:
            peak=max(8,2*np.ceil(np.max(vals)/2))
            ax.set_ylim(0,peak*1.08);ax.set_yticks(np.arange(0,peak+1,2))
        if i==0: ax.set_ylabel("Median diffusivity error (%)")
        ax.spines[["right","top"]].set_visible(False);ax.grid(axis="y",color=LINE)
        ax.tick_params(length=0,pad=4,labelsize=9)
        for j in range(3):
            value=vals[j,1]
            offset = -11 if value < vals[j,0] else 7
            ax.annotate(number(value,"D"),(j,value),xytext=(0,offset),textcoords="offset points",
                        ha="center",color=ONE,fontsize=8,weight="bold")
    unique={r["material_id"]:r for r in groups["fno_ttt"]}
    for j in range(3):
        data=[r for r in unique.values() if int(r["rank"])==j]
        w1,w3=[np.median([float(r[p+"_window_max"])-float(r[p+"_window_min"]) for r in data]) for p in ("one","three")]
        fig.text(.072+.312*j,.176,f"{BAND[j]} current  ·  {len(data)} materials",weight="bold",fontsize=11)
        fig.text(.072+.312*j,.139,f"Median c span:  {w1:.3f} (one)  /  {w3:.3f} (three)",fontsize=9.5,color=MUTED)
    note(fig,"The span is the maximum minus minimum observed concentration. The numerical fits start from seed-0 FNO + TTT.\n"
         "Learned estimates average the three seed-specific medians.")
    return fig


def property_page(groups):
    fig=page("Recovered transport curves on two pilot materials",
             "The first low- and high-current pilot materials are shown. Each fitted curve ends at the bounds of its own observed interval.")
    pilot=sorted(groups[PILOT],key=lambda r:int(r["material_id"]))
    chosen=[next(i for i,r in enumerate(pilot) if int(r["rank"])==j) for j in (0,2)]
    with np.load(REPO/"data/train.npz") as f:
        truth=dict(zip(f["index"],f["theta"]))
    def properties(theta,c):
        u=np.clip(c/2,0,1);z=np.stack([(1-u)**3,3*u*(1-u)**2,3*u*u*(1-u),u**3],axis=-1)
        return (.1*np.exp(np.log(20)*(z@theta[:4])), -.25+1.25*(z@theta[4:]))
    single_fit=RUN/"solver_runs/results/lbfgs/fno_ttt"
    triple_fit=REPO/"results/lbfgs/A_fno_ttt"
    configs=[("FNO + TTT",single_fit,triple_fit,"theta0"),
             ("Data-first PINN\n+ correction",RUN/"pinn_runs/results/data_correction/fno_ttt",
              REPO.parent/"diffion-junbo/results/pinn_prefit/ms2_wd1e-2/A_fno_ttt/cv","theta"),
             ("L-BFGS-B",single_fit,triple_fit,"theta")]
    for h,k in enumerate(chosen):
        r=pilot[k];mid=int(r["material_id"])
        grids=[np.linspace(float(r[p+"_window_min"]),float(r[p+"_window_max"]),241) for p in ("one","three")]
        c=np.linspace(min(grid[0] for grid in grids),max(grid[-1] for grid in grids),301)
        fig.text(.328+.42*h,.792,f"Material {mid}  ·  {BAND[int(r['rank'])].lower()} current {float(r['iota']):.3f}",ha="center",fontsize=12,weight="bold")
        for j,(name,p1,p3,field) in enumerate(configs):
            records=[json.loads((p/f"m{k:03}.json").read_text()) for p in (p1,p3)]
            assert all(int(v["index"])==mid for v in records)
            predictions=[properties(np.asarray(v[field]),grid) for v,grid in zip(records,grids)]
            target=properties(truth[mid],c)
            if h==0: fig.text(.046,.654-.184*j,name,fontsize=10.3,va="center",weight="bold")
            for v in range(2):
                ax=fig.add_axes([.175+h*.423+v*.202,.603-j*.184,.165,.128])
                ax.axvspan(grids[0][0],grids[0][-1],color="#F6E8CE")
                ax.plot(c,target[v],color=INK,lw=2.9,alpha=.83)
                ax.plot(grids[1],predictions[1][v],c=THREE,lw=1.8,ls=(0,(5,3)))
                ax.plot(grids[0],predictions[0][v],c=ONE,lw=1.9,ls=(0,(1.2,1.7)))
                ax.spines[["top","right"]].set_visible(False)
                ax.tick_params(labelsize=7,length=2,pad=2);ax.locator_params(nbins=3)
                if j==0: ax.set_title(["Diffusivity, D","Transference, t+"][v],fontsize=9)
                if j==2: ax.set_xlabel("Concentration",fontsize=8,labelpad=1)
    styles=[Line2D([],[],c=INK,lw=2.8,label="True property"),Line2D([],[],c=THREE,lw=2,ls="--",label="Three currents"),
            Line2D([],[],c=ONE,lw=2,ls=":",label="One current"),matplotlib.patches.Patch(color="#F6E8CE",label="Range observed at one current")]
    fig.legend(handles=styles,frameon=False,ncol=4,loc="lower left",bbox_to_anchor=(.04,.110))
    note(fig,"Curves use the saved eight transport coefficients. Overlapping traces indicate agreement at the plotting scale.\n"
         "The concentration and diffusivity axes use the main simulator’s dimensionless units.")
    return fig


def main():
    groups=load_cases()
    OUT.mkdir(exist_ok=True)
    static="Each fit is scored from the minimum to the maximum concentration observed under its own current protocol. Lower error is better."
    first=LEARNED+["lbfgs_fno_ttt","NM_fno_ttt","CMAES_fno_ttt","BO_fno_ttt",PILOT,
                   "pinn_standard_cv_fno_ttt","pinn_data_correction_fno_ttt","pinn_solver_correction_fno_ttt"]
    figs=[pair_page(groups,first,"One versus three current densities",static,
        "Numerical refinements and PINNs shown here start from seed-0 FNO + TTT. The pilot starting estimate uses the same eight materials as the PINNs.\n"
        "The four learned rows average medians from three training seeds. Other rows show the median across materials; n is the material count."),
        current_focus(groups),
        pair_page(groups,LEARNED+FREE,"Pretraining, TTT and gradient-free inversion",static,
        "Center sets all eight normalized coefficients to 0.5. FNO + TTT uses the learned starting coefficients for that protocol.\n"
        "Nelder–Mead and CMA-ES use up to 2,000 evaluations per fit. Bayesian optimization uses up to 150 on a 30-material subset."),
        pair_page(groups,DIRECT,"Differentiable solver by initialization","All five L-BFGS-B starts fit the same 300 noiseless materials. Both axes are logarithmic.",
        "The same transport model generated the observations and evaluated the inverse fit. The low-current group accounts for most of the upper tail.",small=True,logarithmic=True),
        pair_page(groups,[PILOT]+WARM,"PINN refinements from FNO + TTT","The first row gives the starting error on these same eight materials for each protocol.",
        "c denotes concentration; v is velocity reconstructed from c. Native-grid fits use 40 cells and up to 104 times, versus 20 bins and 13 times.\n"
        "Staged w is the physics loss weight. Solver-seeded w is the data loss weight on the correction."),
        pair_page(groups,CENTER,"PINN refinements from the center start","Each fit starts with all normalized transport coefficients set to 0.5. The same eight pilot materials are paired.",
        "The data-first and staged methods fit fields before updating material properties. The native-grid row uses denser observations at the same currents."),
        heat_page(groups,LEARNED+FREE+DIRECT,"Current band by estimator and numerical method",
        "Materials are grouped by the single current drawn. Error and concentration coverage are specific to each protocol."),
        heat_page(groups,[PILOT]+WARM,"Current band by learned-start PINN",
        "Of eight pilot materials, three drew a low current, one drew a middle current and four drew a high current. The middle column is one case."),
        heat_page(groups,CENTER,"Current band by center-start PINN",
        "The low, middle and high columns contain three, one and four pilot materials. Each method was fit on the same assignment."),
        property_page(groups)]
    direct=figs[3]
    ref=groups["lbfgs_fno_ttt"]
    for j,rank in enumerate(range(3)):
        part=[r for r in ref if int(r["rank"])==rank]
        direct.text(.052+j*.312,.320,f"{BAND[rank]} current  ·  {len(part)} materials",weight="bold",fontsize=12)
        direct.text(.052+j*.312,.266,f"One-current median   {number(statistic(part,D_KEYS[1]),'D')}%",fontsize=11,color=ONE)
        direct.text(.052+j*.312,.227,f"Three-current median   {number(statistic(part,D_KEYS[0]),'D')}%",fontsize=10,color=THREE)
        direct.text(.052+j*.312,.176,f"One-current 95th percentile   {number(statistic(part,D_KEYS[1],.95),'D')}%",fontsize=9)
    with PdfPages(OUT/"main_figures.pdf") as pdf:
        for i,fig in enumerate(figs):
            pdf.savefig(fig)
            if i in (0,1):
                fig.savefig(OUT/("methods_overview.png" if i==0 else "current_bands.png"),dpi=157)
            plt.close(fig)


if __name__=="__main__":
    main()
