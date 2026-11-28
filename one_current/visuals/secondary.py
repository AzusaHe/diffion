"""Plot measured input fields and both secondary campaigns from saved outputs.

From diffion-ms, run python -m one_current.visuals.secondary.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


REPO = Path(__file__).resolve().parents[2]
RANDOM = REPO / "one_current/random"
OBSERVED = RANDOM / "report_data/legacy_observed_ranges.json"
OUT = RANDOM / "visuals/secondary.pdf"
ONE, THREE = "#BF532B", "#006E75"
INK, MUTED, GRID = "#1a343b", "#4b6169", "#dce6e7"
DOMAINS = ("clean", "xray_noisy")
DOMAIN_NAMES = ("Clean concentration", "X-ray-noisy concentration")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def page(title, subtitle):
    fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
    fig.text(.062,.941,title,fontsize=23,fontweight="bold",color=INK)
    fig.text(.063,.893,subtitle,fontsize=11.4,color=MUTED,va="top")
    return fig


def footer(fig, text):
    fig.add_artist(Line2D([.062,.96],[.105,.105],transform=fig.transFigure,lw=.8,color=GRID))
    fig.text(.063,.082,text,fontsize=9.3,color=MUTED,va="top",linespacing=1.5)


def style(ax):
    ax.spines[["top","right"]].set_visible(False)
    for side in ("left","bottom"):
        ax.spines[side].set_color("#9baeb1")
    ax.tick_params(colors=MUTED, length=3.0)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def input_figure():
    data = REPO/"data"
    row = np.flatnonzero(np.load(data/"train_split.npy")=="id_test")[0]
    with np.load(data/"train.npz") as ref, np.load(RANDOM/"data/train.npz") as single:
        material = int(ref["index"][row])
        assert single["index"][row] == material
        original, one = ref["y"][row], single["y"][row,0]
        iotas, selected = ref["iotas"][row], single["iotas"][row,0]
        n_three, n_one = int(ref["K"][row]), int(single["K"][row])
        range_one, range_three = single["window"][row], ref["window"][row]
    rank = np.flatnonzero(iotas == selected).item()
    np.testing.assert_array_equal(original[rank,:n_three],one[:n_three])
    assert material == 21 and rank == 0
    fig = page(f"Concentration fields for material {material}",
               "The first PINN pilot material drew the low current. The three-current experiment used all three fields.")
    fields = [one, *original]
    cmax = np.nanmax(np.abs(np.asarray(fields)-1))
    norm = TwoSlopeNorm(vmin=1-cmax, vcenter=1, vmax=1+cmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eaeeef")
    locations = [(.10,.56),(.52,.56),(.10,.246),(.52,.246)]
    times = (240+960*np.arange(13))/3600
    xedges = np.linspace(0,1,21)
    # Color chip rows are centered at their observation times.
    tedges = np.r_[times-.5*np.diff(times)[0],times[-1]+.5*np.diff(times)[0]]
    for i, ((left,bottom), field) in enumerate(zip(locations,fields)):
        ax = fig.add_axes([left,bottom,.299,.213])
        img = ax.pcolormesh(xedges,tedges,np.ma.masked_invalid(field),norm=norm,cmap=cmap,rasterized=True)
        io = selected if i==0 else iotas[i-1]
        band = "low" if i==0 else ("low","middle","high")[i-1]
        label = "One current" if i==0 else "Three currents"
        ax.set_title(f"{label}  ·  {band},  ι = {io:.3f}",loc="left",fontsize=11.6,pad=7,
                     fontweight="bold",color=ONE if i==0 else THREE)
        ax.set_xlim(0,1)
        ax.set_ylim(0,tedges[-1])
        ax.set_xticks([0,.25,.5,.75,1],["0",".25",".5",".75","1"])
        ax.set_yticks([0,1,2,3])
        if i%2==0:
            ax.set_ylabel("Elapsed time (h)")
        if i>=2:
            ax.set_xlabel("Normalized position")
        style(ax)
        if np.isnan(field).any():
            ax.text(.5,.89,"Outside shared time window",color=MUTED,fontsize=8,
                    transform=ax.transAxes,ha="center",va="center")
    cbax = fig.add_axes([.866,.302,.018,.437])
    cb = fig.colorbar(img,cax=cbax)
    cb.set_label("Dimensionless concentration",fontsize=10,labelpad=7)
    cb.outline.set_visible(False)
    fig.text(.10,.155,"Single-current range",color=ONE,fontsize=10.2,fontweight="bold")
    fig.text(.280,.155,f"{range_one[0]:.3f} to {range_one[1]:.3f}",color=INK,fontsize=11)
    fig.text(.52,.155,"Three-current range",color=THREE,fontsize=10.2,fontweight="bold")
    fig.text(.70,.155,f"{range_three[0]:.3f} to {range_three[1]:.3f}",color=INK,fontsize=11)
    footer(fig, f"All panels use the same concentration scale and 20 spatial bins. The single current remains valid for {n_one} frames.\n"
                f"The three-current experiment keeps the {n_three} frames valid across the full triplet. The overlapping low-current measurements agree exactly.")
    return fig


def compact_figure():
    one = RANDOM/"solver_runs/results/compact"
    three = REPO/"results/numerical baselines"
    fig = page("Which optimizer starts reach the target?",
               "Compact control on one reference material. Each cell shows how many objective evaluations that run used.")
    names = ("Nelder–Mead","CMA-ES","Bayesian optimization")
    methods = ("NM","CMAES","BO")
    success,miss = "#c5e3d6","#f6d9c9"
    for j, (folder,label) in enumerate((("1","1%"),("0.1","0.1%"),("0.01","0.01%"))):
        ax = fig.add_axes([.19+.262*j,.235,.244,.52])
        hits, evals = np.zeros((6,5),dtype=bool), np.zeros((6,5),dtype=int)
        for k,method in enumerate(methods):
            for start in range(5):
                pair = [read(root/f"target_{folder}pct"/f"{method}_start{start}.json") for root in (one,three)]
                np.testing.assert_allclose(pair[0]["z0"],pair[1]["z0"])
                assert pair[0]["iotas"] == [.6]
                assert pair[0]["budget_evals"] == pair[1]["budget_evals"]
                for p,r in enumerate(pair):
                    assert r["start"]==start and r["method"]==method
                    hits[2*k+p,start], evals[2*k+p,start] = r["reached_target"], r["total_evals"]
        ax.imshow(hits,cmap=ListedColormap([miss,success]),vmin=0,vmax=1,aspect="auto")
        for row in range(6):
            for col in range(5):
                ax.text(col,row,str(evals[row,col]),ha="center",va="center",fontsize=9,color=INK)
        for y in (1.5,3.5):
            ax.axhline(y,color="white",lw=5)
        ax.set_xticks(np.arange(5),labels=[str(i) for i in range(5)])
        ax.set_xlabel("Optimizer start",labelpad=7)
        if j==0:
            labels = [f"{name}\n1 current" if p==0 else "3 currents" for name in names for p in range(2)]
            ax.set_yticks(np.arange(6),labels)
            for k,item in enumerate(ax.get_yticklabels()):
                item.set_color(ONE if k%2==0 else THREE)
                item.set_fontsize(9)
        else:
            ax.set_yticks([])
        ax.set_title(f"{label} concentration target",loc="center",fontsize=12.2,color=INK,pad=12,fontweight="bold")
        for g in (ax.xaxis,ax.yaxis):
            g.set_minor_locator(plt.FixedLocator(np.arange(-.5,5.5 if g is ax.xaxis else 6.5,1)))
        ax.grid(which="minor",color="white",linewidth=1.1)
        ax.tick_params(which="both",length=0)
        ax.spines[:].set_visible(False)
    fig.legend(handles=[Patch(color=success,label="Target reached"),Patch(color=miss,label="Budget or time limit reached")],
               loc="lower center",bbox_to_anchor=(.55,.12),ncol=2,frameon=False,fontsize=10)
    footer(fig,"The random one-current assignment used ι = 0.6. The three-current control used ι = 0.3, 0.6 and 0.9. Both used the same five starts.\n"
               "Each target is RMSE as a percentage of mean concentration. The tightest target also imposed a 30-minute time limit.")
    return fig


def cnn_errors(domain):
    case=read(OBSERVED)[domain]["cnn"]
    return (np.asarray(case["one_combined_by_material"]),np.asarray(case["three_combined_by_material"]),
            case["one"],case["three"])


def cnn_figure():
    fig=page("CNN recovery on clean and noisy inputs",
             "Each of the 389 points pairs observed-range errors for one material. Points above parity have higher error with one current.")
    all_data=[cnn_errors(domain) for domain in DOMAINS]
    upper=max(.05*np.ceil(max(a.max() for result in all_data for a in result[:2])/.05),.1)
    for j,(one,three,one_s,three_s) in enumerate(all_data):
        ax=fig.add_axes([.115+j*.48,.38,.302,.425])
        ax.plot([0,upper],[0,upper],color="#7d9295",ls="--",lw=1.15,zorder=0)
        ax.scatter(three,one,color="#376676",alpha=.46,s=15,edgecolors="none",rasterized=True)
        ax.set_xlim(0,upper)
        ax.set_ylim(0,upper)
        ax.set_aspect("equal",adjustable="box")
        ax.set_title(DOMAIN_NAMES[j],loc="left",pad=9,fontsize=14,fontweight="bold",color=INK)
        ax.set_xlabel("Three currents · normalized curve RMSE",color=THREE,labelpad=5)
        ax.set_ylabel("One current · normalized curve RMSE",color=ONE)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        style(ax)
        ax.xaxis.label.set_color(THREE)
        ax.yaxis.label.set_color(ONE)
        base=.10+j*.48
        fig.text(base,.294,"Test-set RMSE",fontsize=9.5,color=MUTED,fontweight="bold")
        for c,(key,label) in enumerate((("D_normalized_rmse","D"),("t_normalized_rmse","t+"),("combined_normalized_rmse","Combined"))):
            x=base+.167+.081*c
            fig.text(x,.294,label,fontsize=9.4,color=MUTED,ha="center")
            fig.text(x,.247,f"{one_s[key]:.4f}",fontsize=11,ha="center",color=ONE)
            fig.text(x,.202,f"{three_s[key]:.4f}",fontsize=11,ha="center",color=THREE)
        fig.text(base,.247,"One current",color=ONE,fontsize=10.7)
        fig.text(base,.202,"Three currents",color=THREE,fontsize=10.7)
    footer(fig,"Each input is scored at 201 points between its lowest and highest measured concentrations in the valid CNN keyframes. For each material,\n"
               "three-current squared errors are averaged across its usable triplets (18,934 in total). All materials receive equal weight in the test-set RMSE.")
    return fig


def pinn_figure():
    fig=page("PINN property curves on clean and noisy inputs",
             "Each fitted curve is shown across its own observed concentration range for material 2.")
    summary=read(OBSERVED)
    for col,domain in enumerate(DOMAINS):
        fit=summary[domain]["pinn"]
        union=(min(fit[k]["u_min"] for k in ("one","three")),max(fit[k]["u_max"] for k in ("one","three")))
        left=.11+.467*col
        fig.text(left,.796,DOMAIN_NAMES[col],color=INK,fontweight="bold",fontsize=14)
        flag="Both protocols passed the fit criteria" if col==0 else "Both protocols failed the fit criteria"
        fig.text(left,.761,flag,color=MUTED,fontsize=10)
        for row,key in enumerate(("D_normalized_rmse","t_normalized_rmse")):
            ax=fig.add_axes([left,.517-.285*row,.343,.173])
            def curve(coeff,lo,hi):
                u=np.linspace(lo,hi,201)
                basis=np.array([(1-u)**3,3*u*(1-u)**2,3*u**2*(1-u),u**3])
                return u,np.asarray(coeff)@basis
            ax.plot(*curve(fit["truth"][row],*union),color=INK,ls=(0,(2.4,2.2)),lw=2,zorder=3)
            for protocol,tint in (("three",THREE),("one",ONE)):
                entry=fit[protocol]
                u,v=curve(entry["coefficients"][row],entry["u_min"],entry["u_max"])
                _,truth=curve(fit["truth"][row],u[0],u[-1])
                np.testing.assert_allclose(np.sqrt(np.mean((v-truth)**2)),entry[key])
                ax.plot(u,v,color=tint,lw=2,marker="o",ms=3,markevery=[0,len(u)-1])
            err_one,err_three=fit["one"][key],fit["three"][key]
            margin=.025*(union[1]-union[0])
            ax.set_xlim(union[0]-margin,union[1]+margin)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.set_ylabel("Normalized "+("D" if row==0 else "t+"))
            if row:
                ax.set_xlabel("Normalized concentration")
            else:
                ax.tick_params(labelbottom=False)
            ax.grid(axis="y",color=GRID,lw=.8)
            style(ax)
            ax.set_title(f"Curve RMSE    1 current  {err_one:.4f}      3 currents  {err_three:.4f}",loc="left",fontsize=9.3,pad=6,color=MUTED)
    fig.legend(handles=[Line2D([],[],color=INK,ls="--",label="Known material properties"),Line2D([],[],color=ONE,label="One current"),
                        Line2D([],[],color=THREE,label="Three currents")],frameon=False,ncol=3,loc="center",
               bbox_to_anchor=(.5,.139),fontsize=10)
    footer(fig,"The single current (ι = 2.2285) also appears in the triplet. Each curve RMSE uses 201 points inside that protocol’s measured interval.\n"
               "Clean curves use accepted final checkpoints. Noisy curves show the last candidates from runs that failed the same five project criteria.")
    return fig


def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.labelsize":10,"xtick.labelsize":9,"ytick.labelsize":9,
                         "pdf.fonttype":42,"savefig.facecolor":"white"})
    with PdfPages(OUT) as pages:
        for build in (input_figure,compact_figure,cnn_figure,pinn_figure):
            fig=build()
            pages.savefig(fig)
            plt.close(fig)
    print(OUT)


if __name__=="__main__":
    main()
