"""Cross-check the FV operator, parameter derivatives and source-grid discretization.

Only this diagnostic uses held-out synthetic coefficients; the fit runner loads
only the observed fields, measurement times, applied currents and material IDs.
"""
import json
import sys
from pathlib import Path
import numpy as np
from .physics_fv import objective,simulate,value_grad,face_terms

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/"latest_synthetic_ml"/"src"))
from electrolyte_mol import MOLConfig, PhysicalScales, NondimensionalElectrolyteMOL, constant_current
from nondimensional_bernstein_pde import CubicBernsteinDensityLaw,CubicBernsteinTransport

def main():
    with np.load(HERE/"cache"/"binned.npz") as z:
        take=z["split"]=="test"
        theta,ids,io,tau,mask,y=[np.ascontiguousarray(z[k][take]) for k in ("theta","ids","iota","tau","mask","clean")]
    fine=[]
    coarse=[]
    for row,truth in enumerate(theta):
        args=io[row],tau[row],mask[row]
        yy=y[row].astype(float)
        coarse.append(np.sqrt(objective(truth,*args,yy)))
        fine.append(np.sqrt(np.mean((simulate(truth,*args,80)[mask[row]]-yy[mask[row]])**2)))
    differences=[]
    rhs_errors=[]
    for row in (0,2,9):
        theta0=np.full(8,0.5)
        args=io[row],tau[row],mask[row],y[row].astype(float)
        gradient=value_grad(theta0,*args)[1]
        directions=np.eye(8)*1e-6
        numeric=np.asarray([(objective(theta0+step,*args)-objective(theta0-step,*args))/2e-6 for step in directions])
        differences.append(float(np.max(np.abs(gradient-numeric))))
        truth=theta[row]
        current=io[row,0]
        dlaw=CubicBernsteinDensityLaw([0.084741,0.218525,0.431779,0.698133])
        props=CubicBernsteinTransport(truth[:4],truth[4:]).as_electrolyte_properties(dlaw,PhysicalScales())
        reference=NondimensionalElectrolyteMOL(props,constant_current(current),config=MOLConfig(n_cells=40,advection="centered"))
        c=1+0.4*np.sin(2*np.pi*reference.x_cells)
        a,b=face_terms(c,truth,current)
        face_flux=np.r_[current,b[1:]-40*a[1:]*np.diff(c),current]
        rhs_errors.append(float(np.max(np.abs(-40*np.diff(face_flux)-reference.rhs(0,c)))))
    assert max(rhs_errors)<1e-9 and max(differences)<1e-7
    summary=dict(n=len(ids),gradient_absolute_error_vs_centered_difference=differences,
                 max_rhs_discrepancy_vs_source=rhs_errors,
                 clean_c_rmse_at_true_theta_by_resolution={n:dict(median=float(np.median(rr)),
                   p95=float(np.quantile(rr,0.95)),max=float(np.max(rr))) for n,rr in ((40,coarse),(80,fine))})
    (HERE/"cache"/"physics_validation.json").write_text(json.dumps(summary,indent=2))
    print(summary)
if __name__=="__main__":
    main()
