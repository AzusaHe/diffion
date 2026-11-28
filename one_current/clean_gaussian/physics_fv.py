"""Differentiable conservative electrolyte forward solver for binned observations.

The finite-volume salt and solvent fluxes follow electrolyte_mol.py. A linearly
implicit midpoint corrector evolves 40 cells, with small startup steps. Pairs of
cells give the predicted 20 bin averages. Complex steps differentiate the
discrete time evolution. Only out-of-domain trial states use clipped properties.
"""
import numpy as np
from numba import njit

@njit(cache=True, nogil=True)
def face_terms(c, theta, current):
    n = len(c)
    transport = np.empty_like(c)
    volume = np.empty_like(c)
    transference = np.empty_like(c)
    rc = (0.084741, 0.218525, 0.431779, 0.698133)
    for j in range(n):
        cc = c[j]
        if cc.real <= 0:
            cc = cc*0
        elif cc.real >= 2:
            cc = cc*0 + 2
        u = 0.5*cc
        q = 1-u
        basis = (q**3, 3*u*q*q, 3*q*u*u, u**3)
        bd,bt,br = theta[0]*0,theta[0]*0,cc*0
        for k in range(4):
            bd += basis[k]*theta[k]
            bt += basis[k]*theta[4+k]
            br += basis[k]*rc[k]
        rho = 1000 + 1500*br
        drho = 2250*((rc[1]-rc[0])*q*q + 2*(rc[2]-rc[1])*u*q + (rc[3]-rc[2])*u*u)
        diff = rho-cc*drho
        volume[j] = (861.27-drho)/diff
        transport[j] = 0.1*np.exp(np.log(20.)*bd)*diff/(rho-861.27*cc)
        transference[j] = -0.25+1.25*bt
    # For each face, split the total salt flux into -a * dc/dx + b.
    a = np.zeros_like(c)
    b = np.zeros_like(c)
    b[0] = current
    vprev,qprev = c[0]*0,current + c[0]*0
    for j in range(1,n):
        cf = 0.5*(c[j]+c[j-1])
        conduct = 2*transport[j]*transport[j-1]/(transport[j]+transport[j-1])
        migr = current*0.5*(transference[j]+transference[j-1])
        qface = migr - n*conduct*(c[j]-c[j-1])
        factor = 1-cf*volume[j-1]
        a[j] = factor*conduct
        b[j] = factor*migr + cf*(vprev + volume[j-1]*qprev)
        vprev -= volume[j-1]*(qface-qprev)
        qprev = qface
    return a,b

@njit(cache=True, nogil=True)
def update(c, a, b, current, dt, weight):
    n = len(c)
    diag = np.ones_like(c)
    upper = np.zeros_like(c)
    rhs = c.copy()
    for j in range(1,n):
        explicit = b[j] - (1-weight)*a[j]*n*(c[j]-c[j-1])
        rhs[j] += dt*n*explicit
        rhs[j-1] -= dt*n*explicit
        s = weight*dt*n*n*a[j]
        upper[j-1] = -s
        diag[j-1] += s
        diag[j] += s
    rhs[0] += dt*n*current
    rhs[-1] -= dt*n*current
    for j in range(1,n):
        f = upper[j-1]/diag[j-1]
        diag[j] -= f*upper[j-1]
        rhs[j] -= f*rhs[j-1]
    rhs[-1] /= diag[-1]
    for j in range(n-2,-1,-1):
        rhs[j] = (rhs[j]-upper[j]*rhs[j+1])/diag[j]
    return rhs

@njit(cache=True, nogil=True)
def simulate(theta, currents, tau, mask, n=40):
    output = np.zeros((len(currents),tau.shape[1],20),dtype=theta.dtype)
    for j in range(len(currents)):
        c = np.ones(n,dtype=theta.dtype)
        previous = 0.
        for t in range(tau.shape[1]):
            if not mask[j,t]:
                continue
            span = tau[j,t]-previous
            while span > 1e-11:
                dt = min(span, min(0.008,0.00025+previous*0.18))
                a,b = face_terms(c,theta,currents[j])
                mid = update(c,a,b,currents[j],dt/2,1.)
                a,b = face_terms(mid,theta,currents[j])
                c = update(c,a,b,currents[j],dt,0.5)
                previous += dt
                span -= dt
            for k in range(20):
                output[j,t,k] = np.mean(c[k*(n//20):(k+1)*(n//20)])
    return output

@njit(cache=True, nogil=True)
def objective(theta, currents, tau, mask, y):
    out = simulate(theta,currents,tau,mask)
    value = theta[0]*0
    for j in range(len(currents)):
        for t in range(tau.shape[1]):
            if mask[j,t]:
                for x in range(20):
                    d = out[j,t,x]-y[j,t,x]
                    value += d*d
    return value/(mask.sum()*20)

@njit(cache=True, nogil=True)
def value_grad(theta, currents, tau, mask, y):
    value = objective(theta,currents,tau,mask,y)
    gradient = np.zeros_like(theta)
    z = theta.astype(np.complex128)
    for k in range(8):
        z[k] += 1e-22j
        gradient[k] = objective(z,currents,tau,mask,y).imag * 1e22
        z[k] = theta[k]
    return value,gradient
