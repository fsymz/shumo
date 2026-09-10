"""Independent radial finite-volume operator. No imports from the common scorer.
C is dry-basis ratio. xi cells move with the material for Q4; rho is only
thermal storage. Boundary film and nonlinear half-cell resistances are in series.
"""
import numpy as np
from numba import njit

@njit(cache=False)
def law_scalar(q,c,t):
    if c < 0.0 or t <= -273.15:
        raise ValueError('outside physical constitutive domain')
    if q == 1:
        return 820.,2600.,.36,0. if c == 0. else 7e-9*np.exp(-.89/c)
    if q == 4:
        return 760.+90.*c,1850.+2150.*c/(1.+c),.12+.20*c/(1.+c),0. if c == 0. else 4.2e-4*np.exp(-.30/c-3850./(t+273.15))
    return 650.+128.*c,1450.+2736.*c/(1.+c),.21+.38*c/(1.+c),0. if c == 0. else 2.4e-3*np.exp(-.45/c-3850./(t+273.15))

def checked_laws(q,c,t):
    c=np.asarray(c,dtype=float);t=np.asarray(t,dtype=float)
    if c.shape != t.shape or np.any(c<0) or np.any(t<=-273.15):
        raise ValueError('invalid shapes or physical states')
    return np.array([law_scalar(q,float(x),float(z)) for x,z in zip(c.flat,t.flat)]).reshape(c.shape+(4,))

@njit(cache=False)
def surface(q,tc,cc,half,ta,ca,ht=25.,hc=8e-7):
    """Solve the two nonlinear Robin/half-cell equations to machine accuracy.
    Face constitutive values evaluated at the half-cell midpoint. This is a
    second-order local closure, tested by spatial refinement, not exact PDE data.
    Flux is outward positive; no concentration floor or frozen last-cell surface.
    """
    if half<=0 or cc<0 or ca<0: raise ValueError('invalid surface inputs')
    lo=min(cc,ca); hi=max(cc,ca)
    cs=cc
    for it in range(60):
        cs=(lo+hi)*.5 if hc>0 else cc
        k=law_scalar(q,.5*(cc+cs),tc)[2]
        ts=tc+(half*ht/(k+half*ht))*(ta-tc)
        d=law_scalar(q,.5*(cc+cs),.5*(tc+ts))[3]
        f=d*(cc-cs)/half-hc*(cs-ca)
        if f>0: lo=cs
        else: hi=cs
    k=law_scalar(q,.5*(cc+cs),tc)[2]
    ts=tc+(half*ht/(k+half*ht))*(ta-tc)
    d=law_scalar(q,.5*(cc+cs),.5*(tc+ts))[3]
    qt=ht*(ts-ta);qc=hc*(cs-ca)
    res=max(abs(k*(tc-ts)/half-qt)/max(1.,abs(qt)),abs(d*(cc-cs)/half-qc))
    return ts,cs,qt,qc,res

@njit(cache=False)
def divergence(u,a,radius,outer_flux):
    n=len(u);dx=radius/n
    face=np.zeros(n+1);vol=np.empty(n);rhs=np.empty(n)
    for i in range(1,n):
        af=0. if a[i]+a[i-1]==0. else 2.*a[i]*a[i-1]/(a[i]+a[i-1])
        face[i]=-af*(u[i]-u[i-1])/dx
    face[n]=outer_flux
    for i in range(n):
        rl=i*dx; rr=(i+1)*dx
        vol[i]=.5*(rr*rr-rl*rl)
        rhs[i]=(rl*face[i]-rr*face[i+1])/vol[i]
    return rhs,vol

@njit(cache=False)
def operator(q,y,radius,ta,ca,ht=25.,hc=8e-7):
    n=len(y)//2;t=y[0::2];c=y[1::2]
    k=np.empty(n);d=np.empty(n);cap=np.empty(n)
    for i in range(n):
        rho,cp,k[i],d[i]=law_scalar(q,c[i],t[i]);cap[i]=rho*cp
    ts,cs,qt,qc,res=surface(q,t[-1],c[-1],radius/(2*n),ta,ca,ht,hc)
    dt,vol=divergence(t,k,radius,qt);dc,_=divergence(c,d,radius,qc)
    dy=np.empty(2*n)
    for i in range(n): dy[2*i]=dt[i]/cap[i];dy[2*i+1]=dc[i]
    # Balance is an algebraic telescoping audit, not independent convergence proof.
    bal_c=abs(np.dot(dc,vol)+radius*qc)
    bal_t=abs(np.dot(dt,vol)+radius*qt)
    obs=np.array([ts,cs,qt,qc,bal_c,bal_t,res])
    return dy,obs

@njit(cache=False)
def axis_value(u):
    return (9.*u[0]-u[1])/8.

@njit(cache=False)
def project_row(y,radius,ta,ca,q,rquery,ht=25.,hc=8e-7):
    n=len(y)//2;t=y[0::2];c=y[1::2]
    ts,cs,_,_,_=surface(q,t[-1],c[-1],radius/(2*n),ta,ca,ht,hc)
    x=np.empty(n+2);x[0]=0.;x[-1]=radius
    tt=np.empty(n+2);cc=np.empty(n+2)
    tt[0]=axis_value(t);cc[0]=axis_value(c);tt[-1]=ts;cc[-1]=cs
    for i in range(n):
        x[i+1]=(i+.5)*radius/n;tt[i+1]=t[i];cc[i+1]=c[i]
    ot=np.interp(rquery,x,tt);oc=np.interp(rquery,x,cc)
    for i in range(len(rquery)):
        if rquery[i]>radius+1e-14: ot[i]=np.nan;oc[i]=np.nan
    return ot,oc,ts,cs

@njit(cache=False)
def extrema(y,radius,ta,ca,q,ht=25.,hc=8e-7):
    t=y[0::2];c=y[1::2]
    ts,cs,_,_,_=surface(q,t[-1],c[-1],radius/(2*len(c)),ta,ca,ht,hc)
    return np.array([min(np.min(t),axis_value(t),ts),max(np.max(t),axis_value(t),ts),min(np.min(c),axis_value(c),cs),max(np.max(c),axis_value(c),cs)])
