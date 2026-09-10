"""Independent P1 radial Galerkin FEM; no import of the shared evaluator.

State uses degrees C and dry-basis moisture. Consistent three-point Gauss
mass matrices retain rho*cp outside the divergence. Material coordinate x=r/R:
 M_T(C) Tdot = -K_k T/R**2 - h_T*(T_surface-T_air)/R
 M_1    Cdot = -K_D C/R**2 - h_C*(C_surface-C_air)/R.
There is no fictitious dilution term for dry-basis C. Only numerical parameters
are configurable; empirical constants are frozen to the supplied appendices.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import solve_banded


def properties(q, c, t):
    c,t=np.broadcast_arrays(np.asarray(c,float),np.asarray(t,float))
    if np.any(c<0): raise ValueError('NEGATIVE_MOISTURE_REJECT_STEP')
    if np.any(t<=-273.15): raise ValueError('NONPOSITIVE_KELVIN')
    inv=np.divide(1.,c,out=np.zeros_like(c),where=c>0)
    if q=='Q1':
        rho=np.full_like(c,820.);cp=np.full_like(c,2600.);k=np.full_like(c,.36)
        mC=np.zeros_like(c);kC=np.zeros_like(c)
        D=7e-9*np.exp(-.89*inv);a=.89;DT=np.zeros_like(c)
    elif q in ('Q2','Q3'):
        rho=650+128*c;cp=1450+2736*c/(1+c);k=.21+.38*c/(1+c)
        mC=128*cp+rho*2736/(1+c)**2;kC=.38/(1+c)**2
        D=2.4e-3*np.exp(-.45*inv-3850/(t+273.15));a=.45
        DT=D*3850/(t+273.15)**2
    elif q=='Q4':
        rho=760+90*c;cp=1850+2150*c/(1+c);k=.12+.20*c/(1+c)
        mC=90*cp+rho*2150/(1+c)**2;kC=.20/(1+c)**2
        D=4.2e-4*np.exp(-.30*inv-3850/(t+273.15));a=.30
        DT=D*3850/(t+273.15)**2
    else: raise ValueError('UNKNOWN_QUESTION')
    D=np.where(c>0,D,0.);DT=np.where(c>0,DT,0.)
    return dict(rho=rho,cp=cp,k=k,D=D,m=rho*cp,mC=mC,kC=kC,DC=D*a*inv**2,DT=DT)


class FEM:
    def __init__(self,N,q,root=None,scenario=None,closed=False):
        self.N=int(N);self.n=self.N+1;self.q=q;self.x=np.linspace(0.,1.,self.n);self.h=1/self.N
        g,w=np.polynomial.legendre.leggauss(3)
        self.phi=np.stack(((1-g)/2,(1+g)/2),axis=1)
        self.xq=self.x[:-1,None]+(g[None,:]+1)*self.h/2
        self.w=self.xq*w[None,:]*self.h/2
        self.scenario=scenario or {};self.closed=closed
        root=Path(root) if root else Path(__file__).resolve().parents[1]
        data=json.loads((root/'configs/drivers.json').read_text())
        self.env=np.asarray(data['environment']);self.radius=np.asarray(data['radius'])
        self.Mc=self.mass_bands(np.ones((self.N,3)));self.rhs_calls=0;self.jac_calls=0
        self.last_balance=None
    def drivers(self,t):
        ta=np.interp(t,self.env[:,0],self.env[:,1]);ca=np.interp(t,self.env[:,0],self.env[:,2])
        if self.scenario.get('ambient_tail')=='last_hour_mean' and t>14400:
            ta,ca=self.env[self.env[:,0]>=10800,1:].mean(axis=0)
        if self.q!='Q4' or self.scenario.get('radius_policy')=='fixed_initial':R=.02
        else:
            if t>259200 and self.scenario.get('radius_policy')!='hold_after72h':raise ValueError('OBSERVED_RADIUS_EXHAUSTED')
            R=float(np.interp(t,self.radius[:,0],self.radius[:,1]))
        return float(ta),float(ca),R
    def mass_bands(self,a):
        v=self.w*np.broadcast_to(a,self.w.shape);P=self.phi
        ll=np.sum(v*P[:,0]**2,axis=1);rr=np.sum(v*P[:,1]**2,axis=1)
        lr=np.sum(v*P[:,0]*P[:,1],axis=1)
        b=np.zeros((3,self.n));b[1,:-1]+=ll;b[1,1:]+=rr;b[0,1:]=lr;b[2,:-1]=lr
        return b
    def mass(self,a):
        b=self.mass_bands(a)
        return np.diag(b[1])+np.diag(b[0,1:],1)+np.diag(b[2,:-1],-1)
    def sample(self,y):
        T,C=y[:self.n],y[self.n:]
        return T,C,T[:-1,None]*self.phi[:,0]+T[1:,None]*self.phi[:,1],C[:-1,None]*self.phi[:,0]+C[1:,None]*self.phi[:,1]
    def _system(self,t,y,with_jac=False):
        T,C,tq,cq=self.sample(y)
        if np.min(C)<0:raise ValueError('NEGATIVE_MOISTURE_REJECT_STEP')
        p=properties(self.q,cq,tq);ta,ca,R=self.drivers(t)
        hT=0. if self.closed else 25.;hC=0. if self.closed else 8e-7
        MT=self.mass_bands(p['m']);Mc=self.Mc;forces=[]
        for u,a,h,ext in [(T,p['k'],hT,ta),(C,p['D'],hC,ca)]:
            e=np.sum(self.w*a,axis=1)/(self.h*R)**2
            flux=e*np.diff(u);F=np.zeros(self.n);F[:-1]+=flux;F[1:]-=flux
            F[-1]-=h/R*(u[-1]-ext);forces.append(F)
        dT=solve_banded((1,1),MT,forces[0],check_finite=False)
        dC=solve_banded((1,1),Mc,forces[1],check_finite=False)
        rate=np.r_[dT,dC]
        # Independent weak integral identities, not the shared FV audit.
        rowT=MT[1].copy();rowT[:-1]+=MT[0,1:];rowT[1:]+=MT[2,:-1]
        rowC=Mc[1].copy();rowC[:-1]+=Mc[0,1:];rowC[1:]+=Mc[2,:-1]
        self.last_balance=(float(rowT@dT+hT/R*(T[-1]-ta)),float(rowC@dC+hC/R*(C[-1]-ca)))
        if not with_jac:return rate
        blocks=[np.zeros((self.n,self.n)) for _ in range(4)]
        Jtt,Jtc,Jct,Jcc=blocks
        def add_local(mat,v):
            i=np.arange(self.N)
            mat[i,i]+=v[:,0,0];mat[i,i+1]+=v[:,0,1]
            mat[i+1,i]+=v[:,1,0];mat[i+1,i+1]+=v[:,1,1]
        for mat,a,h in [(Jtt,p['k'],hT),(Jcc,p['D'],hC)]:
            e=np.sum(self.w*a,axis=1)/(self.h*R)**2
            loc=e[:,None,None]*np.array([[-1.,1.],[1.,-1.]])[None,:,:]
            add_local(mat,loc);mat[-1,-1]-=h/R
        dtq=dT[:-1,None]*self.phi[:,0]+dT[1:,None]*self.phi[:,1]
        for mat,derivative,du in [(Jtc,p['kC'],np.diff(T)),(Jct,p['DT'],np.diff(C)),(Jcc,p['DC'],np.diff(C))]:
            s=np.einsum('eq,qj->ej',self.w*derivative,self.phi)*du[:,None]/(self.h*R)**2
            loc=np.stack((s,-s),axis=1)
            if mat is Jtc:
                loc-=np.einsum('eq,qi,qj->eij',self.w*p['mC']*dtq,self.phi,self.phi)
            add_local(mat,loc)
        J=np.empty((2*self.n,2*self.n))
        J[:self.n,:]=solve_banded((1,1),MT,np.hstack((Jtt,Jtc)),check_finite=False)
        J[self.n:,:]=solve_banded((1,1),Mc,np.hstack((Jct,Jcc)),check_finite=False)
        return J
    def rhs(self,t,y):
        self.rhs_calls+=1
        return self._system(t,y)
    def jac(self,t,y):
        self.jac_calls+=1
        return self._system(t,y,True)
