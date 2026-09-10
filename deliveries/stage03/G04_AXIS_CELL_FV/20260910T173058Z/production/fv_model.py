"""Independent G04 cell-centred axisymmetric finite-volume production operator.

No import of, or numerical call to, the shared scorer occurs in this module.
Units: seconds, metres, degrees Celsius, dry-basis kg/kg. Unknowns interleaved
[T_0,C_0,T_1,C_1,...] on (Nr,Nz); z=0 is the cylinder mid-plane.
The half-cell boundary coefficient is evaluated at its adjacent current Newton
state, exactly the constitutive quadrature in source strategy S04 equation E08.
Its dependence on T and C is differentiated, not frozen across nonlinear steps.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix


def bdf_weights(h: float, previous_h: float) -> tuple[float,float,float]:
    if min(h,previous_h)<=0:raise ValueError('Positive BDF time steps required')
    r=h/previous_h
    return (1+2*r)/(1+r), -(1+r), r*r/(1+r)


def coefficients(q: str,T: np.ndarray,C: np.ndarray) -> dict[str,np.ndarray]:
    T,C=np.broadcast_arrays(np.asarray(T,float),np.asarray(C,float))
    if np.any(C<0):raise ValueError('Negative dry-basis moisture')
    K=T+273.15
    if np.any(K<=0) or not np.isfinite(T).all() or not np.isfinite(C).all():
        raise ValueError('Invalid physical state')
    pos=C>0;inv=np.zeros_like(C);np.divide(1,C,out=inv,where=pos)
    if q=='Q1':
        rho=np.full_like(C,820.);cp=np.full_like(C,2600.);k=np.full_like(C,.36)
        rhoc=np.zeros_like(C);cpc=np.zeros_like(C);kc=np.zeros_like(C)
        aa=.89;D=7e-9*np.exp(-aa*inv);Dt=np.zeros_like(C)
    elif q in ('Q2','Q3'):
        rho=650+128*C;cp=1450+2736*C/(1+C);k=.21+.38*C/(1+C)
        rhoc=np.full_like(C,128.);cpc=2736/(1+C)**2;kc=.38/(1+C)**2
        aa=.45;D=2.4e-3*np.exp(-aa*inv-3850/K);Dt=D*3850/K**2
    elif q=='Q4':
        rho=760+90*C;cp=1850+2150*C/(1+C);k=.12+.20*C/(1+C)
        rhoc=np.full_like(C,90.);cpc=2150/(1+C)**2;kc=.20/(1+C)**2
        aa=.30;D=4.2e-4*np.exp(-aa*inv-3850/K);Dt=D*3850/K**2
    else:raise ValueError('Unknown question')
    D=np.where(pos,D,0.);Dc=np.where(pos,D*aa*inv**2,0.);Dt=np.where(pos,Dt,0.)
    return {'rho':rho,'cp':cp,'B':rho*cp,'Bc':rhoc*cp+rho*cpc,'k':k,'kc':kc,'D':D,'Dc':Dc,'Dt':Dt}


class Model:
    def __init__(self,q: str,nr: int,nz: int,drivers: str|Path,parameters: dict|None=None):
        if min(nr,nz)<2:raise ValueError('Both grid dimensions must be >=2')
        self.q=q;self.nr=nr;self.nz=nz;self.n=nr*nz;self.parameters=parameters or {}
        data=json.loads(Path(drivers).read_text());self.env=np.asarray(data['environment'],float)
        self.rad=np.asarray(data['radius'],float)
        self.hT=25*float(self.parameters.get('hT_multiplier',1.))
        self.hC=8e-7*float(self.parameters.get('hC_multiplier',1.))
        self.end=float(self.parameters.get('end_transfer_factor',1.));self.H=.125
        idx=np.arange(self.n).reshape(nr,nz)
        self.rp=idx[:-1].ravel();self.rq=idx[1:].ravel()
        self.zp=idx[:,:-1].ravel();self.zq=idx[:,1:].ravel()
        self.p=np.r_[self.rp,self.zp];self.qidx=np.r_[self.rq,self.zq]
        i=np.repeat(np.arange(nr-1),nz)
        self.rleft=(i+1)/(i+.5)*nr**2
        self.rright=(i+1)/(i+1.5)*nr**2
        self.br=idx[-1].copy();self.bz=idx[:,-1].copy()
        self.b=np.r_[self.br,self.bz]
        self.ir=np.repeat(np.arange(nr),nz);self.jz=np.tile(np.arange(nz),nr)
        self._colours=np.empty(2*self.n,dtype=int)
        self._colours[0::2]=(self.ir+2*self.jz)%5
        self._colours[1::2]=5+(self.ir+2*self.jz)%5
        # Structural five-point coupling, for a genuinely separate FD-Jacobian ablation.
        ps=np.r_[self.p,self.p,self.qidx,self.qidx,np.arange(self.n)]
        qs=np.r_[self.p,self.qidx,self.p,self.qidx,np.arange(self.n)]
        rs=[];cs=[]
        for outvar in (0,1):
            for invar in (0,1):rs.append(2*ps+outvar);cs.append(2*qs+invar)
        pat=coo_matrix((np.ones(sum(len(a) for a in rs)),(np.concatenate(rs),np.concatenate(cs))),shape=(2*self.n,2*self.n)).tocsc()
        pat.data[:]=1;self.pattern=pat
        self.fd_rows=pat.indices.copy();self.fd_cols=np.repeat(np.arange(2*self.n),np.diff(pat.indptr))
        self.counts={'rhs_calls':0,'analytic_jacobian_calls':0,'finite_difference_jacobian_calls':0}

    def initial(self) -> np.ndarray:
        y=np.empty(2*self.n);y[0::2]=28.;y[1::2]=2.55;return y

    def drivers(self,t: float) -> tuple[float,float,float]:
        ta=float(np.interp(t,self.env[:,0],self.env[:,1]));ca=float(np.interp(t,self.env[:,0],self.env[:,2]))
        if t>14400 and self.parameters.get('ambient_tail','last')=='last_hour_mean':
            tail=self.env[self.env[:,0]>=10800];ta=float(tail[:,1].mean());ca=float(tail[:,2].mean())
        ca*=float(self.parameters.get('Ce_multiplier',1.))
        policy=self.parameters.get('radius_policy','observed')
        if self.q!='Q4' or policy=='fixed_initial':R=.02
        else:
            if t>259200+1e-7 and policy!='hold_after72h':raise ValueError('Radius history exhausted')
            R=float(np.interp(t,self.rad[:,0],self.rad[:,1]))
        return ta,ca,R

    def geometry(self,R: float):
        wl=np.r_[self.rleft/R**2,np.full(len(self.zp),self.nz**2/self.H**2)]
        wr=np.r_[self.rright/R**2,np.full(len(self.zp),self.nz**2/self.H**2)]
        ba=np.r_[np.full(self.nz,self.nr**2/(R*(self.nr-.5))),np.full(self.nr,self.nz/self.H)]
        distances=np.r_[np.full(self.nz,R/(2*self.nr)),np.full(self.nr,self.H/(2*self.nz))]
        return wl,wr,ba,distances

    @staticmethod
    def conductance(a,d,h):
        # Stable at h=0 and a=0; no artificial lower diffusion bound.
        den=d*h+a
        g=np.divide(a*h,den,out=np.zeros_like(a),where=den>0)
        dg=np.divide(d*h*h,den*den,out=np.zeros_like(a),where=den>0)
        return g,dg

    def _field(self,u,a,daT,daC,external,h,geom,jac):
        wl,wr,ba,dist=geom;p=self.p;q=self.qidx
        ap=a[p];aq=a[q];den=ap+aq
        harm=np.divide(2*ap*aq,den,out=np.zeros_like(den),where=den>0)
        du=u[p]-u[q];flux=harm*du
        rate=np.bincount(p,weights=-wl*flux,minlength=self.n)+np.bincount(q,weights=wr*flux,minlength=self.n)
        hs=np.r_[np.full(self.nz,h),np.full(self.nr,h*self.end)]
        gb,dgb=self.conductance(a[self.b],dist,hs)
        fb=gb*(u[self.b]-external)
        rate+=np.bincount(self.b,weights=-ba*fb,minlength=self.n)
        surface=u[self.b].copy();active=hs>0
        surface[active]=external+fb[active]/hs[active]
        boundary={'surface_r':surface[:self.nz],'surface_z':surface[self.nz:],'flux_r':fb[:self.nz],'flux_z':fb[self.nz:]}
        if not jac:return rate,None,boundary
        hp=np.divide(2*aq*aq,den*den,out=np.zeros_like(den),where=den>0)
        hq=np.divide(2*ap*ap,den*den,out=np.zeros_like(den),where=den>0)
        partial=[]
        for field,da in ((0,daT),(1,daC)):
            dp=hp*da[p]*du;dq=hq*da[q]*du
            # Direct derivative of the transported field, apart from its coefficient.
            partial.append((dp,dq,dgb*da[self.b]*(u[self.b]-external)))
        return rate,(harm,gb,partial),boundary

    def rhs(self,t: float,y: np.ndarray,jac: bool=False):
        self.counts['rhs_calls']+=1
        T=np.asarray(y[0::2]);C=np.asarray(y[1::2]);b=coefficients(self.q,T,C)
        ta,ca,R=self.drivers(t);geom=self.geometry(R)
        zero=np.zeros(self.n)
        ht,hj,hb=self._field(T,b['k'],zero,b['kc'],ta,self.hT,geom,jac)
        mc,mj,mb=self._field(C,b['D'],b['Dt'],b['Dc'],ca,self.hC,geom,jac)
        f=np.empty_like(y);f[0::2]=ht/b['B'];f[1::2]=mc
        if not jac:return f,None
        self.counts['analytic_jacobian_calls']+=1
        wl,wr,ba,dist=geom;p=self.p;q=self.qidx
        rows=[];cols=[];data=[]
        for outvar,terms,scale in ((0,hj,1/b['B']),(1,mj,np.ones(self.n))):
            harm,gb,partial=terms
            for invar,(dp0,dq0,db0) in enumerate(partial):
                dp=dp0.copy();dq=dq0.copy();db=db0.copy()
                if outvar==invar:dp+=harm;dq-=harm;db+=gb
                rr=np.r_[p,p,q,q,self.b];cc=np.r_[p,q,p,q,self.b]
                vv=np.r_[-wl*dp,-wl*dq,wr*dp,wr*dq,-ba*db]*scale[rr]
                rows.append(2*rr+outvar);cols.append(2*cc+invar);data.append(vv)
        rows.append(2*np.arange(self.n));cols.append(2*np.arange(self.n)+1)
        data.append(-f[0::2]*b['Bc']/b['B'])
        J=coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(len(y),len(y))).tocsc()
        J.eliminate_zeros()
        return f,J

    def fd_jacobian(self,t: float,y: np.ndarray):
        self.counts['finite_difference_jacobian_calls']+=1
        values=np.empty_like(self.pattern.data);steps=np.cbrt(np.finfo(float).eps)*(np.abs(y)+1)
        for colr in range(10):
            mask=self._colours==colr;dv=np.where(mask,steps,0.)
            fp,_=self.rhs(t,y+dv,False);fm,_=self.rhs(t,y-dv,False)
            which=mask[self.fd_cols]
            values[which]=(fp[self.fd_rows[which]]-fm[self.fd_rows[which]])/(2*steps[self.fd_cols[which]])
        return csc_matrix((values,self.pattern.indices.copy(),self.pattern.indptr.copy()),shape=self.pattern.shape)

    def boundary(self,t: float,y: np.ndarray):
        T=y[0::2];C=y[1::2];b=coefficients(self.q,T,C);ta,ca,R=self.drivers(t);geom=self.geometry(R)
        zero=np.zeros(self.n)
        _,_,hb=self._field(T,b['k'],zero,b['kc'],ta,self.hT,geom,False)
        _,_,mb=self._field(C,b['D'],b['Dt'],b['Dc'],ca,self.hC,geom,False)
        return hb,mb

    def extended(self,t: float,y: np.ndarray):
        ta,ca,R=self.drivers(t);hb,mb=self.boundary(t,y);out={}
        for name,var,bound,ext in (('T_C',y[0::2],hb,ta),('C',y[1::2],mb,ca)):
            u=var.reshape(self.nr,self.nz);a=np.empty((self.nr+2,self.nz+2))
            a[1:-1,1:-1]=u;a[0,1:-1]=(9*u[0]-u[1])/8
            a[1:-1,0]=(9*u[:,0]-u[:,1])/8
            a[-1,1:-1]=bound['surface_r'];a[1:-1,-1]=bound['surface_z']
            a[0,0]=(81*u[0,0]-9*u[1,0]-9*u[0,1]+u[1,1])/64
            a[-1,0]=(9*bound['surface_r'][0]-bound['surface_r'][1])/8
            a[0,-1]=(9*bound['surface_z'][0]-bound['surface_z'][1])/8
            den=u[-1,-1]-ext
            a[-1,-1]=ext if abs(den)<1e-14 else ext+(bound['surface_r'][-1]-ext)*(bound['surface_z'][-1]-ext)/den
            out[name]=a
        out['r']=np.r_[0.,(np.arange(self.nr)+.5)*R/self.nr,R]
        out['z']=np.r_[0.,(np.arange(self.nz)+.5)*self.H/self.nz,self.H]
        return out

    def query(self,t: float,y: np.ndarray,radii: np.ndarray,zcoords: np.ndarray):
        ex=self.extended(t,y);rr=np.asarray(radii);zz=np.asarray(zcoords)
        ix=np.searchsorted(ex['r'],rr,side='right')-1;ix=np.clip(ix,0,self.nr)
        iz=np.searchsorted(ex['z'],zz,side='right')-1;iz=np.clip(iz,0,self.nz)
        ar=(rr-ex['r'][ix])/(ex['r'][ix+1]-ex['r'][ix]);az=(zz-ex['z'][iz])/(ex['z'][iz+1]-ex['z'][iz])
        ans=[]
        for key in ('T_C','C'):
            u=ex[key];v=(1-ar[:,None])*((1-az)*u[ix[:,None],iz]+az*u[ix[:,None],iz+1])+ar[:,None]*((1-az)*u[ix[:,None]+1,iz]+az*u[ix[:,None]+1,iz+1])
            v[(rr<0)|(rr>ex['r'][-1]+1e-14),:]=np.nan;ans.append(v)
        return tuple(ans)

    def maximum(self,t: float,y: np.ndarray):
        ex=self.extended(t,y);C=ex['C'];idx=np.unravel_index(np.argmax(C),C.shape)
        return float(C[idx]),[float(ex['r'][idx[0]]),float(ex['z'][idx[1]])]

    def diagnostics(self,t: float,y: np.ndarray,f: np.ndarray):
        ta,ca,R=self.drivers(t);b=coefficients(self.q,y[0::2],y[1::2]);hb,mb=self.boundary(t,y)
        faces=np.linspace(0,R,self.nr+1);vr=.5*np.diff(faces**2);dz=self.H/self.nz
        vol=np.repeat(vr[:,None]*dz,self.nz,axis=1).ravel()
        oh=float(R*dz*sum(hb['flux_r'])+np.dot(vr,hb['flux_z']))
        om=float(R*dz*sum(mb['flux_r'])+np.dot(vr,mb['flux_z']))
        ex=self.extended(t,y)
        return {'time_s':float(t),'R_m':R,'outward_heat_flux':oh,'outward_moisture_flux':om,
                'heat_balance_defect':float(np.dot(vol*b['B'],f[0::2])+oh),
                'moisture_balance_defect':float(np.dot(vol,f[1::2])+om),
                'min_T_C':float(ex['T_C'].min()),'max_T_C':float(ex['T_C'].max()),
                'min_C':float(ex['C'].min()),'max_C':float(ex['C'].max())}
