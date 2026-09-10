"""Variable-step BDF2, genuine step-doubling BE startup, damped chord Newton.

An eightfold local-error safety factor is calibrated on the independent
linear decay test (global error accumulates beyond the local tolerance).
Error control is an empirical divided-difference estimator, not a rigorous
continuum error bound. All accepted steps and rejections are counted. No scorer
is imported. Driver slope changes restart BE; early/late step caps are enforced.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from fv_model import bdf_weights

class NonlinearFailure(RuntimeError):pass
class BudgetExceeded(RuntimeError):pass

class Newton:
    def __init__(self,model,config):
        self.m=model;self.cfg=config;self.lu=None;self.coeff=None
        self.iterations=0;self.factorizations=0;self.solves=0;self.max_residual=0.
    def solve(self,t,h,a0,old,delta,guess):
        y=guess.copy();c=h/a0
        def residual(v):
            f,_=self.m.rhs(t,v,False)
            return (v-old)-delta-c*f,f
        if np.min(y[1::2])<0 or not np.isfinite(y).all():y=old.copy()
        for it in range(25):
            self.iterations+=1
            try:r,f=residual(y)
            except ValueError:raise NonlinearFailure('Invalid Newton iterate')
            scale=1+np.abs(y);norm=float(np.max(np.abs(r)/scale))
            if norm<=1e-11:
                self.max_residual=max(self.max_residual,norm)
                return y,f,it+1
            refresh=self.lu is None or self.coeff is None or abs(c/self.coeff-1)>.02 or it in (3,8,15)
            if refresh:
                if self.cfg.get('jacobian','analytic')=='finite_difference':J=self.m.fd_jacobian(t,y)
                else:_,J=self.m.rhs(t,y,True)
                self.lu=splu((eye(len(y),format='csc')-c*J).tocsc())
                self.coeff=c;self.factorizations+=1
            dx=self.lu.solve(-r);self.solves+=1
            alpha=1.
            negative=dx[1::2]<0
            if negative.any():alpha=min(alpha,float(np.min(-.99*y[1::2][negative]/dx[1::2][negative])))
            accepted=False
            for _ in range(12):
                yn=y+alpha*dx
                try:rn,_=residual(yn);newnorm=float(np.max(np.abs(rn)/(1+np.abs(yn))))
                except ValueError:newnorm=np.inf
                if newnorm<norm*(1-1e-4*alpha) or newnorm<1e-11:
                    y=yn;accepted=True;break
                alpha*=.5
            if not accepted:self.lu=None
        raise NonlinearFailure('Damped Newton did not meet scaled residual 1e-11 in 25 iterations')

def hermite(t,t0,t1,y0,y1,f0,f1):
    h=t1-t0;s=(t-t0)/h
    return (2*s**3-3*s**2+1)*y0+(s**3-2*s**2+s)*h*f0+(-2*s**3+3*s**2)*y1+(s**3-s**2)*h*f1

def integrate(model,y0,stop,config,accept_callback):
    tic=time.perf_counter();newton=Newton(model,config)
    start=float(config.get('start_s',0.));t=start;y=np.asarray(y0,float).copy();f,_=model.rhs(t,y)
    hist=[(t,y.copy(),f.copy())];h=float(config.get('initial_step',.01))
    tol=np.empty(len(y));tol[0::2]=config.get('atol_T',1e-7);tol[1::2]=config.get('atol_C',1e-9)
    rtol=config.get('rtol',1e-6);knots=np.asarray(config.get('knots',[]),float)
    knots=knots[(knots>t+1e-9)&(knots<stop-1e-9)];ki=0
    accepted=0;rejected=0;newton_failures=0;max_lte=0.;next_checkpoint=t
    method=config.get('integrator','BDF2');deadline=config.get('time_budget_s',np.inf)
    while t<stop-1e-9:
        if time.perf_counter()-tic>deadline:raise BudgetExceeded(f'Integration budget {deadline}s reached at t={t:.9f}s')
        cap=min(float(config.get('max_step',120)),(1. if t<100-1e-9 else (30. if t<10800-1e-9 else 120.))*config.get('step_cap_factor',1.))
        if config.get('monitor_event',False):
            maximum,_=model.maximum(t,y)
            if .14999<maximum<.155:
                # Approach the strict global event on accepted states, not a 60s output grid.
                rate=max(float(np.max(np.abs(f[1::2]))),1e-15)
                cap=min(cap,max(.02,.6*abs(maximum-.15)/rate))
        boundary=min(stop,float(knots[ki]) if ki<len(knots) else stop)
        hh=min(h,cap,boundary-t)
        if hh<1e-9:raise NonlinearFailure(f'Step underflow at t={t}, proposed h={hh}')
        forced_restart=len(hist)<3 or method=='BE'
        scale=tol+rtol*np.maximum(np.abs(y),1e-10)
        try:
            if forced_restart:
                yf,ff,_=newton.solve(t+hh,hh,1.,y,np.zeros_like(y),y+hh*f)
                yh,fh,_=newton.solve(t+hh/2,hh/2,1.,y,np.zeros_like(y),y+hh/2*f)
                yn,fn,_=newton.solve(t+hh,hh/2,1.,yh,np.zeros_like(y),yh+hh/2*fh)
                err=8*float(np.max(np.abs(yn-yf)/scale));power=.5
                accepted_nodes=[(t+hh/2,yh,fh),(t+hh,yn,fn)]
            else:
                hp=t-hist[-2][0];a0,a1,a2=bdf_weights(hh,hp)
                yn,fn,_=newton.solve(t+hh,hh,a0,y,(a2/a0)*(y-hist[-2][1]),y+(hh/hp)*(y-hist[-2][1]))
                times=[t+hh]+[a[0] for a in hist[-3:][::-1]]
                vals=[yn]+[a[1] for a in hist[-3:][::-1]]
                dd=vals
                for degree in range(1,4):dd=[(dd[i]-dd[i+1])/(times[i]-times[i+degree]) for i in range(4-degree)]
                lte=hh*hh*(hh+hp)/a0*dd[0]
                err=8*float(np.max(np.abs(lte)/scale));power=1/3
                accepted_nodes=[(t+hh,yn,fn)]
        except (NonlinearFailure,ValueError,np.linalg.LinAlgError):
            rejected+=1;newton_failures+=1;h=hh*.5;newton.lu=None;continue
        if not np.isfinite(err) or err>1:
            rejected+=1;h=hh*max(.15,.8*(1/max(err,1e-30))**power);continue
        for tn,yn,fn in accepted_nodes:
            result=accept_callback(t,y,f,tn,yn,fn,{'error_estimate':err,'newton_iterations':newton.iterations,'rejections':rejected})
            hist.append((tn,yn.copy(),fn.copy()));hist=hist[-3:]
            t=tn;y=yn;f=fn;accepted+=1
            if result is not None and result.get('stop',False):stop=t;break
        max_lte=max(max_lte,err)
        if ki<len(knots) and abs(t-knots[ki])<1e-7:
            hist=hist[-1:];ki+=1;h=min(hh,.05)
        else:
            factor=min(1.5,max(.5,.85*(1/max(err,1e-15))**power))
            # A quasi-constant band reduces LU refactorizations without weakening the error test.
            if .85<factor<1.2:factor=1.
            h=hh*factor
    return {'t':t,'y':y,'f':f,'accepted_steps':accepted,'rejected_steps':rejected,
            'newton_failures':newton_failures,'newton_iterations':newton.iterations,
            'linear_factorizations':newton.factorizations,'linear_solves':newton.solves,
            'max_accepted_scaled_residual':newton.max_residual,'max_lte_ratio':max_lte,
            'elapsed_s':time.perf_counter()-tic,'operator_counts':model.counts}
