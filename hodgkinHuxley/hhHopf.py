import numpy as np
import sympy as sym
import matplotlib.pyplot as plt


def equationHH():
    #V = sym.Symbol('V',cls=sym.Function)
    V, m,h,n = sym.symbols('V,m,h,n', cls=Function)
    t = sym.Symbol('t')
    g = sym.Symbol('g')

    C = 1
    gNa = 120
    gK = 36
    gL = 0.3
    VNa = 115
    VK = -12
    Vl = 10.599
    
    am_= 0.08
    bm_ = 0.8
    ah_ = 0.07
    bh_ = 1.0
    an_ = 0.01
    bn_ = 0.125

    I = 120

    T = 3 ** (30)
    
    eqs=[]
    eqs.append(sym.Eq())
    eqs.append(sym.Eq(m(t)*sym.diff(m),T*(am*(1-m) - bm*m)))
    eqs.append(sym.Eq(h(t)*sym.diff(h),T*(ah*(1-h) - bh*h)))
    eqs.append(sym.Eq(n(t)*sym.diff(n),T*(an*(1-n) - bn*n)))
    eqs.append(sym.Eq(V(t)*sym.diff*V),(1/C) *[I - gNa*(m**3)*h(V-VNa)] )
    
    sym.dsolve()