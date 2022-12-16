import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

#10.1016/j.chaos.2006.01.035

def equationHH():
    #V = sym.Symbol('V',cls=sym.Function)
    V, m,h,n = sym.symbols('V,m,h,n')
    t = sym.Symbol('t')

    Cm = 1
    gNa = 120
    gK = 36
    gL = 0.3
    VNa = 115
    VK = -12
    VL = 10.599
    
    #am_= 0.08
    #bm_ = 0.8
    #ah_ = 0.07
    #bh_ = 1.0
    #an_ = 0.01
    #bn_ = 0.125

    I = 120

    T = 3 ** (30)
    
    am = 0.1*(2.0 -V)/ (sym.exp((25.0 -V)/10.0)-10.0)
    bm = 4.0 * sym.exp(-1.0 * V /18.0)
    ah = 0.07 * sym.exp(-1.0 * V /20.0)
    bh = 1.0 / sym.exp(((30 - V)/10.0) +1.0)
    an = 0.01 * (10.0 -V) / (sym.exp((10.0 - V )/10.0) -1.0)
    bn = 0.125* sym.exp(-1.0 * V / 80.0)




    eqs=[]
    
    eq1 = sym.Eq( I  -gNa * (m**3)* h * (V- VNa) - gK* (n**4)*(V - VK) -gL*(V- VL))
    
    eq2 = sym.Eq(am * (1 - m) - bm * m)

    eq3 = sym.Eq(ah * (1 - h) - bh * h)

    eq4 = sym.Eq(an * (1 - n) - bn * n)

    eqs.append(eq1,eq2,eq3,eq4)
    
    
    sym.solve(eqs,minimal=True)