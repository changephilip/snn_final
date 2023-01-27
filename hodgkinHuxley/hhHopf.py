import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# 10.1016/j.chaos.2006.01.035

# V = sym.Symbol('V',cls=sym.Function)
V, Ve, m, h, n ,Iext= sym.symbols('V,Ve,m,h,n,Iext')
t = sym.Symbol('t')

Cm = 1
gNa = 120.0
gK = 36.0
gL = 0.3
VNa = 115.0
VK = -12.0
VL = 10.599

I = 120

T = 3 ** (30)

am = 0.1*(25.0 - V) / (sym.exp((25.0 - V)/10.0)-1.0)
bm = 4.0 * sym.exp(-1.0 * V / 18.0)
ah = 0.07 * sym.exp(-1.0 * V / 20.0)
bh = 1.0 / (sym.exp((30 - V)/10.0) + 1.0)
an = 0.01 * (10.0 - V) / (sym.exp((10.0 - V)/10.0) - 1.0)
bn = 0.125 * sym.exp(-1.0 * V / 80.0)


eqs = []

eq1 = (- gNa * (m**3) * h * (V +Ve - VNa) -
       gK * (n**4)*(V +Ve - VK) - gL*(V +Ve - VL))  + I

eq2 = sym.Eq(am * (1.0 - m) - bm * m)

eq3 = sym.Eq(ah * (1.0 - h) - bh * h)

eq4 = sym.Eq(an * (1.0 - n) - bn * n)

eqm = sym.solve(eq2,m)[0]
eqh = sym.solve(eq3,h)[0]
eqn = sym.solve(eq4,n)[0]


me = eqm.subs(V,0)
he = eqh.subs(V,0)
ne = eqn.subs(V,0)
eqFinal=eq1.replace(m,eqm).replace(h,eqh).replace(n,eqn)

from sympy.plotting import plot

#plot(eqFinal,(V,-200,0))

