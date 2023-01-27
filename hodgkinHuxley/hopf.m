syms V;
syms n;
syms h;
syms m;
syms t;
syms Iext;


cm=1; 
gNa=120.0;
gK=36.0;
gL=0.3;
VNa=115.0;
VK=-12.0;
VL=10.599;
VE=0.0;

% argset 1
 am=0.1*(25-V)/(exp((25-V)/10)-1);
 bm = 4.0 *exp(-1.0 * V / 18.0);
 ah = 0.07 * exp(-1.0 * V / 20.0);
 bh = 1.0 / (exp((30 - V)/10.0) + 1.0);
 an = 0.01 * (10.0 - V) / (exp((10.0 - V)/10.0) - 1.0);
 bn = 0.125 * exp(-1.0 * V / 80.0);

% argset 2

%am=(2.5-0.1*(V+65)) ./ (exp(2.5-0.1*(V+65)) -1);
%bm = 4*exp(-(V+65)/18);
%ah = 0.07*exp(-(V+65)/20);
%bh = 1./(exp(3.0-0.1*(V+65))+1);
%an = (0.1-0.01*(V+65)) ./ (exp(1-0.1*(V+65)) -1);
%bn = 0.125*exp(-(V+65)/80);

eq1 = Iext - ( gNa * (m^3) * h * (V +VE -VNa ) +gK * (n^4)*(V+VE -VK) + gL*(V +VE -VL));

eq2 = (am * (1.0 - m) - bm * m);

eq3 = (ah * (1.0 - h) - bh * h);

eq4 = (an * (1.0 - n) - bn * n);

eqns=[eq1,eq2,eq3,eq4];

%写出m(V),h(V),n(V)的表达式
ans_m = solve(eq2,m);
ans_h = solve(eq3,h);
ans_n = solve(eq4,n);

%eq1s=subs(subs(subs(subs(eq1,m=ans_m),n=ans_n),h=ans_h),Iext=9.7);
%将eq1中的m,h,n变量替换
eq1F=subs(subs(subs(eq1,m=ans_m),n=ans_n),h=ans_h);


%构造雅各比矩阵
jv = jacobian(eq1,[V,m,h,n]);
jm = jacobian(eq2,[V,m,h,n]);
jh = jacobian(eq3,[V,m,h,n]);
jn = jacobian(eq4,[V,m,h,n]);

%以I为参数，查找分岔点，将1-200mA以0.1为间隔分成2000份
loop=2000;
I = (1:loop)*0.1;
Vstore = zeros(loop,1);
for i=1:(loop-1)
    %求解Ic时的平衡点
    t=subs(eq1F,Iext=I(i));
    ve = vpasolve(t,V);
    
    Vstore(i)=ve;
    
    me = solve(subs(eq2,V=ve),m);
    he = solve(subs(eq3,V=ve),h);
    ne = solve(subs(eq4,V=ve),n);
    
    me= eval(me);
    he=eval(he);
    ne=eval(ne);

    %代入雅各比矩阵
    ejm=eval(subs(subs(jm,m=me),V=ve));
    ejh=eval(subs(subs(jh,h=he),V=ve));
    ejn=eval(subs(subs(jn,n=ne),V=ve));
    ejv=eval(subs(subs(subs(subs(jv,V=ve),m=me),n=ne),h=he));

    j=[ejv;ejm;ejh;ejn];
    
    %求特征值
    E=eig(j);
    %disp(E);
    syms c0 c1 c2 c3;
    
    l1 = E(1);
    l2 = E(2);
    l3 = E(3) ;
    l4 =E(4);
    %disp(l1);
    %disp(l2);
    
    %求特征方程的系数
    eql1=l1^4 + c3 * l1^3 + c2 * l1^2 +c1 * l1 +c0;
    eql2=l2^4 + c3 * l2^3 + c2 * l2^2 +c1 * l2 +c0;
    eql3=l3^4 + c3 * l3^3 + c2 * l3^2 +c1 * l3 +c0;
    eql4=l4^4 + c3 * l4^3 + c2 * l4^2 +c1 * l4 +c0;
    eqln=[eql1,eql2,eql3,eql4];
    r = solve(eqln);

    %检查代数判据2、3
    d=abs(r.c1*r.c2*r.c3 - (r.c0*r.c3*r.c3+r.c1*r.c1));
    if d<0.1 && r.c0 >0 && r.c1>0 && r.c2>0 && r.c3 >0 
          disp(num2str(i));
    end
    
end
%plot(Vstore,I);
