import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

def extract_CV_from_voltage(voltage,step_t):
    spike_cv = np.zeros(np.shape(voltage)[0]) 
    isi_m = np.zeros(np.shape(voltage)[0]) 
    isi_v = np.zeros(np.shape(voltage)[0]) 
    for ii in np.arange(0,np.shape(voltage)[0],1):
        idx_spike = np.transpose(np.argwhere(voltage[ii,:]==30))
        idx_spike_diff = np.diff(idx_spike)*step_t
        if len(idx_spike_diff[0]):
            mean_isi = np.mean(idx_spike_diff) #mean ISI
            std_isi = np.std(idx_spike_diff)
            spike_cv[ii] = std_isi/mean_isi
            isi_m[ii]=mean_isi
            isi_v[ii]=np.var(idx_spike_diff)
        else:
            spike_cv[ii] = 1
    #return spike_cv,isi_m,isi_v
    return isi_m,isi_v

def calc_next_step_noise(Vm, I, step_t, remaining_refrac_time,sigma):
    Vl = -70
    #Gl = 0.025
    Gl = 0.05
    C = 0.5
    wn = np.random.normal()
    if Vm > -50 and Vm < 0: #threshold
        Vm = 30 #spike potential
    elif Vm > 0:
        Vm = -60 #reset potential
        remaining_refrac_time = remaining_refrac_time*0 + 2 #reset everything to 2
    elif remaining_refrac_time>0:
        Vm = -60 #reset potential
        remaining_refrac_time -= step_t
    else:
        Vm = Vm + step_t*(-Gl*(Vm-Vl) + I)/C + sigma*np.sqrt(step_t)*wn
    if remaining_refrac_time<0:
        remaining_refrac_time = remaining_refrac_time*0
    return Vm,remaining_refrac_time

def calc_next_step_noise_G(Vm, I, step_t, remaining_refrac_time,sigma,G):
    Vl = -70
    #Gl = 0.025
    Gl=G
    C = 0.5
    wn = np.random.normal()
    if Vm > -50 and Vm < 0: #threshold
        Vm = 30 #spike potential
    elif Vm > 0:
        Vm = -60 #reset potential
        remaining_refrac_time = remaining_refrac_time*0 + 2 #reset everything to 2
    elif remaining_refrac_time>0:
        Vm = -60 #reset potential
        remaining_refrac_time -= step_t
    else:
        Vm = Vm + step_t*(-Gl*(Vm-Vl) + I)/C + sigma*np.sqrt(step_t)*wn
    if remaining_refrac_time<0:
        remaining_refrac_time = remaining_refrac_time*0
    return Vm,remaining_refrac_time
""" 
step_t = 0.001
t = np.arange(0,500+step_t,step_t)
Vm_out = np.zeros(np.shape(t)[0])
I = 0.9

ind = 0
Vm_out[0]=-70 #init state
remaining_refrac_time = 0
for tt in t[0:-1]:
    Vm_out[ind+1],remaining_refrac_time = calc_next_step_noise(Vm_out[ind],I,step_t,remaining_refrac_time,1)
    ind += 1

isim,isiv=extract_CV_from_voltage(Vm_out,step_t)

plt.subplot(1,3,1)
plt.plot(t,Vm_out)
plt.xlabel("time /ms")
plt.ylabel("Memberance Potential /mV")

plt.subplot(1,3,2)
plt.plot(t,isim)

plt.subplot(1,3,3)
plt.plot(t,isiv)

plt.show() """

'''
run fixed s  for multi I
'''

def runManyI(s):
    Vm_out=[]
    step_t = 0.001
    t = np.arange(0,500+step_t,step_t)
    i = np.arange(0.0,1.0,0.01)
    S=s*0.1
    for I in i:
        t_Vm_out=np.zeros(np.shape(t)[0])
        t_Vm_out[0]=-70
        ind =0 
        remaining_refrac_time = 0
        for tt in t[0:-1]:
            t_Vm_out[ind+1],remaining_refrac_time = calc_next_step_noise(t_Vm_out[ind],I,step_t,remaining_refrac_time,S)
            ind += 1
        Vm_out.append(t_Vm_out)
    #return Vm_out
    V=np.array(Vm_out)
    isim,isiv=extract_CV_from_voltage(V,step_t)
    #plt.subplot(1,2,1)
    #plt.plot(i,isim)
    #plt.show()

    ws=np.full_like(i,S)
    tm = np.vstack((i,ws,isim))
    tv= np.vstack((i,ws,isiv))

    np.savetxt("m_s{s}.csv".format(s=str(s)),tm.T)
    np.savetxt("v_s{s}.csv".format(s=str(s)),tv.T)
    return isim,isiv,i


'''
run fixed I = 0.5 for multi sigma
'''
def runManyS(i):
    Vm_out=[]
    I= i*0.01
    step_t = 0.001
    t = np.arange(0,500+step_t,step_t)
    s = np.arange(0.0,2.0,0.05)
    for S in s:
        t_Vm_out=np.zeros(np.shape(t)[0])
        t_Vm_out[0]=-70
        ind =0 
        remaining_refrac_time = 0
        for tt in t[0:-1]:
            t_Vm_out[ind+1],remaining_refrac_time = calc_next_step_noise(t_Vm_out[ind],I,step_t,remaining_refrac_time,S)
            ind += 1
        Vm_out.append(t_Vm_out)
    #return Vm_out
    V=np.array(Vm_out)
    isim,isiv=extract_CV_from_voltage(V,step_t)
    #plt.subplot(1,2,1)
    #plt.plot(i,isim)
    #plt.show()
    #np.savetxt("m_s{s}.csv".format(s=str(s)),isim)
    #np.savetxt("v_s{s}.csv".format(s=str(s)),isiv)
    return isim,isiv,s

def runMany(s,i):
    Vm_out=[]
    I= i*0.01
    S=s*0.1
    step_t = 0.001
    t = np.arange(0,500+step_t,step_t)
    #s = np.arange(0.0,2.0,0.05)
    g = np.arange()
    for G in g:
        t_Vm_out=np.zeros(np.shape(t)[0])
        t_Vm_out[0]=-70
        ind =0 
        remaining_refrac_time = 0
        for tt in t[0:-1]:
            t_Vm_out[ind+1],remaining_refrac_time = calc_next_step_noise_G(t_Vm_out[ind],I,step_t,remaining_refrac_time,S,G)
            ind += 1
        Vm_out.append(t_Vm_out)
    #return Vm_out
    V=np.array(Vm_out)
    isim,isiv=extract_CV_from_voltage(V,step_t)
    #plt.subplot(1,2,1)
    #plt.plot(i,isim)
    #plt.show()
    #np.savetxt("m_s{s}.csv".format(s=str(s)),isim)
    #np.savetxt("v_s{s}.csv".format(s=str(s)),isiv)
    return isim,isiv,g

def plotfixS(s):
    isim,isiv,i= runManyI(s)
    plt.subplot(1,2,1)
    plt.xlabel("I/nA")
    plt.ylabel("E(ISI)")
    plt.title('E(ISI) - I')
    plt.plot(i,isim)

    plt.subplot(1,2,2)
    plt.xlabel("I/nA")
    plt.ylabel("Var(ISI)")
    plt.title('Var(ISI) - I')
    plt.plot(i,isiv)

    plt.show()


def plotfixI(i):
    isim,isiv,s= runManyS(i)
    plt.subplot(1,2,1)
    plt.xlabel("sigma")
    plt.ylabel("E(ISI)")
    plt.title('E(ISI) - sigma')
    plt.plot(s,isim)

    plt.subplot(1,2,2)
    plt.xlabel("sigma")
    plt.ylabel("Var(ISI)")
    plt.title('Var(ISI) - sigma')
    plt.plot(s,isiv)
    plt.show()
'''
run meshgrid for I and sigma
I 0.0 -> 1.0 ,0.01
sigma 0.0 -> 2.0, 0.1 
'''


if __name__ == '__main__':
    #s=np.arange(0.0,2.0,0.05)
    s=range(0,20)
    print(s)
    p=Pool(8)
    #for i in list(s):
    #    #print(type(i))
    #    p.apply(runManyI,(float(i),))
    p.map(runManyI,list(s))
    p.close()
    p.join()
