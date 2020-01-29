from diff import *

from math import *
import numpy as np
import matplotlib.pyplot as plt
import time
t1 = time.time()

# [r] in [0.1 Mpc]
# [t] in [10 Myr]
diff_scale = 3000. # from [r/t] to [10^29 cm^2/s] 

def bnd_L_Dirichlet(t):
    return 0.

def bnd_R_Dirichlet(t):
    return 0.

def bnd_L_Neumann():
    return 0.

def initCond(x):
    return 1.e-10

def diffCoef(x,t):
    x_core = 1.0
    x_proto = 100. - 0.8 * t
    D_core = 1.0e2    # [10^29 cm^2/s]
    D_proto = 10.0e2  # [10^29 cm^2/s]  
    D_bkg = 50.e2     # [10^29 cm^2/s]  
    tmp = D_core + D_proto / (1 + np.exp(-(x-x_core))) + D_bkg / (1 + np.exp(-(x-x_proto)))
    return tmp/diff_scale * x**2.0

def source(x,t):
    x_core = 1.
    x_proto = 100. - 0.8 * t
    sfr_core = 1.0
    sfr_proto = 1.e-4
    return sfr_proto * (1 + x/x_proto)**(-4.) + sfr_core * (1 + x/x_core)**(-4.)

def gamma(x):
    return 1./x**2.0

def n_gas(x,t): # [cm^-3]
    x_core = 1.
    x_proto = 100. - 0.8 * t
    n_core = 1.e-2
    n_proto = 1.e-4 
    n_igm = 1.e-7
    return n_igm + n_proto * (1 + x/x_proto)**(-2.0) + n_core * (1 + x/x_core)**(-2.0)

def coef_h(x,t):
    pp_rate = n_gas(x,t) * 3.e-26 * 3.e10 # [s^-1]
    t_pp = 3.15e-14/pp_rate # [10 Myr]
    return -1./t_pp

# calc

test = solver_diff(InitU=initCond,
                   a=diffCoef, g=gamma, f_h=coef_h, f_src=source,
                   X_min=0.1, X_max=5000, T_max=100, Dparam=0.5, Nx=12000,
                   u_Left=bnd_L_Neumann, bnd_Left='Neumann',
                   u_Right=bnd_R_Dirichlet, bnd_Right='Dirichlet')   

# plot

x_list = test[0]
u_list = test[1]
t_list = test[2]

fig = plt.figure(figsize=(15,13))
plt.subplots_adjust(hspace=0.1)

ax1 = fig.add_subplot(331)
ax2 = fig.add_subplot(332)
ax3 = fig.add_subplot(333)
ax4 = fig.add_subplot(334)
ax5 = fig.add_subplot(335)
ax6 = fig.add_subplot(336)
ax7 = fig.add_subplot(337)

clr = ['red','black', 'black', 'black', 'black', 'black', 'black', 'black', 'blue', 'blue', 'blue', 'blue', 'blue']
j=0

for i in range (0, len(t_list)):

    tmp = int(len(t_list)/12.0)

    if i%tmp == 0 and t_list[i]>0:
        ti = t_list[i]
        a = 1. / (1 + i/len(t_list))**1.5

        diff_coef = diffCoef(x_list,ti)/x_list**2.0
        sourc = [source(x,ti) for x in x_list]
        u_result = test[3][i]
        loss_rate = [-1./coef_h(x,ti) for x in x_list]
        nu = 4.0 * 3.1415 * x_list**3.0 * u_result / np.array(loss_rate)
        sour_tot = 4.0 * 3.1415 * x_list**3.0 * np.array(sourc)
        dudx_result = test[4][i]

        dx = test[5]
        dudx2 = np.append(0,np.diff(u_result)) / dx

        ax1.plot(x_list,u_result,c=clr[j],alpha=a)
        ax1.plot(x_list,dudx_result,c=clr[j],alpha=a,ls=':')
        ax2.plot(x_list,diff_coef*diff_scale,c=clr[j],label='t [10 Myr] :'+str(int(ti)),alpha=a)
        ax3.plot(x_list,sourc,c=clr[j],alpha=a)
        ax4.plot(x_list,sour_tot,c=clr[j],alpha=a)
        ax5.plot(x_list,loss_rate,c=clr[j],alpha=a)
        ax6.plot(x_list,nu,c=clr[j],alpha=a)

        ax7.plot(x_list,-dudx2,c=clr[j],alpha=a,ls=':')
        ax7.plot(x_list,-dudx_result,c=clr[j],alpha=a,ls='-')

        j +=1

#

ax1.set_ylabel("$n_p$(r,t)")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylim(2.e-6,1.)
ax1.set_xlim(1.,3000.)
ax1.set_title("${\partial n_p}/{\partial t} = r^{-2}{\partial}/{\partial r}[r^2D{\partial n_p}/{\partial r}] + S(r,t) - n_p/T_{pp}$") 

ax2.set_ylabel("$D(r,t)$ [10$^{29}$ cm$^2$/s]")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(1.,3000.)

ax3.set_ylabel("$S(r,t)$")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_ylim(1.e-6,0.9)
ax3.set_xlim(1.,3000.)

ax4.set_ylabel("$4\pi r^3S(r,t)$")
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_xlim(1.,3000.)

ax5.set_ylabel("$T_{pp}(r,t)$ [10 Myr]")
ax5.set_xscale("log")
ax5.set_yscale("log")
ax5.set_xlabel("r [0.1 Mpc]")
ax5.set_xlim(1.,3000.)

ax6.set_ylabel("4$\pi r^3n_p$ / $T_{pp}$")
ax6.set_xlabel("r [0.1 Mpc]")
ax6.set_xscale("log")
ax6.set_yscale("log")
ax6.set_ylim(1.e-8,1)
ax6.set_xlim(1.,3000.)

ax7.set_ylabel("-d$n_{p}$/dr")
ax7.set_xlabel("r [0.1 Mpc]")
ax7.set_xscale("log")
ax7.set_yscale("log")
ax7.set_xlim(1.,3000.)
ax7.set_ylim(1.e-8,1.)

t2 = time.time()
print(t2-t1)

ax2.legend(fontsize=7)

plt.savefig("fig_tmp.pdf")
