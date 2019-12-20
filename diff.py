import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solver_diff(
                InitU,    # initial condition u(x,0) = InitU(x)
                a,        # Diffusion coefficient alpha(x)
                f_src,    # Source Term
                X_max,    # maximum value of x
                T_max,    # maximum value of t
                Dparam,   # (Delta t)/(Delta x)^2
                Nx,       # step number of x
                u_Left,   # boundary condition u(0,t) = u_Left
                bnd_Left,
                u_Right,  # boundary condition u(X_max,t) = u_Right
                bnd_Right,
                theta=0.5,     # scheme
                ):

    # solve du/dt = d/dx (alpha(x)*du/dx) + f(x,t)

    # (1) prepare

    x = np.linspace(0, X_max, Nx+1)
    dx = x[1] - x[0]
    dt = Dparam * dx**2.0
    Nt = int(round(T_max/float(dt)))
    time = np.linspace(0, T_max, Nt+1)

    u_n1 =  np.zeros(Nx+1)  # u at t[n+1]
    u_n = np.zeros(Nx+1)   # u at t[n]

    Dl = 0.5 * Dparam * theta
    Dr = 0.5 * Dparam * (1-theta)

    A_diagonal = np.zeros(Nx+1)
    A_lower = np.zeros(Nx)
    A_upper = np.zeros(Nx)

    b = np.zeros(Nx+1)

    alpha = np.zeros(Nx+1)
    for i in range(Nx+1):
        alpha[i] = a(x[i])

    A_diagonal[1:-1] = 1.0 + Dl * (alpha[2:] + 2*alpha[1:-1] + alpha[:-2])
    A_lower[:-1] = -Dl * (alpha[1:-1] + alpha[:-2])
    A_upper[1:]  = -Dl * (alpha[2:] + alpha[1:-1])

    A_diagonal[0] = 1
    A_diagonal[Nx] = 1
    A_upper[0] = 0
    A_lower[-1] = 0

    A = diags(
        diagonals = [A_diagonal, A_lower, A_upper],
        offsets = [0, -1, 1],
        shape = (Nx+1, Nx+1),
        format = 'csr'
    )

    for i in range(0, Nx+1):
        u_n[i] = InitU(x[i])

    # (2) Loop in Time

    for n in range (0,Nt):

        # calculate b

        b[1:-1] = u_n[1:-1] + Dr*(
                          (alpha[2:] + alpha[1:-1])*(u_n[2:]-u_n[1:-1]) -
                          (alpha[1:-1] + alpha[0:-2])*(u_n[1:-1]-u_n[:-2])
                    ) + dt*(
                theta * f_src(x[1:-1], time[n]) + f_src(x[1:-1], time[n+1])
        )

        # boundary condition

        if (bnd_Left=='Dirichlet') and (bnd_Right=='Dirichlet'):
            b[0] = u_Left(time[n+1])
            b[-1] = u_Right(time[n+1])

        # solve Au=b

        u_n1[:] = spsolve(A, b)

        # update

        u_n, u_n1 = u_n1, u_n

    return x, u_n

def initCond(x):
    return 1.0

def diffCoef(x):
    return 1.0

def source(x,t):
    return 0.

def bnd_L_Dirichlet(t):
    return 0.

def bnd_R_Dirichlet(t):
    return 0.

Tlist = [1,10,100,200,500,1000]
LSlist = ['-','-','--','--','-.','-.',':',':']
LWlist = [3,2,3,2,3,2,3,2]
Clist = ['black','gray','black','gray','black','gray','black','gray']

plt.figure()

for n in range(len(Tlist)):

    test = solver_diff(initCond, diffCoef, source,
                       100, Tlist[n], 0.5, 100,
                       bnd_L_Dirichlet,'Dirichlet',bnd_R_Dirichlet,'Dirichlet')

    plt.plot(test[0],test[1],label="t="+str(Tlist[n]),ls=LSlist[n],lw=LWlist[n],c=Clist[n])

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("$u_t = u_{xx}$: $u(x,0)=1, u(0,t)=u(100,t)=0$")
plt.legend()
plt.show()