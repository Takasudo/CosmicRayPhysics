import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solver_diff(
                InitU,     # initial condition u(x,0) = InitU(x)
                a,         # Diffusion coefficient alpha(x)
                g,         # Coefficient gamma(x)
                f_h,       # Function h(x,t)
                f_src,     # Source Term
                X_min,     # minimum value of x
                X_max,     # maximum value of x
                T_max,     # maximum value of t
                Dparam,    # (Delta t)/(Delta x)^2
                Nx,        # step number of x
                u_Left,    # boundary condition u(X_min,t) = u_Left
                bnd_Left,
                u_Right,   # boundary condition u(X_max,t) = u_Right
                bnd_Right,
                theta=0.5  # scheme
                ):

    # solve du/dt = gamma(x) * d/dx (alpha(x,t)*du/dx) + f(x,t) + h(x,t) * u(x,t)

    x = np.linspace(X_min, X_max, Nx+1)
    dx = x[1] - x[0]
    dt = Dparam * dx**2.0
    Nt = int(round(T_max/float(dt)))
    time = np.linspace(0, T_max, Nt+1)

    u_n1 = np.zeros(Nx+1)   # u at t[n+1]
    u_n = np.zeros(Nx+1)    # u at t[n]

    Dl = 0.5 * Dparam * theta
    Dr = 0.5 * Dparam * (1-theta)

    A_diagonal = np.zeros(Nx+1)
    A_lower = np.zeros(Nx)
    A_upper = np.zeros(Nx)

    alpha = np.zeros(Nx+1)
    gamma = np.zeros(Nx+1)
    h = np.zeros(Nx+1)
    b = np.zeros(Nx+1)

    u_all = np.zeros((Nt+1, Nx+1))
    dudx_all = np.zeros((Nt+1, Nx+1))

    # initial condition

    for i in range(0, Nx+1):
        u_n[i] = InitU(x[i])

    # Loop in Time

    for n in range (0,Nt):

        # calculate alpha and gamma at time = t

        for i in range(Nx + 1):
            alpha[i] = a(x[i],time[n+1])
            gamma[i] = g(x[i])
            h[i] = f_h(x[i],time[n+1])

        # set A

        A_diagonal[1:-1] = 1.0 \
                           + Dl * gamma[1:-1] * (alpha[2:] + 2 * alpha[1:-1] + alpha[:-2]) \
                           - theta * dt * h[1:-1]
        A_lower[:-1] = -Dl * gamma[1:-1] * (alpha[1:-1] + alpha[:-2])
        A_upper[1:] = -Dl * gamma[1:-1] * (alpha[2:] + alpha[1:-1])

        if (bnd_Left == 'Dirichlet') and (bnd_Right == 'Dirichlet'):
            A_diagonal[0] = 1
            A_diagonal[Nx] = 1
            A_upper[0] = 0
            A_lower[-1] = 0
        elif (bnd_Left == 'Neumann') and (bnd_Right == 'Dirichlet'):
            A_diagonal[0] = 1
            A_diagonal[Nx] = 1
            A_upper[0] = -1
            A_lower[-1] = 0

        A = diags(
            diagonals=[A_diagonal, A_lower, A_upper],
            offsets=[0, -1, 1],
            shape=(Nx + 1, Nx + 1),
            format='csr'
        )

        # set b

        b[1:-1] = u_n[1:-1] \
                  + Dr * gamma[1:-1] * (
                          (alpha[2:] + alpha[1:-1])*(u_n[2:]-u_n[1:-1]) -
                          (alpha[1:-1] + alpha[0:-2])*(u_n[1:-1]-u_n[:-2])
                  ) \
                  + dt*(
                    (1-theta) * f_src(x[1:-1], time[n]) + theta * f_src(x[1:-1], time[n+1])
                    + (1-theta) * h[1:-1] * u_n[1:-1]
                  )

        if (bnd_Left=='Dirichlet') and (bnd_Right=='Dirichlet'):
            b[0] = u_Left(time[n+1])
            b[-1] = u_Right(time[n+1])
        elif (bnd_Left=='Neumann') and (bnd_Right=='Dirichlet'):
            b[0] = u_Left()
            b[-1] = u_Right(time[n+1])

        # solve Au=b to get u

        u_n1[:] = spsolve(A, b)

        # update

        u_n, u_n1 = u_n1, u_n
        u_all[n] += u_n

        # calc du/dx

        #dudx = np.diff(u_n) / dx
        #dudx_all[n] += np.append(0, dudx)

        dudx = (u_n[2:] - u_n[:-2]) / (2. * dx)
        dudx_tmp = np.append(0, dudx)
        dudx_all[n] += np.append(dudx_tmp, 0)
       
    u_all[Nt] += u_n

    return x, u_n, time, u_all, dudx_all, dx
