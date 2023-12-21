# Benney-Luke equations: a reduced water wave model
# =================================================
#
#
# .. rst-class:: emphasis
#
#     This tutorial was contributed by `Anna Kalogirou <mailto:Anna.Kalogirou@nottingham.ac.uk>`__
#     and `Onno Bokhove <mailto:O.Bokhove@leeds.ac.uk>`__.
#
#     The work is based on the article "Variational water wave
#     modelling: from continuum to experiment" by Onno Bokhove and Anna
#     Kalogirou :cite:`2015:lmscup`. The authors gratefully
#     acknowledge funding from EPSRC grant no. `EP/L025388/1
#     <http://gow.epsrc.ac.uk/NGBOViewGrant.aspx?GrantRef=EP/L025388/1>`__
#     with a link to the Dutch Technology Foundation STW for the project
#     "FastFEM: behavior of fast ships in waves".
#
# The Benney-Luke-type equations consist of a reduced potential flow water wave model based on the assumptions of small amplitude parameter :math:`\epsilon` and small dispersion parameter :math:`\mu` (defined by the square of the ratio of the typical depth over a horizontal length scale). They describe the deviation from the still water surface, :math:`\eta(x,y,t)`, and the free surface potential, :math:`\phi(x,y,t)`. A modified version of the Benney-Luke equations can be obtained by the variational principle:
#
# .. math::
#
#   0 &= \delta\int_0^T \int_{\Omega} \eta\phi_t - \frac{\mu}{2}\!\eta\Delta\phi_t + \frac{1}{2}\!\eta^2 + \frac{1}{2}\!\left(1+\epsilon\eta\right)\!\left|\nabla\phi\right|^2 + \frac{\mu}{3}\!\left( \Delta\phi \right)^2 \,dx\,dy\,dt \\
#     &= \delta\int_0^T \int_{\Omega} \eta\phi_t + \frac{\mu}{2}\nabla\eta\cdot\nabla\phi_t + \frac{1}{2}\!\eta^2 + \frac{1}{2}\!\left(1+\epsilon\eta\right)\!\left|\nabla\phi\right|^2 + \mu\left( \nabla q\cdot\nabla\phi - \frac{3}{4}q^2 \right) \,dx\,dy\,dt \\
#     &= \int_0^T \int_{\Omega} \left( \delta\eta\,\phi_t + \frac{\mu}{2}\nabla\delta\eta\cdot\nabla\phi_t + \eta\,\delta\eta + \frac{\epsilon}{2}\delta\eta\left|\nabla\phi\right|^2 \right) \\
#     & \qquad \qquad - \left( \delta\phi\,\eta_t + \frac{\mu}{2}\nabla\eta_t\cdot\nabla\delta\phi - \left(1+\epsilon\eta\right)\!\nabla\phi\cdot\nabla\delta\phi - \mu\nabla q\cdot\nabla\delta\phi \right) \\
#     & \qquad \qquad + \mu\left( \nabla\delta q \cdot\nabla\phi - \frac{3}{2}q\,\delta q  \right) \,dx\,dy\,dt,
#
# where the spatial domain is assumed to be :math:`\Omega` with natural boundary conditions, namely Neumann conditions on all the boundaries. In addition, suitable end-point conditions at :math:`t=0` and :math:`t=T` are used. Note that the introduction of the auxiliary function :math:`q` is performed in order to lower the highest derivatives. This is advantageous in a :math:`C^0` finite element formulation and motivated the modification of the "standard" Benney-Luke equations. The partial variations in the last line of the variational principle can be integrated by parts in order to get expressions that only depend on :math:`\delta\eta,\,\delta\phi,\,\delta q` and not their derivatives:
#
# .. math::
#
#   0 = \int_0^T \int_{\Omega} &\left( \phi_t - \frac{\mu}{2}\Delta\phi_t + \eta + \frac{\epsilon}{2}\left|\nabla\phi\right|^2 \right)\delta\eta \\
#                               - &\left( \eta_t - \frac{\mu}{2}\Delta\eta_t + \nabla\cdot\bigl(\left(1+\epsilon\eta\right)\!\nabla\phi\bigr)+\mu\Delta q \right)\delta\phi \\
#                               - &\mu\left( \Delta\phi + \frac{3}{2}q \right)\delta q \,dx\,dy\,dt.
#
# Since the variations :math:`\delta\eta,\,\delta\phi,\,\delta q` are arbitrary, the modified Benney-Luke equations then arise for functions :math:`\eta,\phi,q\in V` from a suitable function space :math:`V` and are given by:
#
# .. math::
#
#   \phi_t - \frac{\mu}{2}\Delta\phi_t + \eta + \frac{\epsilon}{2}\left|\nabla\phi\right|^2 &= 0 \\
#   \eta_t - \frac{\mu}{2}\Delta\eta_t + \nabla\cdot\bigl(\left(1+\epsilon\eta\right)\!\nabla\phi\bigr)+\mu\Delta q &= 0 \\
#   q &= - \frac{2}{3}\Delta\phi.
#
# We can either directly use the partial variations in the variational principle above (last line) as the fundamental weak formulation (with :math:`\delta\phi,\, \delta\eta,\, \delta q` playing the role of test functions), or multiply the equations by a test function :math:`v\in V` and integrate over the domain in order to obtain a weak formulation in a classic manner
#
# .. math::
#
#   \int_{\Omega} \phi_t\,v + \frac{\mu}{2}\nabla\phi_t\cdot\nabla v + \eta\,v + \frac{\epsilon}{2}\nabla\phi\cdot\nabla\phi\,v \,dx\,dy &= 0 \\
#   \int_{\Omega} \eta_t\,v + \frac{\mu}{2}\nabla\eta_t\cdot\nabla v - \left(1+\epsilon\eta\right)\nabla\phi\cdot\nabla v - \mu\nabla q\cdot\nabla v \,dx\,dy &= 0 \\
#   \int_{\Omega} q\,v - \frac{2}{3}\nabla\phi\cdot\nabla v \,dx\,dy &= 0.
#
# Note that the Neumann boundary conditions have been used to remove every surface term that resulted from the integration by parts. Moreover, the variational form of the system requires the use of a symplectic integrator for the time-discretisation. Here we choose the 2nd-order Stormer-Verlet scheme :cite:`2006:SV`, which requires two half-steps to update :math:`\phi` in time (one implicit and one explicit in general) and one (implicit) step for :math:`\eta`:
#
# .. math::
#
#   \int_{\Omega} \frac{\phi^{n+1/2}-\phi^n}{\frac{1}{2}\!dt}\,v + \frac{\mu}{2}\nabla\left(\frac{\phi^{n+1/2}-\phi^n}{\frac{1}{2}\!dt}\right)\cdot\nabla v + \eta^n\,v + \frac{\epsilon}{2}\nabla\phi^{n+1/2}\cdot\nabla\phi^{n+1/2}\,v \,dx\,dy &= 0 \\
#   \int_{\Omega} q^{n+1/2}\,v - \frac{2}{3}\nabla\phi^{n+1/2}\cdot\nabla v \,dx\,dy &= 0 \\
#   \int_{\Omega} \frac{\eta^{n+1}-\eta^n}{dt}\,v + \frac{\mu}{2}\nabla\left(\frac{\eta^{n+1}-\eta^n}{dt}\right)\cdot\nabla v - \frac{1}{2}\Bigl( \left(1+\epsilon\eta^{n+1}\right) + \left(1+\epsilon\eta^n\right) \Bigr)\nabla\phi^{n+1/2}\cdot\nabla v - \mu\nabla q^{n+1/2}\cdot\nabla v \,dx\,dy &= 0 \\
#   \int_{\Omega} \frac{\phi^{n+1}-\phi^{n+1/2}}{\frac{1}{2}\!dt}\,v + \frac{\mu}{2}\nabla\left(\frac{\phi^{n+1}-\phi^{n+1/2}}{\frac{1}{2}\!dt}\right)\cdot\nabla v + \eta^{n+1}\,v + \frac{\epsilon}{2}\nabla\phi^{n+1/2}\cdot\nabla\phi^{n+1/2}\,v \,dx\,dy &= 0 \\
#   \int_{\Omega} q^{n+1}\,v - \frac{2}{3}\nabla\phi^{n+1}\cdot\nabla v \,dx\,dy &= 0.
#
# Furthermore, we note that the Benney-Luke equations admit asymptotic solutions (correct up to order :math:`\epsilon`). The "exact" solutions can be found by assuming one-dimensional travelling waves of the type
#
# .. math::
#
#   \eta(x,y,t) = \eta(\xi,\tau),\quad \phi(x,y,t) = \Phi(\xi,\tau), \qquad \text{with} \qquad \xi = \sqrt{\frac{\epsilon}{\mu}}(x-t), \quad \tau = \epsilon\sqrt{\frac{\epsilon}{\mu}}t, \quad \Phi = \sqrt{\frac{\epsilon}{\mu}}\phi.
#
# The Benney-Luke equations then become equivalent to a Korteweg-de Vries (KdV) equation for :math:`\eta` at leading order in :math:`\epsilon`. The soliton solution of the KdV :cite:`1989:KdV` travels with speed :math:`c` and is reflected when reaching the solid wall. The initial propagation before reflection matches the asymptotic solution for the surface elevation :math:`\eta` well. The asymptotic solution for the surface potential :math:`\phi` can be found by using :math:`\eta=\phi_{\xi}` (correct at leading order), giving
#
# .. math::
#
#   \eta(x,y,t) &= \frac{c}{3}{\rm sech}^2 \left( \frac{1}{2}\sqrt{\frac{c\epsilon}{\mu}} \left(x-x_0-t-\frac{\epsilon}{6}ct\right) \right), \\
#   \phi(x,y,t) &= \frac{2}{3}\sqrt{\frac{c\mu}{\epsilon}}\,\left( {\rm tanh}\left(\frac{1}{2}\sqrt{\frac{c\epsilon}{\mu}} \left(x-x_0-t-\frac{\epsilon}{6}ct\right) \right)+1 \right).
#
# Finally, before implementing the problem in Firedrake, we calculate the total energy defined by the sum of potential and kinetic energy. The system is then stable if the energy is bounded and shows no drift. The expression for total energy is given by:
#
# .. math::
#
#   E(t) = \int_{\Omega} \frac{1}{2}\eta^2 + \frac{1}{2}\!\left(1+\epsilon\eta\right)\left|\nabla\phi\right|^2 + \mu\left( \nabla q\cdot \nabla\phi - \frac{3}{4}q^2 \right) \,dx\,dy.
#
# The implementation of this problem in Firedrake requires solving two nonlinear variational problems and one linear problem. The Benney-Luke equations are solved in a rectangular domain :math:`\Omega=[0,10]\times[0,1]`, with :math:`\mu=\epsilon=0.01`, time step :math:`dt=0.005` and up to the final time :math:`T=2.0`. Additionally, the domain is split into 50 cells in the x-direction using a quadrilateral mesh. In the y-direction only 1 cell is enough since there are no variations in y::

from firedrake import *
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import os.path
#import os ONNO 11-02-2023: no idea about opnumthreats warning?
os.environ["OMP_NUM_THREADS"] = "1"
# dask-worker --nthreads 1
# export OMP_NUM_THREADS=1

# Settings to reproduce Fig. 3 in Bokhove and Kalogirou 2016::
# Now we move on to defining parameters::
T = 20.0
T = 9.5 #  2.0
T = 10.0
dt = 0.005 # 0.005
Lx = 10
Nx = 200  # 50
Ny = 1
c = 1.5 #  1.0
mu = 0.0 # shallow-water limit
epsilon = 0.01 # BL test BK2016
epsilon = 0.11 # shallow-water limit 0.09 starts to steepen after reflection; more so for 0.11
musol = 0.01
mu = 0.01 # BL test BK2016
mu = 0.0
#  epsilon = 0.5 # BL test BK2016

# Settings used in initial condition of soliton
if mu==0.0: # shallow-water limit
    if epsilon==0:
        muu = musol
    else:
        muu = epsilon
        muu = musol
else: # BL test BK2016
    muu = mu

if epsilon==0.0: # shallow-water limit
    if mu==0.0:
        eps = musol
    else:
        eps = mu
else: # BL test BK2016
    eps= epsilon

# nVP=0 BLE_SV Stormer-Verlet with explicit weak forms; nVP=1: BLE-MMP modified midpoint with weak forms derived from relevant V
# nVP=2 SWE-MMP with q-terms explicitly removed (MMP-BLE for mu=eps=0 crashes)
nVP = 2
print(' nVP, eps, mu, epsilon, muu:',nVP,eps,muu,epsilon,mu)

# Settings for measurements/storage of data::


nmea = 11
tmeas = [0,2,4,6,8,10,12,14,16,18,20,22]
nmea = 10 # number of outputs
tmeas = [0.0,2.0,4.0,5.0,5.5,6.0,6.5,7.0,8.5,9.5,10] # output times

nnm = 0  # counter outputs
dte = 0.1
tmE = 0.0
smallfac = 10.0**(-10.0)

m = UnitIntervalMesh(Nx)
mesh = ExtrudedMesh(m, layers=Ny)
coords = mesh.coordinates
coords.dat.data[:,0] = Lx*coords.dat.data[:,0]

# The function space chosen consists of degree 2 continuous Lagrange polynomials, and the functions :math:`\eta,\,\phi` are initialised to take the exact soliton solutions for :math:`t=0`, centered around the middle of the domain, i.e. with :math:`x_0=\frac{1}{2}L_x`::

nCG = 1
V = FunctionSpace(mesh,"CG",nCG)
nCG1 = 1
V1 = FunctionSpace(mesh,"CG",nCG1)
DG0 = FunctionSpace(mesh, "DQ", 0)
eta0 = Function(V, name="eta")
phi0 = Function(V, name="phi")

#nuf = Function(DG0, name="nuf")
#nuf1 = Function(DG0, name="nuf1")
#nudam = Function(DG0, name="nudam")
#nudamm = Function(DG0, name="nudamm")

nuf = Function(V1, name="nuf")
nuf1 = Function(V1, name="nuf1")
nudam = Function(V1, name="nudam")
nudamm = Function(V1, name="nudamm")

mub = Function(DG0, name="mub")
mub1 = Function(DG0, name="mub1")
eta1 = Function(V, name="eta_next")
phi1 = Function(V, name="phi_next")
q1 = Function(V)
phi_h = Function(V)
q_h = Function(V)
ex_eta = Function(V, name="exact_eta")
ex_phi = Function(V, name="exact_phi")

# Variables for modified midpoint waves
if nVP==1:
    mixed_Vmp = V * V * V
    result_mixedmp = Function(mixed_Vmp)
    vvmp = TestFunction(mixed_Vmp)
    vvmphi, vvmpeta, vvmpq = split(vvmp)  # These represent "blocks".
    phimp, etamp, qmp= split(result_mixedmp)
elif nVP==2:
    mixed_Vmp = V * V
    result_mixedmp = Function(mixed_Vmp)
    vvmp = TestFunction(mixed_Vmp)
    vvmphi, vvmpeta = split(vvmp)  # These represent "blocks".
    phimp, etamp = split(result_mixedmp)
# end if  

q = TrialFunction(V)
v = TestFunction(V)

x = SpatialCoordinate(mesh)
# JUNHO
nx=100
xsmall = 0.0*10**(-6)
xvals = np.linspace(0+xsmall, Lx-xsmall, nx)
yslice = 0.0
# JUNHO END
x0 = 0.5 * Lx
x1 = 1.5 * Lx
# Exact-asymptotic sech-soliton solution as initial conditions, sum of incoming and mirror wave, which need to be updated at every time-step::
eta0.interpolate( 1/3.0*c*pow(cosh(0.5*sqrt(c*eps/muu)*(x[0]-x0)),-2) ) # + 1/3.0*c*pow(cosh(0.5*sqrt(c*epsilon/muu)*(x[0]-x1)),-2) )
phi0.interpolate( 2/3.0*sqrt(c*muu/eps)*(tanh(0.5*sqrt(c*eps/muu)*(x[0]-x0))+1) )
nuf.interpolate(0.0*x[0])  # Initialise to 0
mub.interpolate(0.0*x[0])  # Initialise to 0
nudamm.interpolate( 2/3.0*sqrt(c*muu/eps)*(tanh(0.5*sqrt(c*eps/muu)*(x[0]-x0))+1) )


fig, (ax1, ax2, ax3) = plt.subplots(3)
tsize = 14
# ax2.set_title(r'$\phi$ value in $x$ direction',fontsize=tsize)
# ax1.set_title(r'$\eta$ value in $x$ direction',fontsize=tsize)
ax1.set_title(r'$t=0,2,4,5,5.5,6,6.5,8.5,9.5$',fontsize=tsize)
ax1.set_ylabel(r'$\eta(x,t)$ ',fontsize=tsize)
ax1.grid()
ax2.set_xlabel(r'$x$ ',fontsize=tsize)
ax2.set_ylabel(r'$\phi (x,t)$ ',fontsize=tsize)
if nVP==2:
    ax3.set_xlabel(r'$x$ ',fontsize=tsize)
    ax3.set_ylabel(r'$\nu_b(x,t)$ ',fontsize=tsize)
ax2.grid()
eta12 = np.array([eta0.at(x,yslice) for x in xvals]) # eta12 = np.array([eta0.at(x) for x in xvals])
phi12 = np.array([phi0.at(x,yslice) for x in xvals])
ax1.plot(xvals,eta12) #, color[int(i-1)],label = f' $\eta_n: t = {t:.3f}$')
ax2.plot(xvals,phi12) #, color[int(i-1)], label = f' $\phi_n: t = {t:.3f}$')


if nVP==0:
    #
    # Firstly, :math:`\phi` is updated to a half-step value using a nonlinear variational solver to solve the implicit equation::
    # 
    Fphi_h = ( v*(phi_h-phi0)/(0.5*dt) + 0.5*mu*inner(grad(v),grad((phi_h-phi0)/(0.5*dt)))
               + v*eta0 + 0.5*epsilon*inner(grad(phi_h),grad(phi_h))*v )*dx
    phi_problem_h = NonlinearVariationalProblem(Fphi_h,phi_h)
    phi_solver_h = NonlinearVariationalSolver(phi_problem_h)
    # followed by a calculation of a half-step solution :math:`q`, performed using a linear solver::
    aq = v*q*dx
    Lq_h = 2.0/3.0*inner(grad(v),grad(phi_h))*dx
    q_problem_h = LinearVariationalProblem(aq,Lq_h,q_h)
    q_solver_h = LinearVariationalSolver(q_problem_h)
    # Then the nonlinear implicit equation for :math:`\eta` is solved::
    Feta = ( v*(eta1-eta0)/dt + 0.5*mu*inner(grad(v),grad((eta1-eta0)/dt))
             - 0.5*((1+epsilon*eta0)+(1+epsilon*eta1))*inner(grad(v),grad(phi_h))
             - mu*inner(grad(v),grad(q_h)) )*dx
    solparams={ 'pc_type': 'python', 'pc_python_type': 'firedrake.ASMStarPC','star_sub_sub_pc_type': 'lu', 'sub_sub_pc_factor_mat_ordering_type': 'rcm'}
    eta_problem = NonlinearVariationalProblem(Feta,eta1)
    eta_solver = NonlinearVariationalSolver(eta_problem, solver_parameters=solparams)
    # and finally the second half-step (explicit this time) for the equation of :math:`\phi` is performed and :math:`q` is computed for the updated solution::
    Fphi = ( v*(phi1-phi_h)/(0.5*dt) + 0.5*mu*inner(grad(v),grad((phi1-phi_h)/(0.5*dt)))
             + v*eta1 + 0.5*epsilon*inner(grad(phi_h),grad(phi_h))*v )*dx
    phi_problem = NonlinearVariationalProblem(Fphi,phi1)
    solver_parameters_lin={'ksp_type': 'cg', 'pc_type': 'none'}
    phi_solver = NonlinearVariationalSolver(phi_problem, solver_parameters=solparams)
    Lq = 2.0/3.0*inner(grad(v),grad(phi1))*dx
    q_problem = LinearVariationalProblem(aq,Lq,q1)
    q_solver = LinearVariationalSolver(q_problem,solver_parameters=solver_parameters_lin)
elif nVP==1:
    # Alternatively, define the modified-midpoint time-discrete variational principle (VP) for Benney-Luke system; VP is expression (32a) in Bokhove & Kalogirou 2016:
    VPbl = ( inner(etamp,(phi1-phi0)/dt) - inner(phimp, (eta1 - eta0)/dt) \
             + 0.5*mu*inner(grad(etamp),grad((phi1-phi0)/dt)) - 0.5*mu*inner(grad(phimp),grad((eta1-eta0)/dt)) \
             + 0.5*inner(etamp,etamp) + 0.5*(1.0+epsilon*etamp)*inner(grad(phimp),grad(phimp)) \
             + mu*(inner(grad(qmp),grad(phimp)) - 0.75*inner(qmp,qmp) ) ) * dx
    # To obtain 3 weak forms, take the 3 functional derivative (no partial integration/divergence theorem in space) wrt 3 midpoint n+1/2-time variables
    # Replace phi1 and eta1 by their expressions ito phi0,eta0, phimp and etamp (mp=midpoint ie at n+1/2 time)
    phiexprnl1 = derivative(VPbl, phimp, du=vvmphi)
    phiexprnl1 = replace(phiexprnl1, {phi1: 2.0*phimp-phi0})
    phiexprnl1 = replace(phiexprnl1, {eta1: 2.0*etamp-eta0})
    
    etaexprnl1 = derivative(VPbl, etamp, du=vvmpeta)
    etaexprnl1 = replace(etaexprnl1, {phi1: 2.0*phimp-phi0})
    etaexprnl1 = replace(etaexprnl1, {eta1: 2.0*etamp-eta0})
    
    qexprnl = derivative(VPbl, qmp, du=vvmpq)
    qexprnl = replace(qexprnl, {phi1: 2.0*phimp-phi0})
    qexprnl = replace(qexprnl, {eta1: 2.0*etamp-eta0})
    # Combine the 3 weak formas and solve as coupled system using mix variables function space
    Fexprnl = phiexprnl1 + etaexprnl1 + qexprnl
    parampsi    = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    solparams={ 'pc_type': 'python', 'pc_python_type': 'firedrake.ASMStarPC','star_sub_sub_pc_type': 'lu', 'sub_sub_pc_factor_mat_ordering_type': 'rcm'}
    
    # phicombonl = NonlinearVariationalSolver(NonlinearVariationalProblem(Fexprnl, result_mixedmp), solver_parameters=parampsi)
    phicombonl = NonlinearVariationalSolver(NonlinearVariationalProblem(Fexprnl, result_mixedmp), solver_parameters=solparams)
elif nVP==2:
    # Alternatively, define the modified-midpoint time-discrete variational principle (VP) for SWE system; VP is expression above with mu=0 and q removed:
    VPbl = ( inner(etamp,(phi1-phi0)/dt) - inner(phimp,(eta1 - eta0)/dt) \
             + 0.5*mu*inner(grad(etamp),grad((phi1-phi0)/dt)) \
             + 0.5*inner(etamp,etamp) + 0.5*(1.0+epsilon*etamp)*inner(grad(phimp),grad(phimp)) ) * dx

    H0 = 1
    Ls = 10
    betareal = 0.125
    alphareal = 0.6
    gg = 9.81 # m/s^2
    kk = 2
    nub0 = 1.86/(np.sqrt(gg*H0)*H0) # m^2/s phenomenological value
    betawb = betareal/epsilon
    alphas = alphareal/(epsilon) 
    nonno = 0
    epss = 10**(-5)
    if nonno==1:
        nudam = Function(DG0).interpolate(conditional( inner(grad(etamp),grad(etamp)) < betawb**2, 0.0, nub0 )) # Does not work dissipation choice 1
        # nudam = Function(DG0).interpolate(conditional( inner(grad(etamp),grad(etamp)) < betawb**2, 0.0, nub0 )) # Does not work dissipation choice 1
    elif nonno==3: # works a bit better 
        #  nuf.interpolate( conditional( gt((2*etamp-2*eta0)/dt , alphas*sqrt(1+epsilon*eta0)),0.0,0.0 ) ) # Initialise to 0 ; deta/dt > sqrtgh
        nuf1.interpolate( conditional( lt(inner(grad(etamp),grad(etamp)), betawb**2), 0.0, nub0 ) + nuf*conditional( gt((2*etamp-2*eta0)/dt,0),1.0,0.0 ) ) # Does
        nudam.interpolate( conditional( gt(nuf1,epss), nub0, nuf1) )
    elif nonno==0: # works best atm mub1 is DG0
        # nuf.interpolate( conditional( gt((2*etamp-2*eta0)/dt , alphas*sqrt(1+epsilon*eta0)),0.0,0.0 ) ) # Initialise to 0 ; deta/dt > sqrtgh
        mub1.interpolate( conditional( lt(inner(grad(etamp),grad(etamp)), betawb**2), 0.0, nub0 ) + nuf*conditional( gt((2*etamp-2*eta0)/dt,0),4.0,0.0 ) ) # Does
        nudam.interpolate( conditional( gt(mub1,epss), nub0, mub1) )
        # nudam.interpolate( conditional( lt(inner(grad(etamp),grad(etamp)), betawb**2), 0.0, nub0 ) ) # Does something dissipation choice 1
    elif nonno==4: # works but not best now mub1 abd mub are DG0
        # nuf.interpolate( conditional( gt((2*etamp-2*eta0)/dt , alphas*sqrt(1+epsilon*eta0)),0.0,0.0 ) ) # Initialise to 0 ; deta/dt > sqrtgh
        mub1.interpolate( conditional( lt(inner(grad(etamp),grad(etamp)), betawb**2), 0.0, nub0 ) + mub*conditional( gt((2*etamp-2*eta0)/dt,0),1.0,0.0 ) ) # Does
        nudam.interpolate( conditional( gt(mub1,epss), nub0, mub1) )
    else: # Dissipation choice 2
        nuf.interpolate( conditional( gt((2*etamp-2*eta0)/dt , alphas*sqrt(1+epsilon*eta0)),0.0,0.0 ) ) # Initialise to 0 ; deta/dt > sqrtgh 
        # mub.interpolate( conditional( gt(-(2*etamp-2*eta0)/dt , alphas*sqrt(1+epsilon*eta0)),0.0,0.0 ) ) # Initialise to 0 ; -deta/dt-sqrtgh >0 ; deta/dt < -sqrtgh
        nuf1.interpolate( conditional( gt((2*etamp-2*eta0)/dt , alphas*sqrt(1+epsilon*etamp)),nub0, 0.0*nub0 ) + nuf*conditional( gt((2*etamp-2*eta0)/dt,0),1.0,0.0 ) )
        # mub1.interpolate( conditional( lt((2*etamp-2*eta0)/dt , -alphas*sqrt(1+epsilon*etamp)),nub0, 0.0*nub0 ) + mub*conditional( lt((2*etamp-2*eta0)/dt,0),1.0,0.0 ) )
        nudam.interpolate( conditional( gt(nuf1,nub0), nub0, nuf1) )

    # To obtain 3 weak forms, take the 3 functional derivative (no partial integration/divergence theorem in space) wrt 3 midpoint n+1/2-time variables
    # Replace phi1 and eta1 by their expressions ito phi0,eta0, phimp and etamp (mp=midpoint ie at n+1/2 time)
    phiexprnl1 = derivative(VPbl, phimp, du=vvmphi) # evolution eqn for eta! Notice the minus sign: (-(eta1 - eta0)/dt + ... - nubdamp * (inner(grad(vvmphi),grad(etamp))) ) * dx
    phiexprnl1 = replace(phiexprnl1, {phi1: 2.0*phimp-phi0})
    phiexprnl1 = replace(phiexprnl1, {eta1: 2.0*etamp-eta0})
    etaweakfwavebr = nudam * ( inner(grad(vvmphi),grad(etamp)) ) * dx
 
    etaexprnl1 = derivative(VPbl, etamp, du=vvmpeta) # evolution eqn for phi! ( (phi1-phi0)/dt + ... + nubdamp * ( inner(grad(vvmpeta),grad(phimp)) ) ) * dx
    etaexprnl1 = replace(etaexprnl1, {phi1: 2.0*phimp-phi0})
    etaexprnl1 = replace(etaexprnl1, {eta1: 2.0*etamp-eta0})
    phiweakfwavebr = nudam * ( inner(grad(vvmpeta),grad(phimp)) ) * dx

    # Combine the 2 weak forms and solve as coupled system using mix variables function space
    Fexprnl = (phiexprnl1 - etaweakfwavebr) + (etaexprnl1 + phiweakfwavebr)
    parampsi    = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    solparams={ 'pc_type': 'python', 'pc_python_type': 'firedrake.ASMStarPC','star_sub_sub_pc_type': 'lu', 'sub_sub_pc_factor_mat_ordering_type': 'rcm'}
    
    phicombonl = NonlinearVariationalSolver(NonlinearVariationalProblem(Fexprnl, result_mixedmp), solver_parameters=solparams)
        
else:
    VPSVa = 0.0

# What is left before iterating over all time steps, is to find the initial energy :math:`E_0`, used later to evaluate the energy difference :math:`\left|E-E_0\right|/E_0`::

save_path = 'data/BLf/'
output = File(os.path.join(save_path, "output.pvd")) # OLD:  output = File('output.pvd')
fileE = "data/BLf/energy.txt"
outputE = open(fileE,"w")
fileetaphi12 = "data/BLf/etaphi12.txt"
outputetaphi12 = open(fileetaphi12,"w")

t = 0
E0 = assemble( (0.5*eta0**2 + 0.5*(1+epsilon*eta0)*abs(grad(phi0))**2 + mu*(inner(grad(q1),grad(phi0)) - 0.75*q1**2))*dx )
E = E0
print(t, abs((E-E0)/E0))
tmE = tmE+dte
          
# Define exact solution, sum of incoming and mirror waves, which need to be updated at every time-step::

t_ = Constant(t)
expr_eta = 1/3.0*c*pow(cosh(0.5*sqrt(c*eps/muu)*(x[0]-x0-t_-eps*c*t_/6.0)),-2)+ 1/3.0*c*pow(cosh(0.5*sqrt(c*eps/muu)*(x[0]-x1+t_+eps*c*t_/6.0)),-2)
expr_phi = 2/3.0*sqrt(c*muu/eps)*(tanh(0.5*sqrt(c*eps/muu)*(x[0]-x0-t_-eps*c*t_/6.0))+1)

# Since we will interpolate these values again and again, we use an
# :class:`~.Interpolator` whose :meth:`~.Interpolator.interpolate`
# method we can call to perform the interpolation. ::

eta_interpolator = Interpolator(expr_eta, ex_eta)
phi_interpolator = Interpolator(expr_phi, ex_phi)
phi_interpolator.interpolate()
eta_interpolator.interpolate()

# For visualisation, we save the computed and exact solutions to an output file.
# Note that the visualised data will be interpolated from piecewise quadratic functions to piecewise linears::
number_mesh=len(eta0.dat.data)
eta_data=np.zeros((number_mesh,nmea+1))
phi_data=np.zeros((number_mesh,nmea+1))

if t==tmeas[nnm]:
    if nVP==2:
        nudamm.interpolate(nudam)
        output.write(phi0, eta0, ex_phi, ex_eta, nudamm, time=t)
    else:
        output.write(phi0, eta0, ex_phi, ex_eta, time=t)
    eta_data[:,nnm]=eta0.dat.data
    phi_data[:,nnm]=phi0.dat.data
    nnm = nnm+1

# We are now ready to enter the main time iteration loop::


tic = time.time()
while t <= T:
      t += dt

      t_.assign(t)

      eta_interpolator.interpolate()
      phi_interpolator.interpolate()

      if nVP==0:
          phi_solver_h.solve()
          q_solver_h.solve()
          eta_solver.solve()
          phi_solver.solve()
          q_solver.solve()
          eta0.assign(eta1)
          phi0.assign(phi1)
          if t>tmE+smallfac:
              E = assemble( (0.5*eta1**2 + 0.5*(1+epsilon*eta1)*abs(grad(phi1))**2
                             + mu*(inner(grad(q1),grad(phi1)) - 0.75*q1**2))*dx )
              print(t, abs((E-E0)/E0))
              print("%.11f %.11f" % (t, abs((E-E0)/E0)),file=outputE)
              tmE= tmE+dte
      elif nVP==1:
          phicombonl.solve()
          phimp, etamp, qmp = result_mixedmp.split()
          # Copy update at time level n+1 (really phi1, eta1) into phi0 and eta0 for next time step where it becomes time level n::
          
          phi0.interpolate(2.0*phimp-phi0)
          eta0.interpolate(2.0*etamp-eta0)
          if t>tmE+smallfac:
              E = assemble( (0.5*etamp**2 + 0.5*(1+epsilon*etamp)*abs(grad(phimp))**2
                             + mu*(inner(grad(qmp),grad(phimp)) - 0.75*qmp**2))*dx )
              print(t, abs((E-E0)/E0))
              print("%.11f %.11f  %.11f %.11f" % (t, abs((E-E0)/E0), E, E0),file=outputE)
              tmE= tmE+dte
      elif nVP==2:
          #Fexprnl = (phiexprnl1 - etaweakfwavebr) + (etaexprnl1 + phiweakfwavebr)
          #phicombonl = NonlinearVariationalSolver(NonlinearVariationalProblem(Fexprnl, result_mixedmp), solver_parameters=solparams)    
          #nubdamp = conditional( inner(grad(eta0),grad(eta0)) < betawb**2, 0.0*nub0 + 0.0*x[0], nub0 + 0.0*x[0] ) # OB: Error? I don't think Firedrake makes the update every time step.
          phicombonl.solve()
          phimp, etamp = result_mixedmp.split()
          # Copy update at time level n+1 (really phi1, eta1) into phi0 and eta0 for next time step where it becomes time level n::
           
          if nonno==1:
              nudam = Function(DG0).interpolate(conditional( inner(grad(etamp),grad(etamp)) < betawb**2, 0.0*nub0, nub0))
              # nudam = Function(DG0).interpolate(conditional( inner(grad(etamp),grad(etamp)) < betawb**2, 0.0*nub0, nub0))
              nudamm.interpolate(nudam) # Put into display variable for visualisation
          elif nonno==3: # works a bit better all CG1
              nuf1.interpolate( conditional( lt(inner(grad(etamp),grad(etamp)), betawb**2), 0.0, nub0 ) + nuf*conditional( gt((2*etamp-2*eta0)/dt,0),1.0,0.0 ) ) # Does
              nudam.interpolate( conditional( gt(nuf1,epss), nub0, nuf1) )
              nudamm.interpolate(nudam)
              nuf = nudam
              # nudam.interpolate( conditional( lt(inner(grad(etamp),grad(etamp)), betawb**2), 0.0, nub0 ))
              # Not sure what lt does: nudam.interpolate( conditional( (inner(grad(etamp),grad(etamp)), betawb**2), 0.0, nub0 ))
          elif nonno==0: # works best atm with mub1 DG0 and nuf CG1, nudam CG1
              mub1.interpolate( conditional( lt(inner(grad(etamp),grad(etamp)), betawb**2), 0.0, nub0 ) + nuf*conditional( gt((2*etamp-2*eta0)/dt,0),4.0,0.0 ) ) # Does
              nudam.interpolate( conditional( gt(mub1,epss), nub0, mub1) )
              nudamm.interpolate(nudam)
              nuf = nudam
          elif nonno==4:  #  works mub1 and mub are DG0 and nudam CG1
              mub1.interpolate( conditional( lt(inner(grad(etamp),grad(etamp)), betawb**2), 0.0, nub0 ) + mub*conditional( gt((2*etamp-2*eta0)/dt,0),1.0,0.0 ) ) # Does
              nudam.interpolate( conditional( gt(mub1,epss), nub0, mub1) )
              mub = mub1
              nudamm.interpolate(nudam)
              # nudamm.interpolate( conditional( gt(nuf1,epss), nub0, nuf1) ) # Put into display variable for visualisation
              # nudamm.interpolate(nuf1) # Put into display variable for visualisation
          else: #
              nuf1.interpolate( conditional( gt((2*etamp-2*eta0)/dt , alphas*sqrt(1+epsilon*etamp)),nub0, 0.0*nub0 ) + nuf*conditional( gt((2*etamp-2*eta0)/dt,0),1.0,0.0 ) )
              #mub1.interpolate( conditional( lt((2*etamp-2*eta0)/dt , -alphas*sqrt(1+epsilon*etamp)),nub0, 0.0*nub0 ) + mub*conditional( lt((2*etamp-2*eta0)/dt,0),1.0,0.0 ) )
              nudam.interpolate( conditional( gt(nuf1,nub0), nub0, nuf1) )
              nudamm.interpolate(nudam) # Put into display variable for visualisation
              nuf = nuf1 # ISSUE: is this updating the "nuf"used in the VP==2 definition of the weak forms?
              nuf = nudam
              # nudam.interpolate( conditional( lt((2*etamp-2*eta0)/dt , alphas*sqrt(1+epsilon*etamp)),0.0*nub0, nub0 ) + (nuf/nub0)*conditional( gt((2*etamp-2*eta0)/dt,0),1.0,0.0 ) )
              # nudam.interpolate( conditional( gt(nudam,nub0), nub0, nudam ) )
              
          phi0.interpolate(2.0*phimp-phi0)
          eta0.interpolate(2.0*etamp-eta0)
                            
          
          if t>tmE+smallfac:
              E = assemble( (0.5*etamp**2 + 0.5*(1+epsilon*etamp)*abs(grad(phimp))**2 )*dx )
              print(t, abs((E-E0)/E0))
              print("%.11f %.11f  %.11f %.11f" % (t, abs((E-E0)/E0), E, E0),file=outputE)
              tmE= tmE+dte



      # Output at times indicated e.g at t = 0,2,4,6,7,8.5,9.5::
      if t>tmeas[nnm]-smallfac and t<tmeas[nnm]+smallfac and nnm < nmea:
          if nVP==2:
              output.write(phi0, eta0, ex_phi, ex_eta, nudamm, time=t) # 
              # WORKS output.write(phi0, eta0, ex_phi, ex_eta, time=t) # FAILS
          else:
              output.write(phi0, eta0, ex_phi, ex_eta, time=t)
          # eta_data[:,nnm]=eta0.dat.data
          # phi_data[:,nnm]=phi0.dat.data
          
          print('tmeas:',tmeas[nnm])
          nnm = nnm+1
          eta12 = np.array([eta0.at(x,yslice) for x in xvals]) # eta12 = np.array([eta0.at(x) for x in xvals])
          phi12 = np.array([phi0.at(x,yslice) for x in xvals])
          # print("%.11f %.11f  %.11f %.11f" % (xvals, eta12, phi12),file=outputetaphi12)
          ax1.plot(xvals,eta12) #, color[int(i-1)],label = f' $\eta_n: t = {t:.3f}$')
          ax2.plot(xvals,phi12) #, color[int(i-1)], label = f' $\phi_n: t = {t:.3f}$')
          if nVP==2:
              nudam12 = np.array([nudamm.at(x,yslice) for x in xvals]) # 
              ax3.plot(xvals,nudam12)        

toc = time.time() - tic
print('Elapsed time (min):', toc/60)
          
outputE.close()

with open(save_path+'eta_data.npy', 'wb') as data1:
    np.save(data1, eta_data)
with open(save_path+'phi_data.npy', 'wb') as data2:
    np.save(data2, phi_data)

plt.show() 
print('*************** PROGRAM ENDS ******************')
    
# Below is a plot of Figure 3 in Bokhove and Kalogirou 2016:
#
# .. figure:: fig3BK2016.png
#    :align: center
# The output can be visualised using `paraview <http://www.paraview.org/>`__.
#
# The data in energy.txt can be viewed in seperate python program to monitor the energy fluctuations.
#
# A python script version of this demo can be found `here <benney_luke.py>`__.
#
# The Benney-Luke system and weak formulations presented in this demo have also been used to model extreme waves that occur due to Mach reflection through the intersection of two obliquely incident solitary waves. More information can be found in :cite:`Gidel:2017`.
#
# .. rubric:: References
#
# .. bibliography:: demo_references.bib
#    :filter: docname in docnames http://www1.maths.leeds.ac.uk/~obokhove/lms-cup2015.pdf
# See also: https://www.cambridge.org/gb/academic/subjects/mathematics/fluid-dynamics-and-solid-mechanics/lectures-theory-water-waves
