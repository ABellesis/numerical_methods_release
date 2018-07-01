import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import numpy as np
from  scipy.optimize import line_search as linesearch
import scipy as sp
import scipy.linalg as spla
import scipy.special as sps
import sympy as syp

def plot_func(xminimum,xmaximum,yminimum,ymaximum,steps_x,steps_y,f):
    fig = plt.figure(figsize=(10.5,4.95))
    X=np.linspace(xminimum,xmaximum,100)
    Y=np.linspace(yminimum,ymaximum,100)
    X, Y = np.meshgrid(X, Y)
    points=np.array((X,Y))
    Z = np.zeros_like(X)
    for i in range(len(Z)):
        for j in range(len(Z)):
            Z[i,j] = f((X[i,j],Y[i,j]))
    Zrange= np.max(Z) - np.min(Z)
    zminimum = np.min(Z) - 0.1*Zrange
    zmaximum = np.max(Z) + 0.1*Zrange
    ax = plt.subplot(121, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,alpha=0.45, linewidth=0, antialiased=True, cmap = cm.jet)
    cset = ax.contour(X, Y, Z, zdir='Z', offset = -1, cmap=cm.jet)
    ax.set_xlabel("X", linespacing = 3.2)
    ax.set_zlabel("Z", linespacing=3.2)
    ax.set_ylabel("Y", linespacing=3.2)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.rcParams['ytick.major.pad']='8'
    ax.set_xlim(xminimum, xmaximum)
    ax.set_ylim(yminimum, ymaximum)
    ax.set_zlim(zminimum,zmaximum)

    ax2=plt.subplot(122)
    plt.contour(X,Y,Z, 15, colors ='k')
    plt.contourf(X,Y,Z, 15, cmap=cm.jet,vmax=abs(Z).max(), vmin=-abs(Z).max())
    plt.plot(steps_x,steps_y, marker = 'o',markerfacecolor='white',markeredgecolor='black',color='white')
    steps_z = np.zeros_like(steps_x)
    for i in range(len(steps_x)):
        steps_z[i] = f((steps_x[i],steps_y[i]))
    ax.scatter(steps_x,steps_y,steps_z,color='black',marker='o')
    ax2.set_xlim(steps_x[len(steps_x)-1]-0.5, steps_x[len(steps_x)-1]+0.5)
    ax2.set_ylim(steps_y[len(steps_y)-1]-0.5, steps_y[len(steps_y)-1]+0.5)
    fig.set_tight_layout(True)
    plt.show()

def plotting(step,steps_radius, steps_deformation, steps_energy):
    radius = steps_radius[step]
    deformation =steps_deformation[step]
    fig,ax = plt.subplots(nrows=1,ncols=1,facecolor='white',figsize=(5,3))
    for cur_step in range(step+1):
        molecule = make_h_ring(6, steps_radius[cur_step], steps_deformation[cur_step], 3)
        alpha_val = 0.5*((cur_step+1)/(step+1))+0.5
        for atom in molecule:
            x = atom["pos"][0]
            y = atom["pos"][1]
            ax.plot(x,y,alpha=alpha_val,marker='.',markersize=20,linestyle=None,color = 'black')
        theta = np.linspace(0,2*np.pi,100)
        x = steps_radius[cur_step] * np.cos(theta)
        y = steps_radius[cur_step] * np.sin(theta)
        ax.plot(x,y,linestyle="--",alpha=alpha_val,color="black")
        
    ax.set_xlim(-max(steps_radius),max(steps_radius))
    ax.set_ylim(-max(steps_radius),max(steps_radius))
    plt.subplots_adjust(right = 3/5)
    ax.annotate('Step = {:d}'.format(step), xy=(1.1*3/5, 0.8),
               xycoords='figure fraction',ha="left", va="top")
    ax.annotate('Energy = {: 4.5f}'.format(steps_energy[step]), xy=(1.1*3/5, 0.7),
                   xycoords='figure fraction',ha="left", va="top")
    ax.annotate('Radius = {: 4.5f}'.format(radius), xy=(1.1*3/5, 0.6),
               xycoords='figure fraction',ha="left", va="top")
    ax.annotate('Deformation = {: 4.3f}'.format(deformation), xy=(1.1*3/5, 0.5),
               xycoords='figure fraction',ha="left", va="top")
    plt.show()
    return 

def optimize_H6_ring(radius, deformation):
    num_contracted = 3
    num_atoms = 6
    
    initial_guess = np.array((radius, deformation))  # initial guess
    #convergence criteria
    residual_tolerance = 1e-3  
    update_tolerance = 1e-5
    IT_MAX = 20
    old_guess = initial_guess
    approx_hessian = np.identity(2)
    inverse_approx_hessian = np.identity(2)
    num_steps = 0

    s_i = np.array([1.0, 0.0])
    y_i = np.array([0.0, 1.0])

    converged = False
    steps_radius = []
    steps_deformation = []
    steps_energy = []
    steps_energy.append(HF_energy(old_guess))
    steps_radius.append(old_guess[0])
    steps_deformation.append(old_guess[1])
    print('Starting optimization. Please be patient.')
    while not converged and num_steps <= IT_MAX:
        num_steps += 1
        new_search_direction = np.dot(inverse_approx_hessian, -HF_gradient(old_guess))
        try:
            s_i = linesearch(HF_energy, HF_gradient, old_guess, new_search_direction)[
                0] * new_search_direction
        except:
            print("WARNING: HESSIAN IS NOT POSITIVE DEFINITE... RESETTING SEARCH DIRECTION TO STEEPEST DESCENT!")
            new_search_direction = -HF_gradient(old_guess)
            s_i = linesearch(HF_energy, HF_gradient, old_guess, new_search_direction)[0] * new_search_direction

        new_guess = old_guess + s_i
        if new_guess[1] > 1.0 or new_guess[1] < 0.5:
            new_guess[1] = 1.0 - (np.abs(new_guess[1]) % 0.5)
        y_i = HF_gradient(new_guess) - HF_gradient(old_guess)
        approx_hessian += ((np.outer(y_i, y_i)) / (np.dot(y_i, s_i))) - ((np.dot(approx_hessian,
                                                                                 np.dot(np.outer(s_i, s_i), approx_hessian))) / (np.dot(s_i, np.dot(approx_hessian, s_i))))
        # update using Sherman-Morrison formula
        part_A = np.dot(s_i, y_i)
        part_A += np.dot(y_i, np.dot(inverse_approx_hessian, y_i))
        part_A /= np.dot(s_i, y_i)**2
        part_A *= np.outer(s_i, s_i)
        part_B = np.dot(inverse_approx_hessian, np.outer(y_i, s_i))
        part_B += np.dot(np.outer(s_i, y_i), inverse_approx_hessian)
        part_B /= np.dot(s_i, y_i)
        inverse_approx_hessian += part_A - part_B

        diff = old_guess - new_guess
        diff = np.sqrt(np.dot(diff, diff))
        print ("Step: {}".format(num_steps))
        print ("OLD GUESS: {}".format(old_guess))
        print ("NEW GUESS: {}".format(new_guess))
        print ("DIFFERENCE: {}\n".format(diff))
        old_guess = new_guess
        old_search_direction = new_search_direction
        steps_radius.append(old_guess[0])
        steps_deformation.append(old_guess[1])
        steps_energy.append(HF_energy(old_guess))
        if(diff < residual_tolerance):
            converged = True
    steps_radius = np.array(steps_radius)
    steps_deformation = np.array(steps_deformation)
    steps_energy = np.array(steps_energy)
    return steps_radius, steps_deformation, steps_energy


def make_h_ring(num_atoms, radius, deformation, num_contracted):
    if deformation > 1.0 or deformation < 0.5:
        deformation = 1.0 - (np.abs(deformation) % 0.5)
    molecule = []
    for atom in range(num_atoms):
        curr_atom = dict()
        if atom % 2 == 0:
            deformation_amount = deformation
        else:
            deformation_amount = 1.0 + (1.0 - deformation)
        theta = (atom)*((2.0*np.pi)/(num_atoms))
        theta += ((2.0*np.pi)/(num_atoms))*(deformation_amount)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 0.0
        curr_atom["index"] = atom
        curr_atom["element"] = "H"
        curr_atom["pos"] = np.array([x,y,z])
        curr_atom["basis"] = populate_basis_functions(num_contracted)
        # print(curr_atom)
        molecule.append(curr_atom)
    return molecule
def plot_h_ring(num_atoms, radius, deformation, num_contracted):
    molecule = make_h_ring(num_atoms, radius, deformation, num_contracted)
    fig = plt.figure(facecolor='white',figsize=(3,3))
    ax = fig.add_subplot(111)
    fig.set_tight_layout(True)
    for atom in molecule:
        x = atom["pos"][0]
        y = atom["pos"][1]
        plt.plot(x,y,marker='.',markersize=20,linestyle=None,color = 'black')
    theta = np.linspace(0,2*np.pi,100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    plt.plot(x,y,linestyle="--",color="black")
    plt.show()
def overlap_helper(pos_a, pos_b, exponent_a, exponent_b):
    r = pos_b - pos_a
    r = np.dot(r,r)
    integral = (np.pi/(exponent_a + exponent_b))**(1.5)
    integral *= np.exp(-((exponent_a * exponent_b)*r)/(exponent_a + exponent_b))
    return integral
def kinetic_helper(pos_a, pos_b, exponent_a, exponent_b):
    r = pos_b - pos_a
    r = np.dot(r,r)
    part_1 = (exponent_a * exponent_b)/ (exponent_a + exponent_b)
    part_2 = part_1*(6.0 - (4.0*part_1*r))
    part_3 = (np.pi/(exponent_a + exponent_b))**(1.5)
    part_3 *= np.exp(-part_1 * r)
    integral = part_2 * part_3
    return integral
def Boys(x):
    if x == 0:
        return 1.0
    t = np.sqrt(x)
    return (1.0/t) * (np.sqrt(np.pi)/2) * sps.erf(t)
def nuclear_helper(pos_a, pos_b, exponent_a, exponent_b, center, Z):
    r = pos_b - pos_a
    r = np.dot(r,r)
    part_1 = -2.0*np.pi*Z/(exponent_a+exponent_b)

    part_2 = (exponent_a * exponent_b)/(exponent_a + exponent_b)
    part_2 = np.exp(-part_2 * r)
    r_p = exponent_a*pos_a + exponent_b*pos_b
    r_p /= (exponent_a + exponent_b)
    r_pc = r_p - center
    r_pc = np.dot(r_pc,r_pc)
    part_3 = Boys((exponent_a + exponent_b)*r_pc)
    integral = part_1 * part_2 * part_3
    if integral == 0:
        print(integral)
    return integral
def populate_basis_functions(num_contracted):
    # STO-2G
    """
    !  STO-2G  EMSL  Basis Set Exchange Library   4/11/18 11:33 AM
    ! Elements                             References
    ! --------                             ----------
    !  H - He: W.J. Hehre, R.F. Stewart and J.A. Pople, J. Chem. Phys. 2657 (1969).
    ! Li - Ne:
    ! Na - Ar: W.J. Hehre, R. Ditchfield, R.F. Stewart, J.A. Pople, J. Chem. Phys.
    !  K - Kr: 52, 2769 (1970).
    !


    ****
    H     0
    S   2   1.00
          1.309756377            0.430128498
          0.233135974            0.678913531
    ****
    """
    # STO-3G
    """
    !  STO-3G  EMSL  Basis Set Exchange Library   4/11/18 11:34 AM
    ! Elements                             References
    ! --------                             ----------
    !  H - Ne: W.J. Hehre, R.F. Stewart and J.A. Pople, J. Chem. Phys. 2657 (1969).
    ! Na - Ar: W.J. Hehre, R. Ditchfield, R.F. Stewart, J.A. Pople,
    !          J. Chem. Phys.  2769 (1970).
    ! K,Ca - : W.J. Pietro, B.A. Levy, W.J. Hehre and R.F. Stewart,
    ! Ga - Kr: J. Am. Chem. Soc. 19, 2225 (1980).
    ! Sc - Zn: W.J. Pietro and W.J. Hehre, J. Comp. Chem. 4, 241 (1983) + Gaussian.
    !  Y - Cd: W.J. Pietro and W.J. Hehre, J. Comp. Chem. 4, 241 (1983). + Gaussian
    !


    ****
    H     0
    S   3   1.00
          3.42525091             0.15432897
          0.62391373             0.53532814
          0.16885540             0.44463454
    ****
    """
    basis = dict()
    basis["num_shells"] = 1
    if num_contracted == 2:
        basis["exponents"] = np.array([1.309756377,0.233135974])
        basis["contraction_coefficients"] = np.array([0.430128498,0.678913531])
    elif num_contracted == 3:
        basis["exponents"] = np.array([3.42525091, 0.62391373, 0.16885540])
        basis["contraction_coefficients"] = np.array([0.15432897,0.53532814,0.44463454])
    norm = []
    pos = np.array([0.0,0.0,0.0])
    norm_const = 0.0
    for exponents_a, contraction_a in zip(basis["exponents"],basis["contraction_coefficients"]):
        norm_const = (2.0*exponents_a/np.pi)**(3.0/4.0)
        norm.append(norm_const)
    basis["norm_const"] = np.array(norm)
    return basis
def overlap(molecule):
    """
    Only works for s functions.
    """
    count = 0
    for atom in molecule:
        for shell in range(atom['basis']['num_shells']):
            count += 1
    overlap_mat = np.zeros((count,count))
    i = 0
    j = 0
    for atom_a in molecule:
        for shell_a in range(atom_a['basis']['num_shells']):
            for atom_b in molecule:
                for shell_b in range(atom_b['basis']['num_shells']):
                    overlap_mat[i,j] = 0.0
                    for exponents_a, contraction_a, norm_a in zip(atom_a['basis']['exponents'], atom_a['basis']['contraction_coefficients'], atom_a['basis']['norm_const']):
                        for exponents_b, contraction_b, norm_b in zip(atom_b['basis']['exponents'], atom_b['basis']['contraction_coefficients'], atom_b['basis']['norm_const']):
                            overlap_mat[i,j] += contraction_a * contraction_b * norm_a * norm_b * overlap_helper(atom_a['pos'],atom_b['pos'],exponents_a, exponents_b)
                    j+=1
            i+=1
            j=0
    return overlap_mat
def kinetic(molecule):
    """
    Only works for s functions.
    """
    count = 0
    for atom in molecule:
        for shell in range(atom['basis']['num_shells']):
            count += 1
    kinetic_mat = np.zeros((count,count))
    i = 0
    j = 0
    for atom_a in molecule:
        for shell_a in range(atom_a['basis']['num_shells']):
            for atom_b in molecule:
                for shell_b in range(atom_b['basis']['num_shells']):
                    kinetic_mat[i,j] = 0.0
                    for exponents_a, contraction_a, norm_a in zip(atom_a['basis']['exponents'], atom_a['basis']['contraction_coefficients'], atom_a['basis']['norm_const']):
                        for exponents_b, contraction_b, norm_b in zip(atom_b['basis']['exponents'], atom_b['basis']['contraction_coefficients'], atom_b['basis']['norm_const']):
                            kinetic_mat[i,j] += contraction_a * contraction_b * norm_a * norm_b * kinetic_helper(atom_a['pos'],atom_b['pos'],exponents_a, exponents_b)

                    j+=1
            i+=1
            j=0
    kinetic_mat *= 0.5
    return kinetic_mat
def nuclear(molecule):
    """
    Only works for s functions.
    """
    count = 0
    for atom in molecule:
        for shell in range(atom['basis']['num_shells']):
            count += 1
    nuclear_mat = np.zeros((count,count))
    i = 0
    j = 0
    for center in molecule:
        i=0
        j=0
        for atom_a in molecule:
            for shell_a in range(atom_a['basis']['num_shells']):
                for atom_b in molecule:
                    for shell_b in range(atom_b['basis']['num_shells']):
                        for exponents_a, contraction_a, norm_a in zip(atom_a['basis']['exponents'], atom_a['basis']['contraction_coefficients'], atom_a['basis']['norm_const']):
                            for exponents_b, contraction_b, norm_b in zip(atom_b['basis']['exponents'], atom_b['basis']['contraction_coefficients'], atom_b['basis']['norm_const']):
                                nuclear_mat[i,j]+=contraction_a*contraction_b*norm_a*norm_b*nuclear_helper(atom_a['pos'],atom_b['pos'],exponents_a,exponents_b,center['pos'],1.0)
                    j+=1
            i+=1
            j=0
    return nuclear_mat
def two_elec_helper(exp_a,pos_a,exp_b,pos_b,exp_c,pos_c,exp_d,pos_d):
    part_1 = 2.0*(np.pi**(2.5))
    part_1 /= (exp_a+exp_c)
    part_1 /= (exp_b+exp_d)
    part_1 /= np.sqrt(exp_a+exp_c+exp_b+exp_d)

    r_ac = pos_c-pos_a
    r_ac = np.dot(r_ac,r_ac)
    r_ac *= -(exp_a*exp_c)/(exp_a+exp_c)
    r_bd = pos_d-pos_b
    r_bd = np.dot(r_bd,r_bd)
    r_bd *= -(exp_b*exp_d)/(exp_b+exp_d)
    part_2 = np.exp(r_ac + r_bd)

    r_p = ((exp_a*pos_a)+(exp_c*pos_c))/(exp_a+exp_c)
    r_q = ((exp_b*pos_b)+(exp_d*pos_d))/(exp_b+exp_d)
    r_pq = r_p - r_q
    r_pq = np.dot(r_pq,r_pq)
    part_3 = Boys(((exp_a+exp_c)*(exp_b+exp_d)/(exp_a+exp_c+exp_b+exp_d))*r_pq)

    return part_1 * part_2 * part_3
def two_electron(molecule):
    count = 0
    for atom in molecule:
        for shell in range(atom['basis']['num_shells']):
            count += 1
    two_elec_mat = np.zeros((count,count,count,count))
    i = 0
    j = 0
    k = 0
    l = 0
    for atom_a in molecule:
        for shell_a in range(atom_a['basis']['num_shells']):
            for atom_c in molecule:
                for shell_c in range(atom_c['basis']['num_shells']):
                    for atom_b in molecule:
                        for shell_b in range(atom_b['basis']['num_shells']):
                            for atom_d in molecule:
                                for shell_d in range(atom_d['basis']['num_shells']):
                                    for exponents_a, contraction_a, norm_a in zip(atom_a['basis']['exponents'], atom_a['basis']['contraction_coefficients'], atom_a['basis']['norm_const']):
                                        for exponents_b, contraction_b, norm_b in zip(atom_b['basis']['exponents'], atom_b['basis']['contraction_coefficients'], atom_b['basis']['norm_const']):
                                            for exponents_c, contraction_c, norm_c in zip(atom_c['basis']['exponents'], atom_c['basis']['contraction_coefficients'], atom_c['basis']['norm_const']):
                                                for exponents_d, contraction_d, norm_d in zip(atom_d['basis']['exponents'], atom_d['basis']['contraction_coefficients'], atom_d['basis']['norm_const']):
                                                    two_elec_mat[i,j,k,l]+= \
                                                        contraction_a*norm_a*\
                                                        contraction_b*norm_b*\
                                                        contraction_c*norm_c*\
                                                        contraction_d*norm_d*\
                                                        two_elec_helper( \
                                                            exponents_a,atom_a['pos'], \
                                                            exponents_b,atom_b['pos'], \
                                                            exponents_c,atom_c['pos'], \
                                                            exponents_d,atom_d['pos']  \
                                                        )
                                l += 1
                        k += 1
                        l = 0
                    #CODE HERE
                j += 1
                k = 0
                l = 0
        i += 1
        j = 0
        k = 0
        l = 0
    return two_elec_mat
def nuclear_repulsion_energy(molecule):
    num_atoms = len(molecule)
    repulsion_energy = 0.0
    for i in range(num_atoms):
        for j in range(i+1,num_atoms):
            r = molecule[i]["pos"] - molecule[j]["pos"]
            r = np.sqrt(np.dot(r,r))
            repulsion_energy += 1.0/r
    return repulsion_energy
def form_density_matrix(C, num_ao, num_elec_alpha):
    D = np.zeros((num_ao,num_ao))
    for i in range(num_ao):
        for j in range(num_ao):
            for k in range(num_elec_alpha):
                D[i,j] +=  C[i,k] * C[j,k]
    return D
def rmsc_dm(D, D_last):
    return np.sqrt(np.sum((D - D_last)**2))
def diagonalize_fock(F, S):
    return spla.eigh(F,S)
def RHF_electronic_energy(D, H, F, num_ao, num_elec_alpha):
    E = np.sum(np.multiply(D , (H +  F)))
    return E
def RHF_coulomb_exchange(D, eri, num_ao):
    G = np.zeros((num_ao,num_ao))
    for i in range(num_ao):
        for j in range(num_ao):
            for k in range(num_ao):
                for l in range(num_ao):
                    G[i,j] += D[k,l] * ((2.0*(eri[i,j,k,l])) - (eri[i,k,j,l]))
    return G
def RHF(mol, print_hf = True, return_coeff=False):
    """
    Needs doc
    """
    # get size of basis and number of electrons
    num_ao = 6
    num_elec_alpha, num_elec_beta = (3,3)
    # calculate nuclear repulsions energy (scalar)
    E_nuc = nuclear_repulsion_energy(mol)
    # calculate overlap integrals
    S = overlap(mol)
    # calculate kinetic energy integrals
    T = kinetic(mol)
    # calculate nuclear attraction integrals
    V = nuclear(mol)
    # form core Hamiltonian
    H = T + V
    # parameters for main loop
    iteration_max = 100
    convergence_E = 1e-9
    convergence_DM = 1e-5
    # loop variables
    iteration_num = 0
    E_total = 0
    E_elec = 0.0
    iteration_E_diff = 0.0
    iteration_rmsc_dm = 0.0
    converged = False
    # get two electron integrals
    eri = two_electron(mol)
    # set inital density matrix to zero
    D = np.zeros((num_ao,num_ao))
    # main iteration loop
    while(not converged):
        iteration_num += 1
        # store last iteration
        E_elec_last = E_elec
        D_last = np.copy(D)
        # form G matrix
        G = RHF_coulomb_exchange(D, eri, num_ao)
        # build fock matrix
        F  = H + G
        # calculate electronic energy
        E_elec = RHF_electronic_energy(D, H, F, num_ao, num_elec_alpha)
        E_total = E_elec + E_nuc
        # calculate energy change of iteration
        iteration_E_diff = np.abs(E_elec - E_elec_last)
        # solve the generalized eigenvalue problem
        E_orbitals, C = diagonalize_fock(F,S)
        # compute new density matrix
        D = form_density_matrix(C, num_ao, num_elec_alpha)
        # rms change of density matrix
        iteration_rmsc_dm = rmsc_dm(D, D_last)
        # test convergence
        if(np.abs(iteration_E_diff) < convergence_E and iteration_rmsc_dm < convergence_DM):
            converged = True
        if(iteration_num == iteration_max):
            converged = True
            E_total = None
    if return_coeff:
        return E_total, C
    else:
        return E_total

def HF_energy(rad_def):
    radius, deformation = rad_def
    num_atoms = 6
    num_contracted = 3
    H_ring = make_h_ring(num_atoms, radius, deformation, num_contracted)
    return RHF(H_ring)

def HF_gradient(rad_def):
    radius, deformation = rad_def
    num_atoms = 6
    num_contracted = 3
    h = 1e-3
    gradient = []
    gradient.append((HF_energy((radius + h, deformation)) - HF_energy((radius - h, deformation))) / (2 * h))
    gradient.append((HF_energy((radius, deformation + h)) - HF_energy((radius , deformation- h))) / (2 * h))
    return np.array(gradient)

def question_one_check(ans=0):
        if ans == 2:
            print("Correct! The structure with equally spaced hydrogens was higher in energy than the paired hydrogen structure. This indicates that although BFGS located this minimum, it was in fact not the global minimum.")
        elif ans ==0:
            print("Please input a valid response")
        else:
            print("Incorrect. Please try again.")
    
if __name__ == "__main__":
    print("This file is contains additional plotting functions. It is not meant to be run independently")

def evaluate_hessian(f):
    x,y= syp.symbols('x y')
    fhess=np.array([[None for i in range(2)],[None for i in range(2)]])
    for idx1,v1 in enumerate((x,y)):
        for idx2,v2 in enumerate((x,y)):
                fhess[idx1][idx2]=syp.diff(f(x,y), v1, v2)
    fhess_lambda = syp.lambdify((x,y),fhess,'numpy')
    fhess_lambda_compact = compact_input(fhess_lambda)
    return(fhess_lambda_compact)


def evaluate_gradient(f):
    x,y= syp.symbols('x y')
    fgrad=np.array([None for i in range(2)])
    for idx1,v1 in enumerate((x,y)):
                fgrad[idx1]=syp.diff(f(x,y), v1)
    fgrad_lambda = syp.lambdify((x,y),fgrad,'numpy')
    fgrad_lambda_compact = compact_input(fgrad_lambda)
    return fgrad_lambda_compact

def compact_input(f):
    return lambda X: np.array(f(X[0],X[1]),dtype=float)