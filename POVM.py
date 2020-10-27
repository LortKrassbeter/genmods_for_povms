import numpy as np
import matplotlib.pyplot as plt
import time as time
import scipy.sparse as sparse
import scipy.sparse.linalg as sLA
import itertools
import string

# build sparse Pauli matrices
Sp = sparse.csr_matrix((np.array([[0, 1], [0, 0]])))
Sm = np.transpose(Sp)
Sx = Sp + Sm
Sy = -1j*(Sp - Sm)
Sz = sparse.csr_matrix(np.array([[1, 0], [0, -1]]))
S0 = sparse.eye(2)

# generate a column string of given list entries 
def list_to_strstack(ls):
    st = ""
    for t in ls:
        st += (str(t) + "\n")
    return st

# turn an integer in range [0, 4^n - 1] into an n-dimensional coordinate tuple
def to_a_tuple(a_joint, n):
    vols = 4**np.arange(n)
    coords = np.zeros(n, dtype=int)
    temp_a = a_joint
    for i in range(n - 1, -1, -1):
        coords[i] = temp_a // vols[i]
        temp_a -= coords[i]*vols[i]
    return coords

# turn a n-dim coordinate tuple into an integer in range [0, 4^n - 1]
def to_a_joint(a_tuple, n):
    vols = 4**np.arange(n)
    return np.array(a_tuple) @ vols

# calculate the tetra POVM elements and the overlap matrix for a given number of spins
def tetra_povm(n, debug=False):
    # define elementary POVM elements
    m0 = np.array([[0.5 + 0*1j, 0 + 0*1j], [0 + 0*1j, 0 + 0*1j]])
    m1 = 1/6*np.array([[1 + 0*1j, np.sqrt(2) + 0*1j], [np.sqrt(2) + 0*1j, 2 + 0*1j]])
    m2 = 1/12*np.array([[2 + 0*1j, -np.sqrt(2) - np.sqrt(6)*1j], [-np.sqrt(2) + np.sqrt(6)*1j, 4 + 0*1j]])
    m3 = 1/12*np.array([[2 + 0*1j, -np.sqrt(2) + np.sqrt(6)*1j], [-np.sqrt(2) - np.sqrt(6)*1j, 4 + 0*1j]])
    m_single = [m0, m1, m2, m3]
    
    M = np.zeros((4**n, 2**n, 2**n), dtype=complex) # POVM elements
    T = np.zeros((4**n, 4**n)) # overlap matrix
    outcomes = [] # tuple representation of elements
    for a_joint in range(4**n):
        a_i = to_a_tuple(a_joint, n)
        outcomes.append(a_i)
        if debug:
            print("outcome ", a_i)
        Ma = m_single[a_i[0]]
        for j in a_i[1:]: # multiply together all elements of the tuple
            Ma = np.kron(m_single[j], Ma)
        M[a_joint] = Ma.copy()
        T[a_joint, a_joint] = np.trace(Ma @ Ma) # calculate diag entry of T
        for k in range(a_joint): # calculate off-diag entries of T - TODO optimize
            T[a_joint, k] = np.trace(Ma @ M[k])
            T[k, a_joint] = np.trace(M[k] @ Ma)
    return M, T, outcomes

# returns the string set of all possible basis states for N qubits
def build_basis_strings(N):
    all_states = []
    for i in range(2**N):
        bstring = format(i, "b")
        bstring = "0"*(N - len(bstring)) + bstring
        all_states.append(bstring)
    return np.array(all_states)

# builds and returns the matrix representation of the given operator string, e.g. "1xy" = 1 x simga_x x sigma_y
def build_op(op_string):
    ind = 0
    operator = S0
    for c in op_string[::-1]:
        if ind == 0:
            if c == "x":
                operator = Sx
            elif c == "y":
                operator = Sy
            elif c == "z":
                operator = Sz
            else:
                operator = S0
        else:
            if c == "x":
                operator = sparse.kron(operator, Sx)
            elif c == "y":
                operator = sparse.kron(operator, Sy)
            elif c == "z":
                operator = sparse.kron(operator, Sz)
            else:
                operator = sparse.kron(operator, S0)
        ind += 1
    return operator

# build all spin operator strings up to a specified order
def build_op_strings(N=2, up_to_order=2):
    up_to_order = min(up_to_order, N)
    chars = ["x", "y", "z", "1"]
    for item in itertools.product(chars, repeat=N):
        perm = "".join(item)
        num_ones = perm.count("1")
        if num_ones < N - up_to_order or num_ones == N:
            continue
        yield perm

# calculate the POVM distribution of a given state
def calc_probs(state, povm):
    if povm is None:
        raise ValueError("Specify a POVM to be used.")
    else:
        if callable(povm):
            n = int(np.log2(state.shape[0])) # number of spins
            M, T, _ = povm(n)
            Tinv = np.linalg.inv(T)
        elif len(povm) == 2:
            M, Tinv = povm
        
    p = np.trace(M @ state, axis1=1, axis2=2).astype(np.float)
    return p, M, Tinv

# calculate the operator coefficients for a specified POVM
def calc_op_coeffs(op, povm):
    if sparse.isspmatrix(op):
        op = op.toarray()
    if povm is None:
        raise ValueError("Specify a POVM to be used.")
    else:
        if callable(povm):
            n = int(np.log2(state.shape[0])) # number of spins
            M, T, _ = povm(n)
            Tinv = np.linalg.inv(T)
        elif len(povm) == 2:
            M, Tinv = povm
    tr_op_M = np.trace(np.dot(M, op), axis1=1, axis2=2).astype(np.complex)
    Qop = Tinv @ tr_op_M
    return Qop

# calculate operator expectations
def calc_op_expectations(probs, op_coeffs):
    return np.dot(op_coeffs, probs)

