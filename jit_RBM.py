import numpy as np
import numba
import matplotlib.pyplot as plt

def calc_joint_distribution(w, b, N_visible, N_hidden, spin=False):
    vs = np.zeros((2**N_visible, N_visible))
    p_joint = np.zeros((2**N_visible, 2**N_hidden))
    for ai in np.arange(2**N_visible):
        vs[ai, :] = dec_to_bin(ai, N_visible)
        if spin: vs[ai, :] = bin_to_spin(vs[ai, :])
        for bi in np.arange(2**N_hidden):
            hs[bi, :] = dec_to_bin(bi, N_hidden)
            if spin: hs[bi, :] = bin_to_spin(hs[bi, :])
            p_joint[ai, bi] = np.exp(-energy(vs[ai, :], hs[bi, :], w, a, b))  
    return p_joint/p_joint.sum()

@numba.jit
def calc_pvis_ana(w, b, vis_states, spin=False):
    N_vis = vis_states.shape[1]
    pvis = np.zeros(vis_states.shape[0])
    wres = w[:N_vis, N_vis:]
    vb = b[:N_vis]
    hb = b[N_vis:]
    N_hid = hb.shape[0]
    C = np.exp(np.dot(vis_states, vb))
    B = 1 + np.exp(np.dot(vis_states, wres) + hb) if not spin else 2*np.cosh(np.dot(vis_states, wres) + hb)
    pvis = C
    for i in range(N_hid):
        pvis *= B[:,i]
    return pvis/pvis.sum()

@numba.jit
def calc_pvis_ana_cosh(w, b, vis_states):
    N_vis = vis_states.shape[1]
    pvis = np.zeros(vis_states.shape[0])
    wres = w[:N_vis, N_vis:]
    vb = b[:N_vis]
    hb = b[N_vis:]
    N_hid = hb.shape[0]
    C = np.exp(np.dot(vis_states, vb))
    B = 2*np.cosh(np.dot(vis_states, wres) + hb)
    pvis = C
    for i in range(N_hid):
        pvis *= B[:,i]
    return pvis/pvis.sum()

def plot_distr(distribution1, distribution2, N, distribution3=None, label1="RBM", label2="SNN", label3="target"):
    labels = np.arange(2**N)

    x = np.arange(0, 2*(2**N), 2) # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(figsize=(20, 10))
    rects1 = ax.bar(np.add(x, width/2), distribution1, width, label=label1)

    ax.set_ylabel('probability')
    ax.set_xlabel('state')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc = 0)
    rects2 = ax.bar(np.subtract(x, width/2), distribution2, width, label=label2)
    if not distribution3 is None:
        rects3 = ax.bar(np.add(x, width*1.5), distribution3, width, label=label3)
    ax.legend(loc = 0)
    plt.show()

@numba.jit(nopython=True)
def bin_to_spin(bin_string):
    return 2*bin_string - 1

@numba.jit(nopython=True)
def spin_to_bin(spin_string):
    return 0.5*(spin_string + 1)

def gen_states(N, spin=False):
    states = np.zeros((2**N, N), dtype=float)
    for i in range(2**N):
        states[i] = dec_to_bin(i, N)
        if spin:
            states[i] = bin_to_spin(states[i])
    return states

@numba.jit(nopython=True)
def calc_pvis(samples, N_vis, N_hid, spin=False):#, power_array):
    p = np.zeros(2**N_vis)
    for i in range(samples.shape[0]):
        #state_vals_vis = int(np.dot(samples[i, :], power_array[:]) // 2**N_hid)
        state_vals_vis = bin_to_dec(samples[i, :N_vis], N_vis)
        p[int(state_vals_vis)] += 1.

    p /= np.shape(samples)[0]
    return p

def SNN_sample(w, b, sim_dur, t_ref, sampling_interval, N_vis, N_hid, power_array):
    pop_visible, pop_hidden, sd, multimeter1, noise_ex, noise_in = createSNN(N_vis, N_hid, params_neuron, 
                                                                    params_poisson, params_spike_detector)
    synBiases = setBiases(b, pop_visible, pop_hidden, alpha_1kHz, u_bar_0_1kHz, noise_ex, noise_in, setDirect=False)
    synWeights = setWeights(w, pop_visible, pop_hidden, alpha_1kHz, noise_ex, noise_in)
    nest.Simulate(sim_dur)
    samples = sample_joint(sd, t_ref, sim_dur, int(sampling_interval), N_vis, N_hid)[0]
    nest.ResetKernel()
    #p_sampled, bins = np.histogram(v_idx_samples, bins=range(p_true.shape[0] + 1), density=True)
    p_sampled = calc_pvis(samples, N_vis, N_hid)#, power_array)
    #p_sampled_joint = makeDistribution(N_vis+N_hid, samples)
    #SNN_distribution = makeDistribution(N_vis, samples[:,:N_vis])
    return samples, p_sampled#, p_sampled_joint#SNN_distribution

@numba.jit(nopython=True)
def calc_weight_updates(p_true, p_sampled, samples, N_vis, N_hid):
    # calculate weight updates
    dw = np.zeros((N_vis + N_hid, N_vis + N_hid))
    db = np.zeros(N_vis + N_hid) 
    p_ratio = np.zeros(p_true.shape)
    for i in range(p_true.shape[0]):
        if p_sampled[i] != 0:
            p_ratio[i] = 1. - p_true[i]/p_sampled[i] 

    for i in range(samples.shape[0]):
        #vis_state_id = int(np.dot(samples[i,:], power_array[:]) // (2**n_hid))
        vis_state_id = bin_to_dec(samples[i, :N_vis], N_vis)
        db += - p_ratio[int(vis_state_id)]*samples[i, :]
        dw += - p_ratio[int(vis_state_id)]*np.outer(samples[i, :], samples[i, :])
    return dw/samples.shape[0], db/samples.shape[0]

def dec_to_bin(x, n):                                                                                                                                                                
    binary = np.binary_repr(x, width = n)
    ret = np.array([int(y) for y in list(binary)])
    return ret 

@numba.jit(nopython=True)
def bin_to_dec(x, n):
    decimal = 0
    for i in range(n):
        decimal += x[i] * 2 ** (n - 1 - i)
    return decimal

@numba.jit(nopython=True)
def sigmoid(z):
    return 1./(1. + np.exp(-z))

@numba.jit(nopython=True)
def p_vh_bin1(h, w, a):
    z = np.dot(w, h) + a
    return sigmoid(z)

@numba.jit(nopython=True)
def p_hv_bin1(v, w, b):
    z = np.dot(v, w) + b
    return sigmoid(z)

def p_vh_bin(h, w, a):
    z = a[None,:] + np.einsum("ij,kj->ki", w, h)
    return sigmoid(z)

def p_hv_bin(v, w, b):
    z = b[None,:] + np.einsum("ki,ij->kj", v, w)
    return sigmoid(z)

@numba.jit
def bin_sampling(p, spin=False):
    samples = []
    for i in range(p.shape[0]):
        sample = []
        for j in range(p.shape[1]):
            sample.append(np.random.binomial(1, p[i,j]))
        samples.append(sample)
    return np.array(samples)

@numba.jit(nopython=True)
def bin_sampling1(p, spin=False):
    sample = np.zeros(p.shape[0])
    for j in range(p.shape[0]):
        sample[j] = np.random.binomial(1, p[j])
    return sample if not spin else bin_to_spin(sample)

@numba.jit(nopython=True)
def gibbs_h(v, w, b, spin=False):
    phv = p_hv_bin1(v, w, b)
    h = bin_sampling1(phv, spin=spin)
    return h#, phv

@numba.jit(nopython=True)
def gibbs_v(h, w, b, spin=False):
    pvh = p_vh_bin1(h, w, b)
    v = bin_sampling1(pvh, spin=spin)
    return v#, pvh

@numba.jit(nopython=True)
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

@numba.jit(nopython=True)
def gibbs_sampling(w, b, vis_states, num_chains=4, num_samples=100, pvis=None, burn_in=0, spin=False):
    N_vis = vis_states.shape[1]
    #pvis = np.zeros(vis_states.shape[0])
    wres = w[:N_vis, N_vis:]
    vb = b[:N_vis]
    hb = b[N_vis:]
    N_hid = hb.shape[0]
    vis_samples = np.zeros((int(num_chains*num_samples), N_vis))
    hid_samples = np.zeros((int(num_chains*num_samples), N_hid))
    p_sample = np.zeros((2**N_vis, 2**N_hid))
    
    N_hid = hb.shape[0]
    for c in range(num_chains):
        #v = np.zeros(N_vis)
        #h = np.zeros(N_hid)
        if pvis is None:
            vind = np.random.randint(vis_states.shape[0])
        else:
            vind = int(rand_choice_nb(np.arange(vis_states.shape[0]), pvis))
        v = vis_states[vind, :]
        for s in range(num_samples + burn_in):
            s = s - burn_in
            h = gibbs_h(v, wres, hb, spin=spin)
            v = gibbs_v(h, wres, vb, spin=spin)
            if s >= 0:
                vis_samples[int(c*s), :] = v
                hid_samples[int(c*s), :] = h
                hind = bin_to_dec(h, N_hid) if not spin else bin_to_dec(spin_to_bin(h), N_hid)
                vind = bin_to_dec(v, N_vis) if not spin else bin_to_dec(spin_to_bin(v), N_vis)
                p_sample[int(vind), int(hind)] += 1
    samples = np.concatenate((vis_samples, hid_samples), axis=1)
    return p_sample/num_chains/num_samples, samples