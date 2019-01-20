import numpy as np
from functools import partial
import pickle

def _obj_wrapper(func, args, x):
    return func(x, *args)

def pso(func, args, lb, ub, vmax,
        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=25,
        minstep=0.0005, maxfunc=0.99, debug=False, processes=1,
        particle_output=False):
    """
    Perform a particle swarm optimization (PSO)

    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)

    Optional
    ========
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega : scalar
        Particle velocity scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    processes : int
        The number of processes to use to evaluate objective function and
        constraints (default: 1)
    particle_output : boolean
        Whether to include the best per-particle position and the objective
        values at those.

    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
    p : array
        The best known position per particle
    pf: arrray
        The objective values at each position in p

    """

    assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub > lb), 'All upper-bound values must be greater than lower-bound values'

    vhigh = np.array(vmax)
    vlow = -1*vhigh

    # Initialize objective function
    obj = partial(_obj_wrapper, func, args)

    # Initialize the multiprocessing module if necessary
    if processes > 1:
        from torch.multiprocessing import Pool, Process, set_start_method
        set_start_method('spawn')

    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fx = np.zeros(S)  # current particle function values
    fp = np.zeros(S)  # best particle function values
    g = []  # best swarm position
    fg = 0  # best swarm position starting value
    #devices = [0 if x < S/2 else 1 for x in range(S)]
    devices = [0 for x in range(S)]
	
    # Initialize the particle's position
    x = lb + x * (ub - lb)

    # Calculate objective and constraints for each particle
    
    #historys = pickle.load(open('/home/mcis105/yuhongfei/AI_Course/seqmnist/0','rb'))
    #x = np.array(historys['x'])
    #v = np.array(historys['v'])
    #fx = np.array(historys['fx'])

    if processes > 1:
        mp_pool = Pool(processes)
        fx = np.array(mp_pool.map(obj, np.c_[x,devices]))
        mp_pool.close()
        mp_pool.join()
    else:
        for i in range(S):
            fx[i] = obj(np.c_[x[i, :],devices[i])


    # Store particle's best position (if constraints are satisfied)
    i_update = fx > fp
    p[i_update, :] = x[i_update, :].copy()
    fp[i_update] = fx[i_update]

    # Update swarm's best position
    i_max = np.argmax(fp)
    if fp[i_max] > fg:
        fg = fp[i_max]
        g = p[i_max, :].copy()
    else:
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        g = x[0, :].copy()

    # Initialize the particle's velocity
    v = vlow + np.random.rand(S, D) * (vhigh - vlow)

    # Iterate until termination criterion met ##################################
    it = 1

    while it <= maxiter:
        historys = {}
        historys['x'] = x
        historys['fx'] = fx
        historys['v'] = v
        historys['p'] = p
        historys['fp'] = fp
        historys['g'] = g
        historys['fg'] = fg
        #pickle.dump(historys,file=open('/home/mcis105/yuhongfei/AI_Course/seqmnist/PSO/' + str(it - 1), 'wb'))

        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))

        # Update the particles velocities
        v = omega * v + phip * rp * (p - x) + phig * rg * (g - x)
        print(v)
        print(vhigh)
        for i in range(S):
            for j in range(len(v[0])):
                if v[i][j]<vlow[j]:
                    v[i][j] = vlow[j]
                elif v[i][j]>vhigh[j]:
                    v[i][j] = vhigh[j]
        # Update the particles' positions
        x = x + v
        # Correct for bound violations
        for i in range(S):
            for j in range(len(x[0])):
                if x[i][j] < lb[j]:
                    x[i][j] = lb[j] + lb[j] - x[i][j]
                elif x[i][j] > ub[j]:
                    x[i][j] = ub[j] + ub[j] - x[i][j]
        #maskl = x < lb
        #masku = x > ub
        #x = x * (~np.logical_or(maskl, masku)) + lb * maskl + ub * masku

        # Update objectives and constraints
        if processes > 1:
            mp_pool = Pool(processes)
            fx = np.array(mp_pool.map(obj, np.c_[x,devices]))
            mp_pool.close()
            mp_pool.join()
        else:
            for i in range(S):
                fx[i] = obj(np.c_[x[i, :],devices[i])

        # Store particle's best position (if constraints are satisfied)
        i_update = fx > fp
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        # Compare swarm's best position with global best position
        i_max = np.argmax(fp)
        if fp[i_max] > fg:
            if debug:
                print('New best for swarm at iteration {:}: {:} {:}' \
                      .format(it, p[i_max, :], fp[i_max]))

            p_max = p[i_max, :].copy()
            stepsize = np.sqrt(np.sum((g - p_max) ** 2))

            if np.abs(fg - fp[i_max]) >= maxfunc:
                print('Stopping search: Swarm best objective change less than {:}' \
                      .format(maxfunc))
                if particle_output:
                    return p_max, fp[i_max], p, fp
                else:
                    return p_max, fp[i_max]
            elif stepsize <= minstep:
                print('Stopping search: Swarm best position change less than {:}' \
                      .format(minstep))
                if particle_output:
                    return p_max, fp[i_max], p, fp
                else:
                    return p_max, fp[i_max]
            else:
                g = p_max.copy()
                fg = fp[i_max]

        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
        it += 1

    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))

    if particle_output:
        return g, fg, p, fp
    else:
        return g, fg

def myfunc(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    return x1**4 - 2*x2*x1**2 + x2**2 + x1**2 - 2*x1 + x3**3 + 5
if __name__ == '__main__':
    lb = [50, 50, 2]
    ub = [180, 180, 7]

    xopt1, fopt1 = pso(myfunc, lb, ub, debug=True)

    print('The optimum is at:')
    print('    {}'.format(xopt1))
    print('Optimal function value:')
    print('    myfunc: {}'.format(fopt1))