# distutils: language = c++

from cy_jobgensim cimport JobGenSim, dist_type, dist_param
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport uintptr_t

def get_c_dist_param(str type_dist, **kwargs):
    """
    :param type_dist:
    :type type_dist: str
    """
    cdef dist_param c_dist_param
    if type_dist == 'pareto':
        t = dist_type.pareto
        c_dist_param.pareto_scale = kwargs['pareto_scale']
        c_dist_param.pareto_shape = kwargs['pareto_shape']
    elif  type_dist == 'normal':
        t = dist_type.normal
        c_dist_param.normal_mu = kwargs['normal_mu']
        c_dist_param.normal_sigma = kwargs['normal_sigma']
    elif  type_dist == 'exponential':
        t = dist_type.exponential
        c_dist_param.exponential_rev_lambda = kwargs['exponential_rev_lambda']
    elif  type_dist == 'static':
        t = dist_type.stat_ret
        c_dist_param.stat_ret_val = kwargs['stat_ret_val']
    else:
        raise ValueError
    c_dist_param.type = t
    return c_dist_param

cdef class PyJobGenSim:
    cdef JobGenSim* c_job_gen
    cdef int num_states

    def __init__(self, int start_state, double hold_time, list list_dist_params, int num_servers, unsigned int seed_rng):
        """
        :param start_state:
        :param hold_time:
        :param num_servers:
        :param seed_rng:
        :param list_dist_params:
        :type start_state: int
        :type hold_time: float
        :type num_servers: int
        :type seed_rng: int
        :type list_dist_params: list[dist_param]
        """
        pass

    def __cinit__(self, int start_state, double hold_time, list list_dist_params, int num_servers, unsigned int seed_rng):
        """
        :param start_state:
        :param hold_time:
        :param num_servers:
        :param seed_rng:
        :param list_dist_params:
        :type start_state: int
        :type hold_time: float
        :type num_servers: int
        :type seed_rng: int
        :type list_dist_params: list[dist_param]
        """
        self.num_states = len(list_dist_params)
        cdef dist_param* dist_param_size = <dist_param*> PyMem_Malloc(self.num_states * sizeof(dist_param))
        cdef dist_param* dist_param_iat = <dist_param*> PyMem_Malloc(self.num_states * sizeof(dist_param))
        for i in range(self.num_states):
            dist_param_iat[i] = get_c_dist_param(list_dist_params[i]['iat']['type'], **list_dist_params[i]['iat']['params'])
            dist_param_size[i] = get_c_dist_param(list_dist_params[i]['size']['type'], **list_dist_params[i]['size']['params'])
        self.c_job_gen = new JobGenSim(start_state, hold_time, self.num_states, dist_param_size, dist_param_iat, num_servers, seed_rng)
        PyMem_Free(dist_param_size)
        PyMem_Free(dist_param_iat)

    def gen_job(self):
        """
        :return:
        :rtype: dict[str, float]
        """
        return self.c_job_gen.gen_job()

    cpdef get_ptr(self):
        """
        :return:
        :rtype: uintptr_t 
        """
        return <uintptr_t> self.c_job_gen

    def seed(self, unsigned int seed):
        """
        :param seed:
        :type seed: int
        """
        assert seed > 0, "seed must be unsigned int"
        cdef unsigned int s = seed
        self.c_job_gen.seed(s)

    def get_curr_trace(self):
        """
        :return:
        :rtype: int
        """
        return self.c_job_gen.get_curr_trace()

    def set_curr_trace(self, int new_trace):
        """

        :param new_trace:
        :type new_trace: int
        """
        assert new_trace < self.num_states
        self.c_job_gen.set_curr_trace(new_trace)

    def get_num_traces(self):
        """
        :return:
        :rtype: int
        """
        return self.num_states

    def get_rate(self):
        """
        :return:
        :rtype: float
        """
        return 1

    def reset(self):
        self.c_job_gen.reset()

    def relocate(self):
        self.c_job_gen.relocate()

    def save_state(self):
        self.c_job_gen.save_state()

    def load_state(self):
        self.c_job_gen.load_state()

    def __dealloc__(self):
        del self.c_job_gen


