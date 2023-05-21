# distutils: language = c++

from cy_env cimport LoadBalanceEnv, JobGen, step_return
from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free
from libcpp cimport bool
import numpy as np
cimport numpy as np
np.import_array()

cdef class PyLoadBalanceEnv:
    cdef LoadBalanceEnv* c_load_balance_env
    cdef int obs_size
    cdef int num_servers
    cdef int MAX_WINDOW_SIZE
    cdef double* obs
    cdef double* server_time
    cdef unsigned short* queue_lens
    cdef double* finished_job_completion_times
    cdef double* arrived_proc_time
    cdef double* arrive_iat_time
    cdef object job_gen

    def __init__(self, unsigned short num_servers, double time_window, bool load_in_obs, bool act_in_obs,
                 bool trace_in_obs, bool ext_in_obs, str filename_log, list timeouts, list service_rates,
                 int max_retries, bool use_tw, unsigned short upper_unsafe_bound, unsigned short lower_safe_bound,
                 object job_gen, uintptr_t job_gen_pointer, unsigned int seed_rng, bool skip_log):
        pass

    def __cinit__(self, unsigned short num_servers, double time_window, bool load_in_obs, bool act_in_obs,
                  bool trace_in_obs, bool ext_in_obs, str filename_log, list timeouts, list service_rates,
                  int max_retries, bool use_tw, unsigned short upper_unsafe_bound, unsigned short lower_safe_bound,
                  object job_gen, uintptr_t job_gen_pointer, unsigned int seed_rng, bool skip_log):
        """

        :param num_servers:
        :param time_window:
        :param load_in_obs:
        :param act_in_obs:
        :param trace_in_obs:
        :param ext_in_obs:
        :param filename_log:
        :param timeouts:
        :param service_rates:
        :param max_retries:
        :param use_tw:
        :param upper_unsafe_bound:
        :param lower_safe_bound:
        :param job_gen:
        :param job_gen_pointer:
        :param seed_rng:
        :param skip_log:
        :type num_servers: int
        :type time_window: float
        :type load_in_obs: bool
        :type act_in_obs: bool
        :type trace_in_obs: bool
        :type ext_in_obs: bool
        :type filename_log: str
        :type timeouts: list[int]
        :type service_rates: list[float]
        :type max_retries: int
        :type use_tw: bool
        :type upper_unsafe_bound: int
        :type lower_safe_bound: int
        :type job_gen: pyjobgenfile.PyJobGenFile or pyjobgensim.PyJobGenSim
        :type job_gen_pointer: uintptr_t
        :type seed_rng: int
        :type skip_log: bool
        """
        cdef short* c_timeouts = <short*> malloc(sizeof(short)*len(timeouts))
        cdef short[::1] c_timeouts_view = <short[:len(timeouts)]> c_timeouts
        for i in range(len(timeouts)):
            c_timeouts_view[i] = timeouts[i]
        cdef double* c_service_rates = <double*> malloc(sizeof(double)*len(service_rates))
        cdef double[::1] c_service_rates_view = <double[:len(service_rates)]> c_service_rates
        for i in range(len(service_rates)):
            c_service_rates_view[i] = service_rates[i]

        self.job_gen = job_gen

        self.c_load_balance_env = new LoadBalanceEnv(seed_rng, num_servers, time_window, load_in_obs, ext_in_obs,
                                                     filename_log.encode('utf-8'), act_in_obs, trace_in_obs, c_timeouts,
                                                     len(timeouts), c_service_rates, len(service_rates), max_retries,
                                                     <JobGen*>job_gen_pointer, use_tw, upper_unsafe_bound,
                                                     lower_safe_bound, skip_log)
        free(<void*> c_timeouts)
        free(<void*> c_service_rates)

        self.obs_size = self.c_load_balance_env.observation_len()
        self.num_servers = num_servers
        self.MAX_WINDOW_SIZE = self.c_load_balance_env.MAX_WINDOW_SIZE

        self.obs = <double*> malloc(sizeof(double)*self.obs_size)
        self.server_time = <double*> malloc(sizeof(double)*num_servers)
        self.queue_lens = <unsigned short*> malloc(sizeof(unsigned short)*num_servers)
        self.finished_job_completion_times = <double*> malloc(sizeof(double)*self.MAX_WINDOW_SIZE)
        self.arrived_proc_time = <double*> malloc(sizeof(double)*self.MAX_WINDOW_SIZE)
        self.arrive_iat_time = <double*> malloc(sizeof(double)*self.MAX_WINDOW_SIZE)

    def observe(self):
        """
        :return:
        :rtype: np.ndarray
        """
        self.c_load_balance_env.observe(self.obs)
        return np.asarray(<double[:self.obs_size]> self.obs)

    def get_observation_len(self):
        """
        :return:
        :rtype: int
        """
        return self.obs_size

    def get_avg_rate(self):
        """
        :return:
        :rtype: float
        """
        return self.c_load_balance_env.get_avg_rate()

    def reset(self):
        """
        :return:
        :rtype: np.ndarray
        """
        self.c_load_balance_env.reset(self.obs)
        return np.asarray(<double[:self.obs_size]> self.obs)

    def step(self, unsigned short act, unsigned short model_index):
        """
        :param act:
        :param model_index: int
        :type act: int
        :type model_index: int
        :return:
        :rtype: (np.ndarray, float, bool, dict[str])
        """
        cdef step_return ret = self.c_load_balance_env.step(act, model_index, self.server_time,
                                                            self.finished_job_completion_times, self.arrive_iat_time,
                                                            self.arrived_proc_time, self.obs)
        assert ret.num_arrived_jobs <= self.MAX_WINDOW_SIZE
        assert ret.num_finished_jobs <= self.MAX_WINDOW_SIZE
        cdef np.ndarray[double, ndim=1, mode="c"] arr_job_proc_time
        cdef np.ndarray[double, ndim=1, mode="c"] arr_job_inter_time
        if ret.num_arrived_jobs > 0:
            arr_job_proc_time = np.asarray(<double[:ret.num_arrived_jobs]> self.arrived_proc_time)
            arr_job_inter_time = np.asarray(<double[:ret.num_arrived_jobs]> self.arrive_iat_time)
        else:
            arr_job_proc_time = np.empty(shape=0, dtype=np.double)
            arr_job_inter_time = arr_job_proc_time
        return np.asarray(<double[:self.obs_size]> self.obs), \
               ret.reward, \
               ret.done, \
               {
                   "arrived_job_proc_time": arr_job_proc_time,
                   "arrived_job_inter_time": arr_job_inter_time,
                   "rew_vec_orig": np.asarray(<double[:ret.num_finished_jobs]> self.finished_job_completion_times),
                   "avg_arrived_job_proc_time": ret.avg_arrived_job_proc_time,
                   "avg_arrived_job_inter_time": ret.avg_arrived_job_inter_time,
                   "server_time": np.asarray(<double[:self.num_servers]> self.server_time),
                   "time_elapsed": ret.time_elapsed,
                   "curr_time": ret.curr_time,
                   "unsafe": ret.unsafe
               }

    def timeline_len(self):
        """
        :return:
        :rtype: int
        """
        return self.c_load_balance_env.timeline_len()

    def get_env_job_gen(self):
        """
        :return:
        :rtype: pyjobgenfile.PyJobGenFile or pyjobgensim.PyJobGenSim
        """
        return self.job_gen

    def queue_sizes(self):
        """
        :return:
        :rtype: np.ndarray
        """
        self.c_load_balance_env.queue_sizes(self.queue_lens)
        return np.asarray(<unsigned short[:self.num_servers]> self.queue_lens)

    def seed(self, unsigned int seed):
        """
        :param seed:
        :type seed: int
        """
        assert seed >= 0, "seed must be unsigned int"
        cdef unsigned int s = seed
        self.c_load_balance_env.seed(s)

    def close(self):
        self.c_load_balance_env.close()

    def get_scales(self):
        """
        :return:
        :rtype: (float, float)
        """
        return self.c_load_balance_env.get_arrival_scale(), self.c_load_balance_env.get_size_scale()

    def set_scales(self, double arrival_scale=0, double size_scale=0):
        """
        :param arrival_scale:
        :param size_scale:
        :type arrival_scale: float
        :type size_scale: float
        """
        assert size_scale > 0 or arrival_scale > 0
        if size_scale > 0:
            self.c_load_balance_env.set_size_scale(size_scale)
        if arrival_scale > 0:
            self.c_load_balance_env.set_arrival_scale(arrival_scale)

    def get_work_measure(self):
        """
        :return:
        :rtype: float
        """
        return self.c_load_balance_env.get_work_measure()

    # Attribute access
    @property
    def MAX_WINDOW_SIZE(self):
        return self.c_load_balance_env.MAX_WINDOW_SIZE

    def __dealloc__(self):
        free(<void*> self.obs)
        free(<void*> self.server_time)
        free(<void*> self.queue_lens)
        free(<void*> self.finished_job_completion_times)
        free(<void*> self.arrived_proc_time)
        free(<void*> self.arrive_iat_time)
        del self.c_load_balance_env
        self.job_gen = None
