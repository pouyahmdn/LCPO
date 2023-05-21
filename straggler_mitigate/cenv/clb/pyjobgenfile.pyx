# distutils: language = c++

from cy_jobgenfile cimport JobGenFile
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport uintptr_t
import numpy as np
cimport numpy as np
np.import_array()

cdef class PyJobGenFile:
    cdef JobGenFile* c_job_gen
    cdef int num_states
    cdef int* c_len_arrs
    cdef double rate_chosen

    def __init__(self, list size_arrs, list arrival_arrs, int start_state, double hold_time, int num_servers,
                  unsigned int seed_rng, double rate_chosen):
        """
        :param size_arrs:
        :param arrival_arrs:
        :param start_state:
        :param hold_time:
        :param num_servers:
        :param seed_rng:
        :param rate_chosen:
        :type size_arrs: list[np.ndarray]
        :type arrival_arrs: list[np.ndarray]
        :type start_state: int
        :type hold_time: float
        :type num_servers: int
        :type seed_rng: int
        :type rate_chosen: float
        """
        pass

    def __cinit__(self, list size_arrs, list arrival_arrs, int start_state, double hold_time, int num_servers,
                  unsigned int seed_rng, double rate_chosen):
        """
        :param size_arrs:
        :param arrival_arrs:
        :param start_state:
        :param hold_time:
        :param num_servers:
        :param seed_rng:
        :param rate_chosen:
        :type size_arrs: list[np.ndarray]
        :type arrival_arrs: list[np.ndarray]
        :type start_state: int
        :type hold_time: float
        :type num_servers: int
        :type seed_rng: int
        :type rate_chosen: float
        """
        self.num_states = len(size_arrs)
        assert len(arrival_arrs) == self.num_states
        cdef double** c_size_arrs = <double**> PyMem_Malloc(self.num_states * sizeof(double*))
        cdef double** c_arrival_arrs = <double**> PyMem_Malloc(self.num_states * sizeof(double*))
        self.c_len_arrs = <int*> PyMem_Malloc(self.num_states * sizeof(int))
        cdef np.ndarray[double, ndim=1, mode="c"] size_cont
        cdef np.ndarray[double, ndim=1, mode="c"] arrival_cont

        for i in range(self.num_states):
            size_cont = np.ascontiguousarray(size_arrs[i], dtype=np.double)
            arrival_cont = np.ascontiguousarray(arrival_arrs[i], dtype=np.double)
            assert size_cont.ndim == 1
            assert arrival_cont.ndim == 1
            assert size_cont.shape[0] == arrival_cont.shape[0]
            self.c_len_arrs[i] = size_cont.shape[0]
            c_size_arrs[i] = &size_cont[0]
            c_arrival_arrs[i] = &arrival_cont[0]

        self.c_job_gen = new JobGenFile(c_size_arrs, c_arrival_arrs, self.c_len_arrs, self.num_states, start_state,
                                        hold_time, num_servers, seed_rng, rate_chosen)
        self.rate_chosen = rate_chosen

        PyMem_Free(c_size_arrs)
        PyMem_Free(c_arrival_arrs)

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

    def get_rate(self):
        """
        :return:
        :rtype: float
        """
        return self.rate_chosen

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

    def load_index_arr(self, list index_arrs):
        """
        :param index_arrs:
        :type index_arrs: list[np.ndarray]
        """
        assert len(index_arrs) == self.num_states
        cdef short** c_index_arrs = <short**> PyMem_Malloc(self.num_states * sizeof(short*))
        cdef np.ndarray[np.int16_t, ndim=1, mode="c"] index_cont

        for i in range(self.num_states):
            index_cont = np.ascontiguousarray(index_arrs[i], dtype=np.int16)
            assert index_cont.ndim == 1
            assert index_cont.shape[0] == self.c_len_arrs[i]
            c_index_arrs[i] = <short*> &index_cont[0]

        self.c_job_gen.load_index_arr(c_index_arrs)

        PyMem_Free(c_index_arrs)

    def get_curr_index(self):
        """
        :return:
        :rtype: int
        """
        return self.c_job_gen.get_curr_index()

    def seek(self, double time_point):
        self.c_job_gen.seek(time_point)

    def get_num_traces(self):
        """
        :return:
        :rtype: int
        """
        return self.num_states

    def reset(self):
        self.c_job_gen.reset()

    def relocate(self):
        self.c_job_gen.relocate()

    def save_state(self):
        self.c_job_gen.save_state()

    def load_state(self):
        self.c_job_gen.load_state()

    def __dealloc__(self):
        PyMem_Free(self.c_len_arrs)
        del self.c_job_gen


