cdef extern from "src/JobGenSim.h":
    cdef enum dist_type:
        normal, exponential, pareto, stat_ret
    cdef struct dist_param:
        dist_type type
        double normal_mu
        double normal_sigma
        double exponential_rev_lambda
        double pareto_shape
        double pareto_scale
        double stat_ret_val
    cdef struct JobProcessSample:
        double size
        double inter_arrival_time
    cdef cppclass JobGenSim:
        JobGenSim(int, double, int, dist_param*, dist_param*, int, unsigned int) except +
        void seed(unsigned int)
        int get_curr_trace()
        void set_curr_trace(int)
        void relocate()
        void reset()
        JobProcessSample gen_job()
        void save_state()
        void load_state()