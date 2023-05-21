from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "src/LoadBalanceEnv.h":
    cdef cppclass JobGen:
        pass

    cdef struct step_return:
        unsigned short num_finished_jobs
        bool done
        double* next_obs
        double* server_time
        double time_elapsed
        double* finished_job_completion_times
        double reward
        double* arrived_job_inter_time
        double* arrived_job_proc_time
        double avg_arrived_job_inter_time
        double avg_arrived_job_proc_time
        unsigned short num_arrived_jobs
        double curr_time
        bool unsafe

    cdef cppclass LoadBalanceEnv:
        LoadBalanceEnv(unsigned int, unsigned short, double, bool, bool, const string&, bool, bool, short*,
                       unsigned short, double*, unsigned short, int, JobGen*, bool, unsigned short, unsigned short, bool) except +
        const int MAX_WINDOW_SIZE;
        const int MAX_WINDOWS;
        const int MAX_WORK_MEASURE_WINDOWS;
        void seed(unsigned int)
        void reset(double*)
        double get_arrival_scale()
        double get_size_scale()
        void set_arrival_scale(double)
        void set_size_scale(double)
        double get_avg_rate()
        void observe(double*)
        void close()
        double get_work_measure()
        void queue_sizes(unsigned short*)
        unsigned int observation_len()
        unsigned int timeline_len()
        step_return step(unsigned short, unsigned short, double*, double*, double*, double*, double*)