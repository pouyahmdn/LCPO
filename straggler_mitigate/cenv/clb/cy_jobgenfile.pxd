cdef extern from "src/JobGenFile.h":
    cdef struct JobProcessSample:
        double size
        double inter_arrival_time

    cdef cppclass JobGenFile:
        JobGenFile(double**, double**, const int*, short, int, double, int, unsigned int, double) except +
        void seed(unsigned int)
        int get_curr_trace()
        void set_curr_trace(int)
        void load_index_arr(short**)
        int get_curr_index()
        void seek(double)
        void relocate()
        void reset()
        JobProcessSample gen_job()
        void save_state()
        void load_state()