from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "src/LogSort.h":
    cdef cppclass LogSorter:
        LogSorter(const string&, bool, unsigned int) except +
        void add_file(const string&)
        void flush()
