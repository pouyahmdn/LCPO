# distutils: language = c++

from libc.stdlib cimport malloc, free
from libcpp cimport bool
from libc.stdio cimport fopen, fclose, FILE, fseek, SEEK_END, SEEK_SET
from libc.stdio cimport ftell, fread
from cython.operator cimport dereference as deref
from cy_logsort cimport LogSorter
import numpy as np
cimport numpy as np
np.import_array()

LEN_LOG = 33

def sort_logs(filenames, file_out, valid=True, size_max_heap=2000):
    _sort_logs(filenames, file_out, valid, size_max_heap)


def read_constrained_log(filename, offset, arr_limit=250000000, wlen=500):
    ids, timeout_idx, first, arrival, delay, num_instance, instance_index, qu, first_qu, size, model_index, tw, \
        trace_index, duration, first_duration, done = _read_constrained_log(filename, offset, arr_limit, wlen)
    return {
            "ID": ids,
            "TIMEOUT_INDEX": timeout_idx,
            "FIRST": first,
            "ARRIVAL": arrival,
            "DELAY": delay,
            "NUM_INSTANCES": num_instance,
            "INSTANCE_INDEX": instance_index,
            "QUEUE": qu,
            "QUEUE_FIRST": first_qu,
            "SIZE": size,
            "MODEL_INDEX": model_index,
            "TRAINING WHEELS": tw,
            "TRACE_INDEX": trace_index,
            "PROC": duration,
            "PROC_FIRST": first_duration,
    }, done

def read_all_logs(filenames, valid=True):
    count = 0
    for file in filenames:
        count += _count_log(file, valid)
    ids, timeout_idx, first, arrival, delay, num_instance, instance_index, qu, first_qu, size, model_index, tw, \
        trace_index, duration, first_duration = _read_all_logs(filenames, count, valid)
    return {
            "ID": ids,
            "TIMEOUT_INDEX": timeout_idx,
            "FIRST": first,
            "ARRIVAL": arrival,
            "DELAY": delay,
            "NUM_INSTANCES": num_instance,
            "INSTANCE_INDEX": instance_index,
            "QUEUE": qu,
            "QUEUE_FIRST": first_qu,
            "SIZE": size,
            "MODEL_INDEX": model_index,
            "TRAINING WHEELS": tw,
            "TRACE_INDEX": trace_index,
            "PROC": duration,
            "PROC_FIRST": first_duration
    }


def read_log(filename):
    ids, timeout_idx, first, arrival, delay, num_instance, instance_index, qu, first_qu, size, model_index, tw, \
        trace_index, duration, first_duration = _read_log(filename)
    return {
            "ID": ids,
            "TIMEOUT_INDEX": timeout_idx,
            "FIRST": first,
            "ARRIVAL": arrival,
            "DELAY": delay,
            "NUM_INSTANCES": num_instance,
            "INSTANCE_INDEX": instance_index,
            "QUEUE": qu,
            "QUEUE_FIRST": first_qu,
            "SIZE": size,
            "MODEL_INDEX": model_index,
            "TRAINING WHEELS": tw,
            "TRACE_INDEX": trace_index,
            "PROC": duration,
            "PROC_FIRST": first_duration,
    }

cdef _sort_logs(list filenames, str file_out, bool valid, int size_max_heap):
    cdef LogSorter* log_sorter = new LogSorter(file_out.encode('utf-8'), valid, size_max_heap)
    for file_in in filenames:
        log_sorter.add_file(file_in.encode('utf-8'))
    log_sorter.flush()
    del log_sorter

    py_byte_string = file_out.encode('UTF-8')
    cdef char* c_string = py_byte_string
    cdef FILE *fp = fopen(py_byte_string, "r")
    if fp == NULL:
        raise FileNotFoundError(2, "No such file or directory: '%s'" % file_out)
    # get the length of the file
    fseek(fp, 0, SEEK_END)
    file_size = ftell(fp)
    cdef unsigned int array_size = file_size / LEN_LOG
    assert file_size % LEN_LOG == 0
    fseek(fp, 0, SEEK_SET)
    # allocate memory for reading in the file
    cdef char* buffer = <char*>malloc(LEN_LOG*sizeof(char))
    cdef int* id_p = <int*> buffer
    cdef int prev_id = -1

    cdef size_t i
    for i in range(array_size):
        # read file piece into buffer
        assert fread(buffer, 1, LEN_LOG, fp) == LEN_LOG
        assert deref(id_p) > prev_id, f"ID diff at index {i}/{array_size}: prev was {prev_id}, new is: {str_entry(buffer)}"
        prev_id = deref(id_p)

    # close the file once it's read into the char array
    fclose(fp)
    free(buffer)


cdef str_entry(char* buffer):
    return f'ID:{deref(<int*> buffer)}\n' + f'Timeout Index:{deref(<unsigned char*> (buffer+4))}\n' + \
           f'First:{deref(<bool*> (buffer+5))}\n' + f'Arrival:{deref(<float*> (buffer+6))}\n' + \
           f'Delay:{deref(<float*> (buffer+10))}\n' + f'Number of Instances:{deref(<unsigned char*> (buffer+14))}\n' + \
           f'Instance Index:{deref(<unsigned char*> (buffer+15))}\n' + \
           f'Queue:{deref(<unsigned char*> (buffer+16))}\n' + f'First Queue:{deref(<unsigned char*> (buffer+17))}\n' + \
           f'Size:{deref(<float*> (buffer+18))}\n' + \
           f'Model Index:{deref(<unsigned char*> (buffer+22))}\n' + f'TW:{deref(<bool*> (buffer+23))}\n' + \
           f'Trace Index:{deref(<unsigned char*> (buffer+24))}\n' + f'Duration:{deref(<float*> (buffer+25))}\n' + \
           f'First Duration:{deref(<float*> (buffer+29))}\n'


cdef _read_constrained_log(str filename, unsigned int offset, unsigned int arr_limit, double wlen):
    py_byte_string = filename.encode('UTF-8')
    cdef char* c_string = py_byte_string
    cdef FILE *fp = fopen(py_byte_string, "r")
    if fp == NULL:
        raise FileNotFoundError(2, "No such file or directory: '%s'" % filename)
    # get the length of the file
    fseek(fp, 0, SEEK_END)
    file_size = ftell(fp)
    cdef unsigned int array_size = file_size / LEN_LOG
    assert file_size % LEN_LOG == 0
    cdef size_t jump = <size_t>offset * LEN_LOG
    fseek(fp, jump, SEEK_SET)
    # allocate memory for reading in the file
    cdef char* buffer = <char*>malloc(LEN_LOG*sizeof(char))
    cdef int* id_p = <int*> buffer
    cdef unsigned char* timeout_idx_p = <unsigned char*> (buffer+4)
    cdef bool* first_p = <bool*> (buffer+5)
    cdef float* arrival_p = <float*> (buffer+6)
    cdef float* delay_p = <float*> (buffer+10)
    cdef unsigned char* num_instance_p = <unsigned char*> (buffer+14)
    cdef unsigned char* instance_index_p = <unsigned char*> (buffer+15)
    cdef unsigned char* qu_p = <unsigned char*> (buffer+16)
    cdef unsigned char* first_qu_p = <unsigned char*> (buffer+17)
    cdef float* size_p = <float*> (buffer+18)
    cdef unsigned char* model_index_p = <unsigned char*> (buffer+22)
    cdef bool* tw_p = <bool*> (buffer+23)
    cdef unsigned char* trace_index_p = <unsigned char*> (buffer+24)
    cdef float* duration_p = <float*> (buffer+25)
    cdef float* first_duration_p = <float*> (buffer+29)

    assert array_size > offset
    cdef unsigned int max_read = min(array_size-offset, arr_limit)

    cdef int[:] id_view = np.empty(shape=max_read, dtype=np.int32)
    cdef unsigned char[:] timeout_idx_view = np.empty(shape=max_read, dtype=np.uint8)
    cdef bool[:] first_view = np.empty(shape=max_read, dtype=np.bool)
    cdef float[:] arrival_view = np.empty(shape=max_read, dtype=np.float32)
    cdef float[:] delay_view = np.empty(shape=max_read, dtype=np.float32)
    cdef unsigned char[:] num_instance_view = np.empty(shape=max_read, dtype=np.uint8)
    cdef unsigned char[:] instance_index_view = np.empty(shape=max_read, dtype=np.uint8)
    cdef unsigned char[:] qu_view = np.empty(shape=max_read, dtype=np.uint8)
    cdef unsigned char[:] first_qu_view = np.empty(shape=max_read, dtype=np.uint8)
    cdef float[:] size_view = np.empty(shape=max_read, dtype=np.float32)
    cdef unsigned char[:] model_index_view = np.empty(shape=max_read, dtype=np.uint8)
    cdef bool[:] tw_view = np.empty(shape=max_read, dtype=np.bool)
    cdef unsigned char[:] trace_index_view = np.empty(shape=max_read, dtype=np.uint8)
    cdef float[:] duration_view = np.empty(shape=max_read, dtype=np.float32)
    cdef float[:] first_duration_view = np.empty(shape=max_read, dtype=np.float32)

    cdef size_t i
    cdef size_t last_pos = 0
    cdef float last_time = 0
    cdef int last_win = 0
    cdef int last_trace = 0

    for i in range(max_read):
        # read file piece into buffer
        assert fread(buffer, 1, LEN_LOG, fp) == LEN_LOG
        if deref(trace_index_p) != last_trace:
            last_trace = deref(trace_index_p)
            last_time = deref(arrival_p)
            last_pos = i
        elif int((deref(arrival_p)-last_time) // wlen) > last_win:
            last_win = int((deref(arrival_p)-last_time) // wlen)
            last_pos = i
        id_view[i] = deref(id_p)
        timeout_idx_view[i] = deref(timeout_idx_p)
        first_view[i] = deref(first_p)
        arrival_view[i] = deref(arrival_p)
        delay_view[i] = deref(delay_p)
        num_instance_view[i] = deref(num_instance_p)
        instance_index_view[i] = deref(instance_index_p)
        qu_view[i] = deref(qu_p)
        first_qu_view[i] = deref(first_qu_p)
        size_view[i] = deref(size_p)
        model_index_view[i] = deref(model_index_p)
        tw_view[i] = deref(tw_p)
        trace_index_view[i] = deref(trace_index_p)
        duration_view[i] = deref(duration_p)
        first_duration_view[i] = deref(first_duration_p)

    # close the file once it's read into the char array
    fclose(fp)
    free(buffer)

    if max_read == arr_limit:
        return np.asarray(id_view[0:last_pos:1]), \
               np.asarray(timeout_idx_view[0:last_pos:1]), \
               np.asarray(first_view[0:last_pos:1]), \
               np.asarray(arrival_view[0:last_pos:1]), \
               np.asarray(delay_view[0:last_pos:1]), \
               np.asarray(num_instance_view[0:last_pos:1]), \
               np.asarray(instance_index_view[0:last_pos:1]), \
               np.asarray(qu_view[0:last_pos:1]), \
               np.asarray(first_qu_view[0:last_pos:1]), \
               np.asarray(size_view[0:last_pos:1]), \
               np.asarray(model_index_view[0:last_pos:1]), \
               np.asarray(tw_view[0:last_pos:1]), \
               np.asarray(trace_index_view[0:last_pos:1]), \
               np.asarray(duration_view[0:last_pos:1]), \
               np.asarray(first_duration_view[0:last_pos:1]), \
               False
    else:
        return np.asarray(id_view), \
               np.asarray(timeout_idx_view), \
               np.asarray(first_view), \
               np.asarray(arrival_view), \
               np.asarray(delay_view), \
               np.asarray(num_instance_view), \
               np.asarray(instance_index_view), \
               np.asarray(qu_view), \
               np.asarray(first_qu_view), \
               np.asarray(size_view), \
               np.asarray(model_index_view), \
               np.asarray(tw_view), \
               np.asarray(trace_index_view), \
               np.asarray(duration_view), \
               np.asarray(first_duration_view), \
               True


cdef _read_all_logs(list filenames, int array_size, bool valid):
    # allocate memory for reading in the file
    cdef int[:] id_view = np.empty(shape=array_size, dtype=np.int32)
    cdef unsigned char[:] timeout_idx_view = np.empty(shape=array_size, dtype=np.uint8)
    cdef bool[:] first_view = np.empty(shape=array_size, dtype=np.bool)
    cdef float[:] arrival_view = np.empty(shape=array_size, dtype=np.float32)
    cdef float[:] delay_view = np.empty(shape=array_size, dtype=np.float32)
    cdef unsigned char[:] num_instance_view = np.empty(shape=array_size, dtype=np.uint8)
    cdef unsigned char[:] instance_index_view = np.empty(shape=array_size, dtype=np.uint8)
    cdef unsigned char[:] qu_view = np.empty(shape=array_size, dtype=np.uint8)
    cdef unsigned char[:] first_qu_view = np.empty(shape=array_size, dtype=np.uint8)
    cdef float[:] size_view = np.empty(shape=array_size, dtype=np.float32)
    cdef unsigned char[:] model_index_view = np.empty(shape=array_size, dtype=np.uint8)
    cdef bool[:] tw_view = np.empty(shape=array_size, dtype=np.bool)
    cdef unsigned char[:] trace_index_view = np.empty(shape=array_size, dtype=np.uint8)
    cdef float[:] duration_view = np.empty(shape=array_size, dtype=np.float32)
    cdef float[:] first_duration_view = np.empty(shape=array_size, dtype=np.float32)
    cdef size_t i = 0
    cdef char* buffer = <char*>malloc(LEN_LOG*sizeof(char))
    cdef int* id_p = <int*> buffer
    cdef unsigned char* timeout_idx_p = <unsigned char*> (buffer+4)
    cdef bool* first_p = <bool*> (buffer+5)
    cdef float* arrival_p = <float*> (buffer+6)
    cdef float* delay_p = <float*> (buffer+10)
    cdef unsigned char* num_instance_p = <unsigned char*> (buffer+14)
    cdef unsigned char* instance_index_p = <unsigned char*> (buffer+15)
    cdef unsigned char* qu_p = <unsigned char*> (buffer+16)
    cdef unsigned char* first_qu_p = <unsigned char*> (buffer+17)
    cdef float* size_p = <float*> (buffer+18)
    cdef unsigned char* model_index_p = <unsigned char*> (buffer+22)
    cdef bool* tw_p = <bool*> (buffer+23)
    cdef unsigned char* trace_index_p = <unsigned char*> (buffer+24)
    cdef float* duration_p = <float*> (buffer+25)
    cdef float* first_duration_p = <float*> (buffer+29)
    cdef char* c_string
    cdef FILE *fp
    cdef unsigned int row_size

    for filename in filenames:
        py_byte_string = filename.encode('UTF-8')
        c_string = py_byte_string
        fp = fopen(py_byte_string, "r")

        if fp == NULL:
            raise FileNotFoundError(2, "No such file or directory: '%s'" % filename)
        # get the length of the file
        fseek(fp, 0, SEEK_END)
        file_size = ftell(fp)
        row_size = file_size / LEN_LOG
        assert file_size % LEN_LOG == 0
        fseek(fp, 0, SEEK_SET)

        for i_file in range(row_size):
            # read file piece into buffer
            assert fread(buffer, 1, LEN_LOG, fp) == LEN_LOG
            if deref(first_p) == valid:
                id_view[i] = deref(id_p)
                timeout_idx_view[i] = deref(timeout_idx_p)
                first_view[i] = deref(first_p)
                arrival_view[i] = deref(arrival_p)
                delay_view[i] = deref(delay_p)
                num_instance_view[i] = deref(num_instance_p)
                instance_index_view[i] = deref(instance_index_p)
                qu_view[i] = deref(qu_p)
                first_qu_view[i] = deref(first_qu_p)
                size_view[i] = deref(size_p)
                model_index_view[i] = deref(model_index_p)
                tw_view[i] = deref(tw_p)
                trace_index_view[i] = deref(trace_index_p)
                duration_view[i] = deref(duration_p)
                first_duration_view[i] = deref(first_duration_p)
                i += 1

        # close the file once it's read into the char array
        fclose(fp)

    free(buffer)
    return np.asarray(id_view), \
           np.asarray(timeout_idx_view), \
           np.asarray(first_view), \
           np.asarray(arrival_view), \
           np.asarray(delay_view), \
           np.asarray(num_instance_view), \
           np.asarray(instance_index_view), \
           np.asarray(qu_view), \
           np.asarray(first_qu_view), \
           np.asarray(size_view), \
           np.asarray(model_index_view), \
           np.asarray(tw_view), \
           np.asarray(trace_index_view), \
           np.asarray(duration_view), \
           np.asarray(first_duration_view)


cdef _read_log(str filename):
    py_byte_string = filename.encode('UTF-8')
    cdef char* c_string = py_byte_string
    cdef FILE *fp = fopen(py_byte_string, "r")
    if fp == NULL:
        raise FileNotFoundError(2, "No such file or directory: '%s'" % filename)
    # get the length of the file
    fseek(fp, 0, SEEK_END)
    file_size = ftell(fp)
    cdef unsigned int array_size = file_size / LEN_LOG
    assert file_size % LEN_LOG == 0
    fseek(fp, 0, SEEK_SET)
    # allocate memory for reading in the file
    cdef char* buffer = <char*>malloc(LEN_LOG*sizeof(char))
    cdef int* id_p = <int*> buffer
    cdef unsigned char* timeout_idx_p = <unsigned char*> (buffer+4)
    cdef bool* first_p = <bool*> (buffer+5)
    cdef float* arrival_p = <float*> (buffer+6)
    cdef float* delay_p = <float*> (buffer+10)
    cdef unsigned char* num_instance_p = <unsigned char*> (buffer+14)
    cdef unsigned char* instance_index_p = <unsigned char*> (buffer+15)
    cdef unsigned char* qu_p = <unsigned char*> (buffer+16)
    cdef unsigned char* first_qu_p = <unsigned char*> (buffer+17)
    cdef float* size_p = <float*> (buffer+18)
    cdef unsigned char* model_index_p = <unsigned char*> (buffer+22)
    cdef bool* tw_p = <bool*> (buffer+23)
    cdef unsigned char* trace_index_p = <unsigned char*> (buffer+24)
    cdef float* duration_p = <float*> (buffer+25)
    cdef float* first_duration_p = <float*> (buffer+29)

    cdef int[:] id_view = np.empty(shape=array_size, dtype=np.int32)
    cdef unsigned char[:] timeout_idx_view = np.empty(shape=array_size, dtype=np.uint8)
    cdef bool[:] first_view = np.empty(shape=array_size, dtype=np.bool)
    cdef float[:] arrival_view = np.empty(shape=array_size, dtype=np.float32)
    cdef float[:] delay_view = np.empty(shape=array_size, dtype=np.float32)
    cdef unsigned char[:] num_instance_view = np.empty(shape=array_size, dtype=np.uint8)
    cdef unsigned char[:] instance_index_view = np.empty(shape=array_size, dtype=np.uint8)
    cdef unsigned char[:] qu_view = np.empty(shape=array_size, dtype=np.uint8)
    cdef unsigned char[:] first_qu_view = np.empty(shape=array_size, dtype=np.uint8)
    cdef float[:] size_view = np.empty(shape=array_size, dtype=np.float32)
    cdef unsigned char[:] model_index_view = np.empty(shape=array_size, dtype=np.uint8)
    cdef bool[:] tw_view = np.empty(shape=array_size, dtype=np.bool)
    cdef unsigned char[:] trace_index_view = np.empty(shape=array_size, dtype=np.uint8)
    cdef float[:] duration_view = np.empty(shape=array_size, dtype=np.float32)
    cdef float[:] first_duration_view = np.empty(shape=array_size, dtype=np.float32)

    cdef size_t i
    for i in range(array_size):
        # read file piece into buffer
        assert fread(buffer, 1, LEN_LOG, fp) == LEN_LOG
        id_view[i] = deref(id_p)
        timeout_idx_view[i] = deref(timeout_idx_p)
        first_view[i] = deref(first_p)
        arrival_view[i] = deref(arrival_p)
        delay_view[i] = deref(delay_p)
        num_instance_view[i] = deref(num_instance_p)
        instance_index_view[i] = deref(instance_index_p)
        qu_view[i] = deref(qu_p)
        first_qu_view[i] = deref(first_qu_p)
        size_view[i] = deref(size_p)
        model_index_view[i] = deref(model_index_p)
        tw_view[i] = deref(tw_p)
        trace_index_view[i] = deref(trace_index_p)
        duration_view[i] = deref(duration_p)
        first_duration_view[i] = deref(first_duration_p)

    # close the file once it's read into the char array
    fclose(fp)
    free(buffer)

    return np.asarray(id_view), \
           np.asarray(timeout_idx_view), \
           np.asarray(first_view), \
           np.asarray(arrival_view), \
           np.asarray(delay_view), \
           np.asarray(num_instance_view), \
           np.asarray(instance_index_view), \
           np.asarray(qu_view), \
           np.asarray(first_qu_view), \
           np.asarray(size_view), \
           np.asarray(model_index_view), \
           np.asarray(tw_view), \
           np.asarray(trace_index_view), \
           np.asarray(duration_view), \
           np.asarray(first_duration_view)


def count_log(filename, valid=True):
    return _count_log(filename, valid)

cdef _count_log(str filename, bool valid):
    py_byte_string = filename.encode('UTF-8')
    cdef char* c_string = py_byte_string
    cdef FILE *fp = fopen(py_byte_string, "r")
    if fp == NULL:
        raise FileNotFoundError(2, "No such file or directory: '%s'" % filename)
    # get the length of the file
    fseek(fp, 0, SEEK_END)
    file_size = ftell(fp)
    cdef unsigned int array_size = file_size / LEN_LOG
    assert file_size % LEN_LOG == 0
    fseek(fp, 0, SEEK_SET)
    # allocate memory for reading in the file
    cdef char* buffer = <char*>malloc(LEN_LOG*sizeof(char))
    cdef bool* first_p = <bool*> (buffer+5)
    cdef int count = 0

    cdef size_t i
    for i in range(array_size):
        # read file piece into buffer
        assert fread(buffer, 1, LEN_LOG, fp) == LEN_LOG
        if valid == deref(first_p):
            count += 1

    # close the file once it's read into the char array
    fclose(fp)
    free(buffer)

    return count
