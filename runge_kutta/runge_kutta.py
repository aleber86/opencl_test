#Runge-Kutta 4th order (for low non-linear equations)
#Code translated from FORTRAN 77 to Opencl
#Meant for academic purpose only ('toy kernel')

import numpy as np
import pyopencl as cl

#Program variable values
dim = np.int64(10**5)  #Set Steps of integration and dmiension of values
#***************************************************************************
#Do not change values below
_wp = np.float64 #Working precision
_dim = np.int64(dim+1) #Steps for integration and matrix result dimension
#USE ODD NUMBERS FOR _dim
#****************************************************************************
#Set the initial conditions of the integration. y(in_timet) = in_first
in_first = np.asarray([np.random.random() for i in range(20)]).astype(_wp)
in_time = np.float64(-5.0)

out_time = np.float64(3.0)
#-----------------------------------------------------------------------
#Safty cond.
number_in_val = in_first.size
byte_per_dig = np.asarray(0).astype(_wp).nbytes
mem_alloc = in_first.nbytes * 2 + number_in_val*_dim*byte_per_dig #Size of memory to alloc.
print('Memory block alloc. dev. : ',mem_alloc,'Bytes','\t', mem_alloc/1024**2,'MB')
if mem_alloc >= 1024**3:
    print('Max mem alloc exceeded')
    exit(-1)
elif number_in_val >= 2**11:
    #Depends on the size of the problem. (Watchdog timer)
    print('Max num of initial values exceeded')
    exit(-1)
#---------------------------------------------------------------------------
out_values = np.empty((number_in_val, _dim)).astype(_wp)
in_values = np.empty_like(in_first).astype(_wp)
time_matrix = np.empty((1, _dim)).astype(_wp)
#***************************************************************************
#Memory buffer to device

context = cl.create_some_context()
queue = cl.CommandQueue(context)


mem_f = cl.mem_flags

initial_val_y = cl.Buffer(context, mem_f.READ_ONLY | mem_f.COPY_HOST_PTR, hostbuf = in_first)
val_i = cl.Buffer(context, mem_f.WRITE_ONLY | mem_f.COPY_HOST_PTR, hostbuf = in_values)
val_o = cl.Buffer(context, mem_f.WRITE_ONLY | mem_f.COPY_HOST_PTR, hostbuf = out_values)
time_m = cl.Buffer(context, mem_f.WRITE_ONLY | mem_f.COPY_HOST_PTR, hostbuf = time_matrix)
#****************************************************************************
#Kernel load:

with open('kernel_runge_kutta.cl', 'r') as kernel_file:
    kernel_script = kernel_file.read()

program = cl.Program(context, kernel_script).build()
ker = program.integration

ker(queue,in_first.shape, None ,initial_val_y, val_i, val_o ,in_time, out_time, _dim, time_m)

queue.finish()

#Transfer results from device:
#time_matrix, out_values are empty from definition
cl.enqueue_copy(queue, time_matrix, time_m)
cl.enqueue_copy(queue, out_values , val_o)

out_values = out_values.reshape(_dim, number_in_val)
time_matrix =  time_matrix.reshape(_dim, 1)

to_file = np.column_stack((time_matrix,out_values))

np.savetxt('datos_int.dat',to_file, fmt='%.18e', delimiter='  ')
