/* Runge-Kutta integration method 
 * First implementation with Opencl
 *                                  */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable // float64

double func(double y, double t){
    //Def. dy/dt 
    double res;
    double pi = 4.0* atan(1.0);
    res = sin(t)+pi*cos(t);
    return  res;

}


void subroutine_rk4( __global double *y_i, 
                __global double *y_o, 
                double t_i, 
                double t_f, long id_c, long thread ){
    double h;
    h = t_f - t_i; //time step

    double k1;
    double k2;
    double k3;
    double k4;
    
    //Partian steps of integration
    k1 = h * func(y_i[thread],t_i);
    k2 = h * func(y_i[thread] + k1/2.0, t_i+h/2.0);
    k3 = h * func(y_i[thread] + k2/2.0, t_i+h/2.0);
    k4 = h * func(y_i[thread] + k3, t_i+h);


    //Output value 
    y_o[id_c] = y_i[thread] + (k1 + 2.0 * k2 + 2.0 * k3 + k4)/6.0;

/*  rk4    SUBROUTINE END           */
}


/* ---- Kernel FUNCTION ----        */

__kernel void integration(__global double *first, __global double *val_i, 
                        __global double *val_f, double tiempo_i, 
                       double tiempo_f, ulong dimension, __global double *time_it){
    ulong gid = get_global_id(0); //thread id
    ulong tm = get_global_size(0); //size of dim 0-------> |0|1|2|...|n| (count initial values of y)
    val_i[gid] = first[gid]; //initial value of y

    double paso; 
    ulong id_col;
    double time_start = tiempo_i;
    paso = (tiempo_f - tiempo_i)/(dimension-1); //step size
    for (long i = 0; i < dimension; i++){

        id_col = gid+tm*i; //to reach every value next below first 
        tiempo_f = time_start + i * paso;
        time_it[i] = tiempo_f;

        subroutine_rk4(val_i, val_f, tiempo_i, tiempo_f, id_col ,gid); //Exec. integration
        tiempo_i = tiempo_f; //Values out of subtoutine_rk4 -> in subtoutine_rk4
        val_i[gid] = val_f[id_col];
}
    }
