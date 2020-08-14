#include <math.h>
double PI;
int ncomp;
double delta_x;

double* rd_hom_rhs(double *a, double *x, double *y, double *ydot) {
  int i; double dx_2;
  *(ydot++)=0;
  dx_2=(delta_x*delta_x); 
  for (i=1;i<ncomp-1;i++) {
    // if (i==ncomp-2){
    //  printf("a:%lf, %lf, %lf, %lf\n",a[i-3],a[i-2],a[i-1],a[i]);
    //  printf("y:%lf, %lf, %lf, %lf\n",y[i-3],y[i-2],y[i-1],y[i]);
    //  printf("x:%lf, %lf, %lf, %lf\n",x[i-3],x[i-2],x[i-1],x[i]);
    // }
    *(ydot++)=(a[i]*(y[i+1]-y[i])-a[i-1]*(y[i]-y[i-1]));
  }
  *(ydot++)=0;
}
