%module rhs
%{
#include <Python.h>
#include <Numeric/arrayobject.h>
  %}
%typemap(python,in) (double *) {
   $1=(double *)((PyArrayObject *) ($input))->data;
}

extern void rd_hom_rhs(double *, double *,double *, double *);
extern int ncomp;
extern double delta_x;
extern double PI;
