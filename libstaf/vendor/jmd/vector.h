/*
 * vector.h  —  adapted from neuralmdGPU/full_atom/src/vector.h
 * The randomVector function uses gsl_rng; if you don't call it you can
 * compile without -lgsl by defining JMD_NO_GSL before including this header.
 * All other functions have no GSL dependency.
 */
#ifndef VECTOR_H
#define VECTOR_H

#ifndef JMD_NO_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif

typedef struct _vector {
    double x;
    double y;
    double z;
} vector;

vector vectorAdd(const vector *v1, const vector *v2);
vector vectorSub(const vector *v1, const vector *v2);
vector* vectorScale(double scalar, vector *v);
double vectorScalarProduct(const vector *v1, const vector *v2);
double vectorNorm(const vector *v);
double vectorSquareNorm(const vector *v);
vector* vectorVersor(vector *v);
vector* vectorOpposite(vector *v);
vector vectorVectorProduct(const vector *v1, const vector *v2);
void gramSchmidt(vector *v1, vector *v2, vector *v3);
double determinant(double (*m)[3]);
vector matrix_vector_multiplication(double (*m)[3], vector *v);
#ifndef JMD_NO_GSL
void randomVector(vector *rv, gsl_rng *random);
#endif
vector rotateVector(vector *v, vector *axis, double teta);

#endif /* VECTOR_H */
