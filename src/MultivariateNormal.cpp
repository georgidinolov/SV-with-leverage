#include <armadillo>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <cmath>
#include "MultivariateNormal.hpp"

void rmvnorm(const gsl_rng *r, 
	     const int n, 
	     const gsl_vector *mean, 
	     const gsl_matrix *var, 
	     gsl_vector *result){
  /* multivariate normal distribution random number generator */
  /*
   *	n	dimension of the random vetor
   *	mean	vector of means of size n
   *	var	variance matrix of dimension n x n
   *	result	output variable with a sigle random vector normal distribution generation
   */
  int k;
  gsl_matrix *work = gsl_matrix_alloc(n,n);
  
  gsl_matrix_memcpy(work,var);
  gsl_linalg_cholesky_decomp(work);
  
  for(k=0; k<n; k++) {
    gsl_vector_set( result, k, gsl_ran_ugaussian(r) );
  }
    
  gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, work, result);
  gsl_vector_add(result,mean);
  
  gsl_matrix_free(work);
}

arma::vec rmvnorm(const gsl_rng *r,
		  const int n,
		  arma::vec mean,
		  arma::mat var) 
{
  gsl_vector * mu = gsl_vector_alloc(n);
  gsl_vector * result = gsl_vector_alloc(n);
  gsl_matrix * Sigma = gsl_matrix_alloc(n,n);
  
  // Assigning mean and covariance to gsl objects memory
  for (int k=0; k<n; ++k) {
    gsl_vector_set(mu, k, mean(k));
    for (int l=0; l<n; ++l) {
      gsl_matrix_set(Sigma,k,l,var(k,l));
    }
  }

  rmvnorm(r, n, mu, Sigma, result);
  
  arma::vec output = arma::vec (n);
  for (int k=0; k<n; ++k) {  
    output(k) = gsl_vector_get(result, k);
  }
  
  gsl_vector_free(mu);
  gsl_vector_free(result);
  gsl_matrix_free(Sigma);
  return output;
}

double dmvnorm(const int n, const gsl_vector *x, const gsl_vector *mean, const gsl_matrix *var){
  /* multivariate normal density function    */
  /*
   *	n	dimension of the random vetor
   *	mean	vector of means of size n
   *	var	variance matrix of dimension n x n
   */
  int s;
  double ax,ay;
  gsl_vector *ym, *xm;
  gsl_matrix *work = gsl_matrix_alloc(n,n), 
    *winv = gsl_matrix_alloc(n,n);
  gsl_permutation *p = gsl_permutation_alloc(n);
  
  gsl_matrix_memcpy( work, var );
  gsl_linalg_LU_decomp( work, p, &s );
  gsl_linalg_LU_invert( work, p, winv );
  ax = gsl_linalg_LU_det( work, s );
  gsl_matrix_free( work );
  gsl_permutation_free( p );
  
  xm = gsl_vector_alloc(n);
  gsl_vector_memcpy( xm, x);
  gsl_vector_sub( xm, mean );
  ym = gsl_vector_alloc(n);
  gsl_blas_dsymv(CblasUpper,1.0,winv,xm,0.0,ym);
  gsl_matrix_free( winv );
  gsl_blas_ddot( xm, ym, &ay);
  gsl_vector_free(xm);
  gsl_vector_free(ym);
  ay = exp(-0.5*ay)/sqrt( pow((2*M_PI),n)*ax );
  
  return ay;
}

double dmvnorm_log(const int n, 
		   const gsl_vector *x, 
		   const gsl_vector *mean, 
		   const gsl_matrix *var){
  /* multivariate normal density function    */
  /*
   *	n	dimension of the random vetor
   *	mean	vector of means of size n
   *	var	variance matrix of dimension n x n
   */

  int s;
  double ax,ay;
  gsl_vector *ym, *xm;
  gsl_matrix *work = gsl_matrix_alloc(n,n), 
    *winv = gsl_matrix_alloc(n,n);
  gsl_permutation *p = gsl_permutation_alloc(n);
  
  gsl_matrix_memcpy( work, var );
  gsl_linalg_LU_decomp( work, p, &s );
  gsl_linalg_LU_invert( work, p, winv );
  ax = gsl_linalg_LU_det( work, s );
  gsl_matrix_free( work );
  gsl_permutation_free( p );
  
  xm = gsl_vector_alloc(n);
  gsl_vector_memcpy( xm, x);
  gsl_vector_sub( xm, mean );
  ym = gsl_vector_alloc(n);
  gsl_blas_dsymv(CblasUpper,1.0,winv,xm,0.0,ym);
  gsl_matrix_free( winv );
  gsl_blas_ddot( xm, ym, &ay);
  gsl_vector_free(xm);
  gsl_vector_free(ym);
  ay = (-0.5*ay)- 0.5*log( pow((2*M_PI),n)*ax );
  
  return ay;
}

double dmvnorm_log(const int n, 
		   const std::vector<double>& xx, 
		   const std::vector<double>& mmean, 
		   const gsl_matrix *var){
  /* multivariate normal density function    */
  /*
   *	n	dimension of the random vetor
   *	mean	vector of means of size n
   *	var	variance matrix of dimension n x n
   */

  int s;
  double ax,ay;
  gsl_vector *ym, *xm;

  gsl_vector *x = gsl_vector_alloc(n);
  gsl_vector *mean = gsl_vector_alloc(n);
  for (int i=0; i<n; ++i) {
    gsl_vector_set(x, i, xx[i]);
    gsl_vector_set(mean, i, mmean[i]);
  }

  gsl_matrix *work = gsl_matrix_alloc(n,n), 
    *winv = gsl_matrix_alloc(n,n);
  gsl_permutation *p = gsl_permutation_alloc(n);
  
  gsl_matrix_memcpy( work, var );
  gsl_linalg_LU_decomp( work, p, &s );
  gsl_linalg_LU_invert( work, p, winv );
  ax = gsl_linalg_LU_det( work, s );
  gsl_matrix_free( work );
  gsl_permutation_free( p );
  
  xm = gsl_vector_alloc(n);
  gsl_vector_memcpy( xm, x);
  gsl_vector_sub( xm, mean );
  ym = gsl_vector_alloc(n);
  gsl_blas_dsymv(CblasUpper,1.0,winv,xm,0.0,ym);
  gsl_matrix_free( winv );
  gsl_blas_ddot( xm, ym, &ay);
  gsl_vector_free(xm);
  gsl_vector_free(ym);
  ay = -0.5*ay - 0.5*(log( pow((2*M_PI),n) ) +
		      log( ax ));

  gsl_vector_free(x);
  gsl_vector_free(mean);

  return ay;
}

int rmvt(const gsl_rng *r, 
	 const int n, 
	 const gsl_vector *location, 
	 const gsl_matrix *scale, 
	 const int dof, 
	 gsl_vector *result){
  /* multivariate Student t distribution random number generator */
  /*
   *	n	 dimension of the random vetor
   *	location vector of locations of size n
   *	scale	 scale matrix of dimension n x n
   *	dof	 degrees of freedom
   *	result	 output variable with a single random vector normal distribution generation
   */
  int k;
  gsl_matrix *work = gsl_matrix_alloc(n,n);
  double ax = 0.5*dof; 
  
  ax = gsl_ran_gamma(r,ax,(1/ax));     /* gamma distribution */
  
  gsl_matrix_memcpy(work,scale);
  gsl_matrix_scale(work,(1/ax));       /* scaling the matrix */
  gsl_linalg_cholesky_decomp(work);
  
  for(k=0; k<n; k++)
    gsl_vector_set( result, k, gsl_ran_ugaussian(r) );
  
  gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, work, result);
  gsl_vector_add(result, location);
  
  gsl_matrix_free(work);
  
  return 0;
}

double dmvt_log(const int n, 
		const gsl_vector *x, 
		const gsl_vector *location, 
		const gsl_matrix *scale, 
		const int dof){
  /* multivariate Student t density function */
  /*
   *	n	 dimension of the random vetor
   *	location vector of locations of size n
   *	scale	 scale matrix of dimension n x n
   *	dof	 degrees of freedom
   */
  int s;
  double ax,ay,az=0.5*(dof + n);
  gsl_vector *ym, *xm;
  gsl_matrix *work = gsl_matrix_alloc(n,n), 
    *winv = gsl_matrix_alloc(n,n);
  gsl_permutation *p = gsl_permutation_alloc(n);
  
  gsl_matrix_memcpy( work, scale );
  gsl_linalg_LU_decomp( work, p, &s );
  gsl_linalg_LU_invert( work, p, winv );
  ax = gsl_linalg_LU_det( work, s );
  gsl_matrix_free( work );
  gsl_permutation_free( p );
  
  xm = gsl_vector_alloc(n);
  gsl_vector_memcpy( xm, x);
  gsl_vector_sub( xm, location );
  ym = gsl_vector_alloc(n);
  gsl_blas_dsymv(CblasUpper,1.0,winv,xm,0.0,ym);
  gsl_matrix_free( winv );
  gsl_blas_ddot( xm, ym, &ay);
  gsl_vector_free(xm);
  gsl_vector_free(ym);
  
  ay = -az*log(1+ay/dof) +
    gsl_sf_lngamma(az) - 
    gsl_sf_lngamma(0.5*dof) -
    0.5*( n*log(dof) + n*log(M_PI) + log(ax) ); 
  
  return ay;
}

double dmvt_log(const int n, 
		const std::vector<double>& xx, 
		const std::vector<double>& llocation, 
		const gsl_matrix *scale, 
		const int dof){
  /* multivariate Student t density function */
  /*
   *	n	 dimension of the random vetor
   *	location vector of locations of size n
   *	scale	 scale matrix of dimension n x n
   *	dof	 degrees of freedom
   */
  int s;
  double ax,ay,az=0.5*(dof + n);
  gsl_vector *ym, *xm;

  gsl_vector *x = gsl_vector_alloc(n);
  gsl_vector *location = gsl_vector_alloc(n);
  for (int i=0; i<n; ++i) {
    gsl_vector_set(x, i, xx[i]);
    gsl_vector_set(location, i, llocation[i]);
  }

  gsl_matrix *work = gsl_matrix_alloc(n,n), 
    *winv = gsl_matrix_alloc(n,n);
  gsl_permutation *p = gsl_permutation_alloc(n);
  
  gsl_matrix_memcpy( work, scale );
  gsl_linalg_LU_decomp( work, p, &s );
  gsl_linalg_LU_invert( work, p, winv );
  ax = gsl_linalg_LU_det( work, s );
  gsl_matrix_free( work );
  gsl_permutation_free( p );
  
  xm = gsl_vector_alloc(n);
  gsl_vector_memcpy( xm, x);
  gsl_vector_sub( xm, location );
  ym = gsl_vector_alloc(n);
  gsl_blas_dsymv(CblasUpper,1.0,winv,xm,0.0,ym);
  gsl_matrix_free( winv );
  gsl_blas_ddot( xm, ym, &ay);
  gsl_vector_free(xm);
  gsl_vector_free(ym);
  
  ay = -az*log(1+ay/dof) +
    gsl_sf_lngamma(az) - 
    gsl_sf_lngamma(0.5*dof) -
    0.5*( n*log(dof) + n*log(M_PI) + log(ax) ); 
  
  gsl_vector_free(x);
  gsl_vector_free(location);

  return ay;
}
