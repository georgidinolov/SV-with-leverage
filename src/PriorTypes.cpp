#include <algorithm>
#include <iostream>
#define MATHLIB_STANDALONE
#include <Rmath.h>
#include <vector>
#include "PriorTypes.hpp"

namespace{ 
  inline double square(double x) {
    return x * x;
  }

  inline double cube(double x) {
    return x * x * x;
  }
  
  inline double pi() {
    return std::atan(1)*4;
  }

  inline double dtruncnorm(double x, double a, double b, 
			   double mean, double std_dev,
			   bool LOG) {
    
    double zeta = (x-mean)/std_dev;
    double Z = pnorm(b,mean,std_dev,1,0) - pnorm(a,mean,std_dev,1,0);
    
    if (LOG) {
      return dnorm(zeta,0,1,1) - log(std_dev) - log(Z);
    } else {
      return dnorm(zeta,0,1,0)/(std_dev*Z);
    }
  }
}

// ================ THETA PRIOR ==========================
ThetaPrior::ThetaPrior()
  : theta_hat_mean_(5.6e-7),
    theta_hat_std_dev_(1e-6),
    theta_hat_lb_(-1.0*HUGE_VAL),
    theta_hat_ub_(HUGE_VAL)
{}

ThetaPrior::ThetaPrior(double delta_t)
  : theta_hat_mean_(5.6e-7),
    theta_hat_std_dev_(1e-6),
    delta_t_(delta_t),
    constant_delta_t_(true),
    theta_hat_lb_(-1.0*HUGE_VAL),
    theta_hat_ub_(HUGE_VAL)
{
  set_theta_mean_std_dev();
}

ThetaPrior::ThetaPrior(double theta_hat_mean,
		       double theta_hat_std_dev)
  : theta_hat_mean_(theta_hat_mean),
    theta_hat_std_dev_(theta_hat_std_dev),
    theta_hat_lb_(-1.0*HUGE_VAL),
    theta_hat_ub_(HUGE_VAL)
{}

ThetaPrior::ThetaPrior(double theta_hat_mean,
		       double theta_hat_std_dev,
		       double delta_t)
  : theta_hat_mean_(theta_hat_mean),
    theta_hat_std_dev_(theta_hat_std_dev),
    delta_t_(delta_t),
    constant_delta_t_(true),
    theta_hat_lb_(-1.0*HUGE_VAL),
    theta_hat_ub_(HUGE_VAL)
{
  set_theta_mean_std_dev();
}

double ThetaPrior::get_theta_hat_mean() const
{
  return theta_hat_mean_;
}
double ThetaPrior::get_theta_hat_std_dev() const
{
  return theta_hat_std_dev_;
}
double ThetaPrior::get_theta_mean() const
{
  return theta_mean_;
}
double ThetaPrior::get_theta_std_dev() const
{
  return theta_std_dev_;
}

void ThetaPrior::set_theta_hat_mean(double theta_hat_mean)
{
  theta_hat_mean_ = theta_hat_mean;
  set_theta_mean_std_dev();
}

void ThetaPrior::set_theta_hat_std_dev(double theta_hat_std_dev)
{
  theta_hat_std_dev_ = theta_hat_std_dev;
  set_theta_mean_std_dev();
}

double ThetaPrior::log_likelihood(double theta) const
{
  // double alpha = 0;
  // double beta = 0;
  // if ( theta_std_dev_ <= 0.25 ) {
  //   alpha = 
  //     ((1.0-theta_mean_)/square(theta_std_dev_) -
  //      1.0/theta_mean_)*square(theta_mean_);
    
  //   beta = alpha*(1/theta_mean_ - 1);
  // } else {
  //   double theta_std_dev = 0.249;
  //   alpha = 
  //     ((1.0-theta_mean_)/square(theta_std_dev) -
  //      1.0/theta_mean_)*square(theta_mean_);
    
  //   beta = alpha*(1/theta_mean_ - 1);
  // }

  //  return dbeta(theta, alpha, beta, 1);

  double theta_hat = -1.0*log(theta)/delta_t_;
  double out = 0;
  if ( theta_hat <= theta_hat_lb_) {
    std::cout << "theta_hat breaks bounds: lb = " 
	      << theta_hat_lb_ << "; "
	      << "theat_hat = " << theta_hat
	      << "\n";
    out = -1.0*HUGE_VAL;
  } else if (theta_hat >= theta_hat_ub_) {
    std::cout << "theta_hat breaks bounds: up = " 
	      << theta_hat_ub_ << "; "
	      << "theat_hat = " << theta_hat
	      << "\n";
    out = -HUGE_VAL;
  } else {
    out = dtruncnorm(theta, 0.0, 1.0, 
  		    theta_mean_, theta_std_dev_,
  		    true);
  }
  return out;
}

void ThetaPrior::set_theta_mean_std_dev()
{
  if (constant_delta_t_) {
    nlopt::opt opt(nlopt::LN_NELDERMEAD, 2);
    std::vector<double> theta_prior_params_start = 
      {theta_hat_mean_ * delta_t_, log(theta_hat_std_dev_ * delta_t_)};
					
    std::vector<double> lb = {-1,-HUGE_VAL};
    std::vector<double> ub = {1,HUGE_VAL};
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);

    double minf;  
    opt.set_min_objective(ThetaPrior::wrapper_theta, this);
    opt.optimize(theta_prior_params_start, minf);

    theta_mean_ = theta_prior_params_start[0];
    theta_std_dev_ = exp(theta_prior_params_start[1]);
  } else {
    std::cout << "WARNING: No constant delta_t_ set" << std::endl;
    theta_mean_ = 0;
    theta_std_dev_ = 0;
    //TODO(georgid): NEED TO THROW AN EXCEPTION HERE   
  }
}

double ThetaPrior::
theta_mean_std_dev_minimization(const std::vector<double> &x,
				std::vector<double> &grad)
{
  if (!grad.empty()) {
    // grad is always empty!
    // should throw an exception if not empty
  }

  double a_delta = x[0];
  double b_delta = exp(x[1]);
  
  double alpha = -a_delta/b_delta;
  double beta = (1-a_delta)/b_delta;

  double dnorm_alpha = dnorm(alpha,0,1,0);
  double pnorm_alpha = pnorm(alpha,0,1,1,0);

  double dnorm_beta = dnorm(beta,0,1,0);
  double pnorm_beta = pnorm(beta,0,1,1,0);

  double Z = pnorm_beta - pnorm_alpha;

  double expectation_lhs = a_delta + 
    (dnorm_alpha - dnorm_beta)/Z * b_delta;

  double variance_lhs = square(b_delta) * 
    (1 + 
     (alpha*dnorm_alpha - beta*dnorm_beta)/Z
     +
     square((dnorm_alpha - dnorm_beta)/Z));

  double expectation_rhs = exp(-theta_hat_mean_ * delta_t_) * 
    (1 + 0.5*square(theta_hat_std_dev_)*square(delta_t_));

  double second_moment_rhs = exp(-2*theta_hat_mean_ * delta_t_) * 
    (1 + 2*square(theta_hat_std_dev_)*square(delta_t_));

  double variance_rhs = second_moment_rhs - square(expectation_rhs);
  
  return sqrt(square(expectation_rhs - expectation_lhs) + 
	      square(variance_rhs - variance_lhs));
}

double ThetaPrior::
wrapper_theta(const std::vector<double> &x,
	      std::vector<double> &grad,
	      void * data)
{
  ThetaPrior * params = 
    reinterpret_cast<ThetaPrior*>(data);
  return  params->theta_mean_std_dev_minimization(x,grad);  
}

// ================ TAU SQ PRIOR =========================
TauSquarePrior::TauSquarePrior(const ThetaPrior& theta_prior)
  : tau_square_hat_mean_(1.3e-7),
    tau_square_hat_std_dev_(1e-6),
    theta_prior_(theta_prior)
{}

TauSquarePrior::TauSquarePrior(const ThetaPrior& theta_prior,
			       double delta_t)
  : tau_square_hat_mean_(1.3e-7),
    tau_square_hat_std_dev_(1e-6),
    delta_t_(delta_t),
    constant_delta_t_(true),
    theta_prior_(theta_prior)
{
  set_tau_square_shape_scale();
}

TauSquarePrior::TauSquarePrior(const ThetaPrior& theta_prior,
			       double tau_square_hat_mean,
			       double tau_square_hat_std_dev)
  : tau_square_hat_mean_(tau_square_hat_mean),
    tau_square_hat_std_dev_(tau_square_hat_std_dev),
    delta_t_(0.0),
    constant_delta_t_(false),
    theta_prior_(theta_prior)
{
  set_tau_square_shape_scale();
}

TauSquarePrior::TauSquarePrior(const ThetaPrior& theta_prior,
			       double tau_square_hat_mean,
			       double tau_square_hat_std_dev,
			       double delta_t)
  : tau_square_hat_mean_(tau_square_hat_mean),
    tau_square_hat_std_dev_(tau_square_hat_std_dev),
    delta_t_(delta_t),
    constant_delta_t_(true),
    theta_prior_(theta_prior)
{
  set_tau_square_shape_scale();
}

double TauSquarePrior::get_tau_square_hat_mean() const
{
  return tau_square_hat_mean_;
}
double TauSquarePrior::get_tau_square_hat_std_dev() const
{
  return tau_square_hat_std_dev_;
}
double TauSquarePrior::get_tau_square_hat_shape() const
{
  return tau_square_hat_shape_;
}
double TauSquarePrior::get_tau_square_hat_scale() const
{
  return tau_square_hat_scale_;
}
double TauSquarePrior::get_tau_square_shape() const
{
  return tau_square_shape_;
}
double TauSquarePrior::get_tau_square_scale() const
{
  return tau_square_scale_;
}
double TauSquarePrior::get_tau_square_mean() const
{
  return tau_square_mean_;
}
double TauSquarePrior::get_tau_square_std_dev() const
{
  return tau_square_std_dev_;
}

void TauSquarePrior::set_tau_square_hat_mean(double tau_square_hat_mean)
{
  tau_square_hat_mean_ = tau_square_hat_mean;
  set_tau_square_shape_scale();
}
void TauSquarePrior::set_tau_square_hat_std_dev(double tau_square_hat_std_dev)
{
  tau_square_hat_std_dev_ = tau_square_hat_std_dev;
  set_tau_square_shape_scale();
}

double TauSquarePrior::log_likelihood(double tau_square) const
{
  return dgamma(1.0/tau_square, tau_square_shape_, 1.0/tau_square_scale_, 1);
}

void TauSquarePrior::set_tau_square_shape_scale()
{
  if (constant_delta_t_) {
    nlopt::opt opt(nlopt::LN_NELDERMEAD, 2);

    double alpha_hat = square(tau_square_hat_mean_) / 
      square(tau_square_hat_std_dev_) + 2;
    double beta_hat = tau_square_hat_mean_ * (alpha_hat-1);

    tau_square_hat_shape_ = alpha_hat;
    tau_square_hat_scale_ = beta_hat;

    double first_moment_approx = tau_square_hat_mean_ * 
      (1.0-exp(-2.0*theta_prior_.get_theta_hat_mean()*delta_t_)) / 
      (2.0*theta_prior_.get_theta_hat_mean());

    double mean_approx = first_moment_approx;
    double variance_approx = square(first_moment_approx * 10);

    double alpha_start = square(mean_approx) / variance_approx + 2;
    alpha_start = std::max(2+1e-6, alpha_start);
    double beta_start = mean_approx * (alpha_start-1);

    std::vector<double> tau_sq_params = {alpha_start,
    					 beta_start};
					
    std::vector<double> lb = {2,0};
    std::vector<double> ub = {HUGE_VAL,HUGE_VAL};
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
    double minf;  
    opt.set_min_objective(TauSquarePrior::wrapper_tau_sq, this);
    opt.optimize(tau_sq_params, minf);
    
    tau_square_shape_ = tau_sq_params[0];
    tau_square_scale_ = tau_sq_params[1];

    tau_square_mean_ = tau_square_scale_ / (tau_square_shape_-1);
    tau_square_std_dev_ = square(tau_square_mean_) / (tau_square_shape_-2);
  } else {
    std::cout << "WARNING: No constant delta_t_ set" << std::endl;
    tau_square_shape_ = 0.0;
    tau_square_scale_ = 0.0;
    //TODO(georgid): NEED TO THROW AN EXCEPTION HERE   
  }
}

double TauSquarePrior::
tau_sq_parameter_minimization(const std::vector<double> &x,
			      std::vector<double> &grad)
{
  if (!grad.empty()) {
    // grad is always empty!
    // should throw an exception if not empty
  }

  double a_delta = x[0];
  double b_delta = x[1];

  std::vector<double> alpha_beta = tau_sq_parameters();
  double mean_rhs = alpha_beta[1] / (alpha_beta[0]-1);
  double variance_rhs = square(mean_rhs) / (alpha_beta[0]-2);
  
  double mean_lhs = b_delta / (a_delta-1);
  double variance_lhs = square(b_delta) / (square(a_delta-1)*(a_delta-2));

  return sqrt(square(mean_rhs - mean_lhs) + 
	      square(variance_rhs - variance_lhs));
}

std::vector<double> TauSquarePrior::tau_sq_parameters() const
{
  double theta_hat_mean = 
    theta_prior_.get_theta_hat_mean();
  double theta_hat_std_dev = 
    theta_prior_.get_theta_hat_std_dev();

  double f = 
    0.5  
    * tau_square_hat_mean_
    * (1.0-exp(-2.0*theta_hat_mean*delta_t_))/theta_hat_mean;

  double f_prime_theta = 0.5 
    * tau_square_hat_mean_ 
    * (2.0*delta_t_*exp(-2.0*theta_hat_mean*delta_t_)/theta_hat_mean - 
       (1.0-exp(-2.0*theta_hat_mean*delta_t_))/square(theta_hat_mean));

  double f_prime_tau_sq = 
    0.5 
    * (1.0-exp(-2.0*theta_hat_mean*delta_t_))/theta_hat_mean;

  double f_double_prime_theta = 
    0.5 
    * tau_square_hat_mean_ 
    * (-4.0*square(delta_t_)*exp(-2.0*theta_hat_mean*delta_t_)/theta_hat_mean
       - 2.0*2.0*delta_t_*exp(-2.0*theta_hat_mean*delta_t_)/square(theta_hat_mean)
       + 2.0*(1-exp(-2.0*theta_hat_mean*delta_t_))/cube(theta_hat_mean));

  double f_double_prime_tau_sq = 0.0;

  double first_moment_first_term = f;

  double first_moment_second_term = 
    0.5 * f_double_prime_tau_sq * square(tau_square_hat_std_dev_);

  double first_moment_third_term = 
    0.5 * f_double_prime_theta * square(theta_hat_std_dev);
  
  double second_moment_first_term = 
    square(f);

  double second_moment_second_term = 
    0.5 
    * (2.0*square(f_prime_theta) + 2.0*f*f_double_prime_theta) 
    * square(theta_hat_std_dev);

  double second_moment_third_term = 
    0.5 
    * (2.0*square(f_prime_tau_sq) + 2.0*f*f_double_prime_tau_sq) 
    * square(tau_square_hat_std_dev_);

  double first_moment_rhs = 
    first_moment_first_term + 
    first_moment_second_term +
    first_moment_third_term;

  double second_moment_rhs = 
    second_moment_first_term + 
    second_moment_second_term + 
    second_moment_third_term;

  double mean_rhs = first_moment_rhs;
  double variance_rhs = second_moment_rhs - square(first_moment_rhs);

  double alpha = square(mean_rhs) / variance_rhs + 2;
  double beta = mean_rhs * (alpha-1.0);
  return std::vector<double> {alpha, beta};
}

double TauSquarePrior::
wrapper_tau_sq(const std::vector<double> &x,
	       std::vector<double> &grad,
	       void * data)
{
  TauSquarePrior * params = 
    reinterpret_cast<TauSquarePrior*>(data);
  return  params->tau_sq_parameter_minimization(x,grad);  
}

// ================== MU PRIOR =============================
MuPrior::MuPrior()
  : mu_hat_mean_(1.7e-12),
    mu_hat_std_dev_(1e-11)
{}

MuPrior::MuPrior(double delta_t)
  : mu_hat_mean_(1.7e-12),
    mu_hat_std_dev_(1e-11),
    delta_t_(delta_t),
    constant_delta_t_(true)
{
  set_mu_mean_std_dev();
}

MuPrior::MuPrior(double mu_hat_mean,
		 double mu_hat_std_dev,
		 double delta_t)
  : mu_hat_mean_(mu_hat_mean),
    mu_hat_std_dev_(mu_hat_std_dev),
    delta_t_(delta_t),
    constant_delta_t_(true)
{
  set_mu_mean_std_dev();
}

double MuPrior::get_mu_hat_mean() const
{
  return mu_hat_mean_;
}
double MuPrior::get_mu_hat_std_dev() const
{
  return mu_hat_std_dev_;
}
double MuPrior::get_mu_mean() const
{
  return mu_mean_;
}
double MuPrior::get_mu_std_dev() const
{
  return mu_std_dev_;
}

void MuPrior::set_mu_hat_mean(double mu_hat_mean)
{
  mu_hat_mean_ = mu_hat_mean;
  set_mu_mean_std_dev();
}
void MuPrior::set_mu_hat_std_dev(double mu_hat_std_dev)
{
  mu_hat_std_dev_ = mu_hat_std_dev;
  set_mu_mean_std_dev();
}

void MuPrior::set_mu_mean_std_dev()
{
  mu_mean_ = mu_hat_mean_ * delta_t_;
  mu_std_dev_ = mu_hat_std_dev_ * delta_t_;
}

double MuPrior::log_likelihood(double mu) const
{
  return dnorm(mu, mu_hat_mean_*delta_t_, mu_std_dev_*delta_t_, 1);
}

// ================== ALPHA PRIOR ==========================
AlphaPrior::AlphaPrior()
  : alpha_hat_mean_(-13), 
    alpha_hat_std_dev_(10)
{}

AlphaPrior::AlphaPrior(double delta_t)
  : alpha_hat_mean_(-13), 
    alpha_hat_std_dev_(10),
    delta_t_(delta_t),
    constant_delta_t_(true)
{
  set_alpha_mean_std_dev();
}

AlphaPrior::AlphaPrior(double alpha_hat_mean,
		       double alpha_hat_std_dev,
		       double delta_t)
  : alpha_hat_mean_(alpha_hat_mean),
    alpha_hat_std_dev(alpha_hat_std_dev),
    delta_t_(delta_t),
    constant_delta_t_(true)
{
  set_alpha_mean_std_dev();
}

double AlphaPrior::get_alpha_hat_mean() const 
{
  return alpha_hat_mean_;
}

double AlphaPrior::get_alpha_mean() const 
{
  return alpha_hat_mean_ + 0.5*log(delta_t_);
}

double AlphaPrior::get_alpha_hat_std_dev() const
{
  return alpha_hat_std_dev_;
}

double AlphaPrior::get_alpha_std_dev() const
{
  return alpha_hat_std_dev_;
}

void AlphaPrior::set_alpha_hat_mean(double alpha_hat_mean)
{
  alpha_hat_mean_ = alpha_hat_mean;
  set_alpha_mean_std_dev();
}

void AlphaPrior::set_alpha_hat_std_dev(double alpha_hat_std_dev)
{
  alpha_hat_std_dev_ = alpha_hat_std_dev;
  set_alpha_mean_std_dev();
}

double AlphaPrior::log_likelihood(double alpha) const
{
  return dnorm(alpha,alpha_mean_,alpha_std_dev_,1);
}

void AlphaPrior::set_alpha_mean_std_dev()
{
  alpha_mean_ = alpha_hat_mean_ + 0.5*log(delta_t_);
  alpha_std_dev_ = alpha_hat_std_dev_;
}

// ================== RHO PRIOR ===========================
RhoPrior::RhoPrior()
  : rho_mean_(0.0),
    rho_std_dev_(0.9)
{}

RhoPrior::RhoPrior(double rho_mean,
		   double rho_std_dev)
  : rho_mean_(rho_mean),
    rho_std_dev_(rho_std_dev)
{}

double RhoPrior::get_rho_mean() const
{
  return rho_mean_;
}

double RhoPrior::get_rho_std_dev() const
{
  return rho_std_dev_;
}

void RhoPrior::set_rho_mean(double rho_mean)
{
  rho_mean_ = rho_mean;
}

void RhoPrior::set_rho_std_dev(double rho_std_dev)
{
  rho_std_dev_ = rho_std_dev;
}

double RhoPrior::log_likelihood(double rho) const
{
  double out = dtruncnorm(rho, -1.0, 1.0, 
		    rho_mean_, rho_std_dev_,
		    true);
  return out;

  // double transformed_mean = (rho_mean_ + 1.0)/2.0;
  // double transformed_std_dev = rho_std_dev_ / 2.0;
  // double alpha = ((1-transformed_mean)/square(transformed_std_dev) - 1.0/transformed_mean) * 
  //   square(transformed_mean);
  // double beta = alpha * (1.0/transformed_mean - 1.0);

  // std::cout << "alpha=" << alpha << "\n";
  // std::cout << "beta=" << alpha << "\n";

  // double rho_tilde = (rho + 1.0)/2.0;
  // return dbeta(rho_tilde, alpha, beta, 1) + log(0.5);
}

// ==================== XI SQUARE PRIOR ==========================
XiSquarePrior::XiSquarePrior()
  : xi_square_mean_(6.25e-8),
    xi_square_std_dev_(1e-7)
{}

double XiSquarePrior::get_xi_square_mean() const
{
  return xi_square_mean_;
}

double XiSquarePrior::get_xi_square_std_dev() const
{
  return xi_square_std_dev_;
}

double XiSquarePrior::get_xi_square_shape() const
{
  double shape = 
    square(xi_square_mean_) / square(xi_square_std_dev_) + 2.0;
  return shape;
}

double XiSquarePrior::get_xi_square_scale() const
{
  return xi_square_mean_ * (get_xi_square_shape() - 1);
}

void XiSquarePrior::set_xi_square_mean(double xi_square_mean)
{
  xi_square_mean_ = xi_square_mean;
}

void XiSquarePrior::set_xi_square_std_dev(double xi_square_std_dev)
{
  xi_square_std_dev_ = xi_square_std_dev;
}

double XiSquarePrior::log_likelihood(double xi_square) const
{
  double alpha = get_xi_square_shape();
  double beta = get_xi_square_scale();
  double out = dgamma(1.0/xi_square, alpha, 1.0/beta, 1);
  return out;
}

// ===================== OBSERVATIONAL PRIORS ===================
ObservationalPriors::ObservationalPriors()
  : xi_square_prior_(XiSquarePrior()),
    nu_prior_(NuPrior())
{}

ObservationalPriors::ObservationalPriors(double delta_t)
  : xi_square_prior_(XiSquarePrior()),
    nu_prior_(NuPrior())
{}

const XiSquarePrior& ObservationalPriors::get_xi_square_prior() const
{
  return xi_square_prior_;
}

const NuPrior& ObservationalPriors::get_nu_prior() const
{
  return nu_prior_;
}

NuPrior::NuPrior()
  : nu_max_(20.0),
    nu_min_(1.0)
{}

// ===================== LAMBDA PRIOR ===========================
LambdaPrior::LambdaPrior()
  : lambda_mean_(1.0/(1*6.5*60*60*1000)),
    lambda_std_dev_(lambda_mean_/4.0)
{}

LambdaPrior::LambdaPrior(double lambda_mean,
			 double lambda_std_dev)
  : lambda_mean_(lambda_mean),
    lambda_std_dev_(lambda_std_dev)
{}

double LambdaPrior::get_lambda_mean() const
{
  return lambda_mean_;
}

double LambdaPrior::get_lambda_std_dev() const
{
  return lambda_std_dev_;
}

double LambdaPrior::get_lambda_alpha() const
{
  double beta = lambda_mean_ / square(lambda_std_dev_);
  double alpha = lambda_mean_ * beta;
  return alpha;
}

double LambdaPrior::get_lambda_beta() const
{
  double beta = lambda_mean_ / square(lambda_std_dev_);
  return beta;
}

double LambdaPrior::log_likelihood(double lambda) const
{
  double beta = lambda_mean_ / square(lambda_std_dev_);
  double alpha = lambda_mean_ * beta;
  return dgamma(lambda, alpha, 1.0/beta, 1);
}

void LambdaPrior::set_lambda_mean(double lambda_mean)
{
  lambda_mean_ = lambda_mean;
}

void LambdaPrior::set_lambda_std_dev(double lambda_sd)
{
  lambda_std_dev_ = lambda_sd;
}

// ===================== SIGMA SQUARE PRIOR =====================
SigmaSquarePrior::SigmaSquarePrior()
  : sigma_square_mean_(1e-2),
    sigma_square_std_dev_(1e-1)
{}

SigmaSquarePrior::SigmaSquarePrior(double mean,
				   double std_dev)
  : sigma_square_mean_(mean),
    sigma_square_std_dev_(std_dev)
{}

double SigmaSquarePrior::get_sigma_square_alpha() const
{
  double beta = 
    square(sigma_square_mean_) / square(sigma_square_std_dev_) + 2.0;
  double alpha = beta / sigma_square_mean_ + 1;
  return alpha;
}

double SigmaSquarePrior::get_sigma_square_beta() const
{
  double beta = 
    square(sigma_square_mean_) / square(sigma_square_std_dev_) + 2.0;
  return beta;
}

double SigmaSquarePrior::log_likelihood(double sigma_sq) const
{
  double beta = 
    square(sigma_square_mean_) / square(sigma_square_std_dev_) + 2.0;

  double alpha = beta / sigma_square_mean_ + 1;
 
  return dgamma(1.0/sigma_sq, alpha, 1.0/beta, 1);
}

// ======================= DELTA PRIOR ==========================
DeltaPrior::DeltaPrior()
  : nu_(10.0)
{}

DeltaPrior::DeltaPrior(double nu)
  : nu_(nu)
{}

double DeltaPrior::log_likelihood(double delta) const
{
  return dgamma(delta, nu_/2.0, 2.0/nu_, 1);
}

// =================== CONSTANT VOLATILITY PRIORS ===============

ConstantVolatilityPriors::ConstantVolatilityPriors()
  : mu_prior_(MuPrior())
{}

ConstantVolatilityPriors::ConstantVolatilityPriors(double delta_t)
  : mu_prior_(MuPrior(delta_t))
{}

const MuPrior & ConstantVolatilityPriors::get_mu_prior() const
{
  return mu_prior_;
}

// ================ STOCHASTIC VOLATILITY PRIORS ===========================

StochasticVolatilityPriors::StochasticVolatilityPriors()
  : theta_prior_(ThetaPrior()),
    tau_square_prior_(TauSquarePrior(theta_prior_)),
    mu_prior_(MuPrior()),
    alpha_prior_(AlphaPrior()),
    rho_prior_(RhoPrior()),
    xi_square_prior_(XiSquarePrior())
{}

StochasticVolatilityPriors::StochasticVolatilityPriors(double delta_t)
  : theta_prior_(ThetaPrior(delta_t)),
    tau_square_prior_(TauSquarePrior(theta_prior_,
				     delta_t)),
    mu_prior_(MuPrior(delta_t)),
    alpha_prior_(AlphaPrior(delta_t)),
    rho_prior_(RhoPrior()),
    xi_square_prior_(XiSquarePrior())
{}

StochasticVolatilityPriors::StochasticVolatilityPriors(double mu_hat_mean,
						       double mu_hat_std_dev,
						       double theta_hat_mean,
						       double theta_hat_std_dev,
						       double alpha_hat_mean, 
						       double alpha_hat_std_dev,
						       double tau_square_hat_mean,
						       double tau_square_hat_std_dev,
						       double xi_square_mean,
						       double xi_square_std_dev,
						       double rho_mean,
						       double rho_std_dev,
						       double delta_t)
  : theta_prior_(ThetaPrior(delta_t,
			    theta_hat_mean,
			    theta_hat_std_dev)),
    tau_square_prior_(TauSquarePrior(theta_prior_,
				     tau_square_hat_mean,
				     tau_square_hat_std_dev,
				     delta_t)),
    mu_prior_(MuPrior(mu_hat_mean,
		      mu_hat_std_dev,
		      delta_t)),
    alpha_prior_(AlphaPrior(alpha_hat_mean,
			    alpha_hat_std_dev,
			    delta_t)),
    rho_prior_(RhoPrior(rho_mean,
			rho_std_dev)),
    xi_square_prior_(XiSquarePrior())
{}

StochasticVolatilityPriors * StochasticVolatilityPriors::clone() const
{
  return new StochasticVolatilityPriors(*this);
}

const ThetaPrior & StochasticVolatilityPriors::get_theta_prior() const 
{
  return theta_prior_;
}

const TauSquarePrior & StochasticVolatilityPriors::get_tau_square_prior() const 
{
  return tau_square_prior_;
}

const MuPrior & StochasticVolatilityPriors::get_mu_prior() const
{
  return mu_prior_;
}

const AlphaPrior & StochasticVolatilityPriors::get_alpha_prior() const
{
  return alpha_prior_;
}

const RhoPrior & StochasticVolatilityPriors::get_rho_prior() const
{
  return rho_prior_;
}

const XiSquarePrior & StochasticVolatilityPriors::get_xi_square_prior() const
{
  return xi_square_prior_;
}

// ================ MULTIFACTOR STOCHASTIC VOLATILITY PRIORS ============

MultifactorStochasticVolatilityPriors::MultifactorStochasticVolatilityPriors()
  : mu_prior_(MuPrior()),
    xi_square_prior_(XiSquarePrior())
{}

MultifactorStochasticVolatilityPriors::MultifactorStochasticVolatilityPriors(double delta_t)
  : mu_prior_(MuPrior(delta_t)),
    xi_square_prior_(XiSquarePrior())
{}

const MuPrior & MultifactorStochasticVolatilityPriors::get_mu_prior() const
{
  return mu_prior_;
}

const XiSquarePrior & MultifactorStochasticVolatilityPriors::get_xi_square_prior() const
{
  return xi_square_prior_;
}
