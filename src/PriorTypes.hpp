#include <iostream>
#include <vector>
#include "nlopt.hpp"

class ThetaPrior
{
public:
  ThetaPrior();
  ThetaPrior(double delta_t_);
  ThetaPrior(double theta_hat_mean,
	     double theta_hat_std_dev);

  ThetaPrior(double theta_hat_mean,
	     double theta_hat_std_dev,
	     double delta_t_);

  double get_theta_hat_mean() const;
  double get_theta_hat_std_dev() const;
  //
  double get_theta_mean() const;
  double get_theta_std_dev() const;
  //
  double get_theta_shape() const;
  double get_theta_scale() const;

  void set_theta_hat_mean(double theta_hat_mean);
  void set_theta_hat_std_dev(double theta_hat_std_dev);

  inline void set_theta_hat_lb(double theta_hat_lb) {
    theta_hat_lb_ = theta_hat_lb;
  };

  inline void set_theta_hat_ub(double theta_hat_ub) {
    theta_hat_ub_ = theta_hat_ub;
  };
  
  // theta is on the discrete scale!
  double log_likelihood(double theta) const;

private:
  void set_theta_mean_std_dev();
  double theta_mean_std_dev_minimization(const std::vector<double> &x,
					 std::vector<double> &grad);
  static double wrapper_theta(const std::vector<double> &x, 
			      std::vector<double> &grad,
			      void * data);
  double theta_hat_mean_;
  double theta_hat_std_dev_;
  double theta_mean_;
  double theta_std_dev_;

  double delta_t_;
  double constant_delta_t_;
  
  double theta_hat_lb_;
  double theta_hat_ub_;
};

class TauSquarePrior
{
public:
  TauSquarePrior(const ThetaPrior& theta_prior);

  TauSquarePrior(const ThetaPrior& theta_prior,
		 double delta_t_);

  TauSquarePrior(const ThetaPrior& theta_prior,
  		 double tau_square_hat_mean,
  		 double tau_square_hat_std_dev);

  TauSquarePrior(const ThetaPrior& theta_prior,
  		 double tau_square_hat_mean,
  		 double tau_square_hat_std_dev,
  		 double delta_t_);
		 
  double get_tau_square_hat_mean() const;
  double get_tau_square_hat_std_dev() const;
  double get_tau_square_hat_shape() const;
  double get_tau_square_hat_scale() const;
  double get_tau_square_shape() const;
  double get_tau_square_scale() const;
  double get_tau_square_mean() const;
  double get_tau_square_std_dev() const;

  void set_tau_square_hat_mean(double tau_square_hat_mean);
  void set_tau_square_hat_std_dev(double tau_square_hat_std_dev);

  double log_likelihood(double tau_square) const;
private:
  // sets shape and scale parameters in continuous and disrete time.
  void set_tau_square_shape_scale();
  double tau_sq_parameter_minimization(const std::vector<double> &x,
				       std::vector<double> &grad);
  std::vector<double> tau_sq_parameters() const;
  static double wrapper_tau_sq(const std::vector<double> &x, 
  			       std::vector<double> &grad,
  			       void * data);
  double tau_square_hat_mean_;
  double tau_square_hat_std_dev_;
  double tau_square_mean_;
  double tau_square_std_dev_;
  double tau_square_hat_shape_;
  double tau_square_hat_scale_;
  double tau_square_shape_;
  double tau_square_scale_;

  double delta_t_; 
  bool constant_delta_t_;
  const ThetaPrior theta_prior_;
};

class MuPrior
{
public:
  MuPrior();
  MuPrior(double delta_t_);
  MuPrior(double mu_hat_mean,
	  double mu_hat_std_dev);

  MuPrior(double mu_hat_mean,
	  double mu_hat_std_dev,
	  double delta_t_);
		 
  double get_mu_hat_mean() const;
  double get_mu_hat_std_dev() const;
  double get_mu_mean() const;
  double get_mu_std_dev() const;

  void set_mu_hat_mean(double mu_hat_mean);
  void set_mu_hat_std_dev(double mu_hat_std_dev);
  
  double log_likelihood(double mu) const;
private:
  // sets mean and std dev parameters in discrete time.
  void set_mu_mean_std_dev();

  double mu_hat_mean_;
  double mu_hat_std_dev_;
  double mu_mean_;
  double mu_std_dev_;
  
  double delta_t_; 
  bool constant_delta_t_;
};

class AlphaPrior
{
public:
  AlphaPrior();
  AlphaPrior(double delta_t_);
  AlphaPrior(double alpha_hat_mean,
	     double alpha_hat_std_dev,
	     double delta_t);
    
  double get_alpha_hat_mean() const;
  double get_alpha_hat_std_dev() const;

  double get_alpha_mean() const;
  double get_alpha_std_dev() const;

  void set_alpha_hat_mean(double alpha_hat_mean);
  void set_alpha_hat_std_dev(double alpha_hat_std_dev);
  
  // this is on the discrete scale
  double log_likelihood(double alpha) const;
private:
  // sets mean and std dev parameters in discrete time.
  void set_alpha_mean_std_dev();

  double alpha_hat_mean_;
  double alpha_hat_std_dev_;
  double alpha_mean_;
  double alpha_std_dev_;
  
  double delta_t_; 
  bool constant_delta_t_;
};

class RhoPrior
{
public:
  RhoPrior();
  RhoPrior(double rho_mean,
	   double rho_std_dev);

  double get_rho_mean() const;
  double get_rho_std_dev() const;

  void set_rho_mean(double rho_mean);
  void set_rho_std_dev(double rho_std_dev);
  
  // continuous scale
  double log_likelihood(double rho) const;
private:
  double rho_mean_;
  double rho_std_dev_;
};

class XiSquarePrior
{
public:
  XiSquarePrior();

  double get_xi_square_mean() const;
  double get_xi_square_std_dev() const;
  
  double get_xi_square_shape() const;
  double get_xi_square_scale() const;

  void set_xi_square_mean(double xi_square_mean);
  void set_xi_square_std_dev(double xi_square_std_dev);
  
  double log_likelihood(double xi_square) const;
private:
  double xi_square_mean_;
  double xi_square_std_dev_;
};

class LambdaPrior
{
public:
  LambdaPrior();
  LambdaPrior(double lambda_mean,
	      double lambda_std_dev);

  double get_lambda_mean() const;
  double get_lambda_std_dev() const;
  double get_lambda_alpha() const;
  double get_lambda_beta() const;
  double log_likelihood(double lambda) const;

  void set_lambda_mean(double lambda_mean);
  void set_lambda_std_dev(double lambda_sd);

private:
  double lambda_mean_;
  double lambda_std_dev_;
};

class SigmaSquarePrior
{
public:
  SigmaSquarePrior();
  SigmaSquarePrior(double mean,
		   double std_dev);

  inline double get_sigma_square_mean() const {
    return sigma_square_mean_;
  }
  inline double get_sigma_square_std_dev() const {
    return sigma_square_std_dev_;
  }
  double get_sigma_square_alpha() const;
  double get_sigma_square_beta() const;

  double log_likelihood(double sigma_sq) const;

  inline void set_sigma_square_mean(double mean) {
    sigma_square_mean_ = mean;
  }
  inline void set_sigma_square_std_dev(double std_dev) {
    sigma_square_std_dev_ = std_dev;
  }

private:
  double sigma_square_mean_;
  double sigma_square_std_dev_;
};


// ======================== DELTA PRIOR ========================

class DeltaPrior
{
public:
  DeltaPrior();
  DeltaPrior(double nu);
	     
  inline double get_nu() const {
    return nu_;
  }
  inline void set_nu(double nu) {
    nu_ = nu;
  }

  double log_likelihood(double delta) const;

private:
  double nu_;
};

// =================== CONSTANT VOLATILITY PRIORS ===============
class ConstantVolatilityPriors
{
public:
  ConstantVolatilityPriors();
  ConstantVolatilityPriors(double delta_t);

  // ConstantVolatilityPriors(double mu_hat_mean,
  // 			     double mu_hat_std_dev,
  // 			     double theta_hat_mean,
  // 			     double theta_hat_std_dev,
  // 			     double alpha_hat_mean, 
  // 			     double alpha_hat_std_dev,
  // 			     double tau_square_hat_mean,
  // 			     double tau_square_hat_std_dev,
  // 			     double xi_square_mean,
  // 			     double xi_square_std_dev,
  // 			     double rho_mean,
  // 			     double rho_std_dev);

  // ConstantVolatilityPriors(double mu_hat_mean,
  // 			     double mu_hat_std_dev,
  // 			     double theta_hat_mean,
  // 			     double theta_hat_std_dev,
  // 			     double alpha_hat_mean, 
  // 			     double alpha_hat_std_dev,
  // 			     double tau_square_hat_mean,
  // 			     double tau_square_hat_std_dev,
  // 			     double xi_square_mean,
  // 			     double xi_square_std_dev,
  // 			     double rho_mean,
  // 			     double rho_std_dev,
  // 			     double delta_t);

  const MuPrior& get_mu_prior() const;
private:
  MuPrior mu_prior_;
};

// ============ OBSERVATIONAL PRIORS =============
class NuPrior
{
public:
  NuPrior();

  inline double get_nu_max() const
  {
    return nu_max_;
  }
  
  inline double get_nu_min() const
  {
    return nu_min_;
  }

  inline double log_likelihood(double nu) const
  {
    return -log(nu_max_ - nu_min_ + 1);
  }

private:
  double nu_max_;
  double nu_min_;
};

class ObservationalPriors
{
public:
  ObservationalPriors();
  ObservationalPriors(double delta_t);

  const XiSquarePrior& get_xi_square_prior() const;
  const NuPrior& get_nu_prior() const;

private:
  XiSquarePrior xi_square_prior_;
  NuPrior nu_prior_;
};

// ==================== SV PRIORS =====================
class StochasticVolatilityPriors
{
public:
  StochasticVolatilityPriors();
  StochasticVolatilityPriors(double delta_t);

  // StochasticVolatilityPriors(double mu_hat_mean,
  // 			     double mu_hat_std_dev,
  // 			     double theta_hat_mean,
  // 			     double theta_hat_std_dev,
  // 			     double alpha_hat_mean, 
  // 			     double alpha_hat_std_dev,
  // 			     double tau_square_hat_mean,
  // 			     double tau_square_hat_std_dev,
  // 			     double xi_square_mean,
  // 			     double xi_square_std_dev,
  // 			     double rho_mean,
  // 			     double rho_std_dev);

  StochasticVolatilityPriors(double mu_hat_mean,
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
			      double delta_t);
  StochasticVolatilityPriors * clone()const;
  
  const ThetaPrior & get_theta_prior() const;
  const TauSquarePrior& get_tau_square_prior() const;
  const MuPrior& get_mu_prior() const;
  const AlphaPrior& get_alpha_prior() const;
  const RhoPrior& get_rho_prior() const;
  const XiSquarePrior& get_xi_square_prior() const;
private:
  ThetaPrior theta_prior_;
  TauSquarePrior tau_square_prior_;
  MuPrior mu_prior_;
  AlphaPrior alpha_prior_;
  RhoPrior rho_prior_;
  XiSquarePrior xi_square_prior_;
};

// ================== MULTIFACTOR SV PRIORS ====================
class MultifactorStochasticVolatilityPriors
{
public:
  MultifactorStochasticVolatilityPriors();
  MultifactorStochasticVolatilityPriors(double delta_t);

  // MultifactorStochasticVolatilityPriors(double mu_hat_mean,
  // 			     double mu_hat_std_dev,
  // 			     double theta_hat_mean,
  // 			     double theta_hat_std_dev,
  // 			     double alpha_hat_mean, 
  // 			     double alpha_hat_std_dev,
  // 			     double tau_square_hat_mean,
  // 			     double tau_square_hat_std_dev,
  // 			     double xi_square_mean,
  // 			     double xi_square_std_dev,
  // 			     double rho_mean,
  // 			     double rho_std_dev);

  // MultifactorStochasticVolatilityPriors(double mu_hat_mean,
  // 			     double mu_hat_std_dev,
  // 			     double theta_hat_mean,
  // 			     double theta_hat_std_dev,
  // 			     double alpha_hat_mean, 
  // 			     double alpha_hat_std_dev,
  // 			     double tau_square_hat_mean,
  // 			     double tau_square_hat_std_dev,
  // 			     double xi_square_mean,
  // 			     double xi_square_std_dev,
  // 			     double rho_mean,
  // 			     double rho_std_dev,
  // 			     double delta_t);

  const MuPrior& get_mu_prior() const;
  const XiSquarePrior& get_xi_square_prior() const;
private:
  MuPrior mu_prior_;
  XiSquarePrior xi_square_prior_;
};

// // =========== MULTIFACTOR SV PRIORS WITH JUMPS =================
// class MultifactorStochasticVolatilityPriorsWithJumps
//   : private MultifactorStochasticVolatilityPriors
// {
// private:
//   MultifactorStochasticVolatilityPriorsWithJumps();
//   MultifactorStochasticVolatilityPriorsWithJumps(double delta_t);

//   const MuPrior& get_jump_mean_prior() const;
//   const SigmaSquarePrior& get_jump_std_dev_prior() const;
//   const LambdaPrior& get_jump_rate_prior() const;

// private:
//   MuPrior jump_mean_prior_;
//   SigmaSquarePrior jump_std_dev_prior_;
//   LambdaPrior jump_rate_prior_;
// }
