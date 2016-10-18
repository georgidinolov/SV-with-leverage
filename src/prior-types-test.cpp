#include <cmath>
#include <iostream>
#include "PriorTypes.hpp"

using namespace std;

int main ()
{
  double delta_t = 1000;
  ThetaPrior theta_prior = ThetaPrior(delta_t);
  std::cout << "theta_hat_mean = "
	    << theta_prior.get_theta_hat_mean() << std::endl;
  std::cout << "theta_hat_std_dev = "
	    << theta_prior.get_theta_hat_std_dev() << std::endl;

  std::cout << "theta_mean = "
	    << theta_prior.get_theta_mean() << std::endl;
  std::cout << "theta_std_dev = "
	    << theta_prior.get_theta_std_dev() << std::endl;

  TauSquarePrior tau_sq_prior = TauSquarePrior(theta_prior,
					       delta_t);
  std::cout << "tau2 Dicrete time shape = " 
	    <<  tau_sq_prior.get_tau_square_shape() << std::endl;
  std::cout << "tau2 Dicrete time scale = " 
	    << tau_sq_prior.get_tau_square_scale() << std::endl;
  std::cout << "tau2 Dicrete time mean = " 
	    << tau_sq_prior.get_tau_square_mean() << std::endl;
  std::cout << "tau2 Dicrete time std dev = " 
	    << tau_sq_prior.get_tau_square_std_dev() << std::endl;
  std::cout << "tau2_hat_mean = " 
	    << tau_sq_prior.get_tau_square_hat_mean() << std::endl;
  std::cout << "tau2_hat_std_dev = "
	    << tau_sq_prior.get_tau_square_hat_std_dev() << std::endl;

  MuPrior mu_prior = MuPrior(delta_t);
  std::cout << "mu_hat_mean = "
	    << mu_prior.get_mu_hat_mean() << std::endl;
  std::cout << "mu_hat_std_dev = "
	    << mu_prior.get_mu_hat_std_dev() << std::endl;
  std::cout << "mu_mean = "
	    << mu_prior.get_mu_mean() << std::endl;
  std::cout << "mu_std_dev = "
	    << mu_prior.get_mu_std_dev() << std::endl;

  AlphaPrior alpha_prior = AlphaPrior(delta_t);
  std::cout << "alpha_hat_mean = "
	    << alpha_prior.get_alpha_hat_mean() << std::endl;
  std::cout << "alpha_hat_std_dev = "
   	    << alpha_prior.get_alpha_hat_std_dev() << std::endl;

  RhoPrior rho_prior = RhoPrior();
  std::cout << "rho_mean = "
	    << rho_prior.get_rho_mean() << std::endl;
  std::cout << "rho_std_dev = "
   	    << rho_prior.get_rho_std_dev() << std::endl;

  XiSquarePrior xi_sq_prior = XiSquarePrior();
  std::cout << "xi_sq_mean = "
	    << xi_sq_prior.get_xi_square_mean() << std::endl;
  std::cout << "rho_std_dev = "
   	    << xi_sq_prior.get_xi_square_std_dev() << std::endl;
  
  StochasticVolatilityPriors priors = StochasticVolatilityPriors();

  // Clone 
  StochasticVolatilityPriors * cloned_priors = priors.clone();
  std::cout << "theta prior hat mean = " 
	    << cloned_priors->get_theta_prior().get_theta_hat_mean()
	    << std::endl;

  delete cloned_priors;
  return 0;
}
