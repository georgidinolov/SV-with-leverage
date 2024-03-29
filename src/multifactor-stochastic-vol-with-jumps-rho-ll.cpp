#include <armadillo>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#define MATHLIB_STANDALONE 1
#include <Rmath.h>
#include <string>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "StochasticVolatilityPosteriorSampler.hpp"

int main ()
{
  // GSL stuff
  gsl_rng_env_setup();
  const gsl_rng_type * T = gsl_rng_default;
  gsl_rng * r = gsl_rng_alloc (T);
  unsigned long int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  gsl_rng_set(r, seed);

  printf ("generator type: %s\n", gsl_rng_name (r));
  printf ("seed = %lu\n", gsl_rng_default_seed);
  printf ("first value = %lu\n", gsl_rng_get (r));

  std::ifstream file ("/home/gdinolov/Research/SV-with-leverage/simulated-data/simulation-1/bid-ask-noise/simulated-prices-and-returns-bid-ask-noise.csv");

  double dt = 1*10*1000;
  int dt_int = dt;
  int dt_simulation = 1000;
  // READ IN THE DATA
  std::vector<OpenCloseDatum> data_vector = std::vector<OpenCloseDatum> ();
  std::vector<double> sigma_hats_slow = std::vector<double> ();
  std::vector<double> sigma_hats_fast = std::vector<double> ();

  std::string value;
  if (file.is_open()) {
    std::cout << "file opened" << std::endl;

    std::getline(file, value, ' ');
    double previous_price_true = std::stod(value);

    std::getline(file, value, ' ');
    double previous_price = std::stod(value);

    std::getline(file, value, ' ');
    double log_sigma_hat = std::stod(value);
    sigma_hats_slow.push_back(exp(log_sigma_hat));

    std::getline(file, value);
    log_sigma_hat = std::stod(value);
    sigma_hats_fast.push_back(exp(log_sigma_hat));

    double previous_time = 0;
    long int index = 0;
    long int number_rows = 1;
    double log_sigma_hat_slow = 0;
    double log_sigma_hat_fast = 0;
    
    //    std::cout << previous_price << "," << log_sigma_hat << std::endl;
    while ( std::getline(file, value, ' ') ) {
      index = index+1;
      number_rows = number_rows + 1;
      double closing_price_true = std::stod(value);

      std::getline(file, value, ' ');
      double closing_price = std::stod(value);

      std::getline(file, value, ' ');
      log_sigma_hat_slow = std::stod(value);

      std::getline(file, value);
      log_sigma_hat_fast = std::stod(value);

      if (index*dt_simulation % dt_int == 0) {
	// std::cout << "index = " << index << std::endl;
	sigma_hats_slow.push_back(exp(log_sigma_hat_slow));
	sigma_hats_fast.push_back(exp(log_sigma_hat_fast));
	
	OpenCloseDatum datum = OpenCloseDatum(previous_price,
					      closing_price,
					      dt,
					      previous_time);

	previous_price_true = closing_price_true;
	previous_price = closing_price;
	previous_time = previous_time + dt;
	data_vector.push_back(datum);
      }
    }

    std::cout << "data length = " << data_vector.size() << std::endl;
    std::cout << "sigma length = " << sigma_hats_slow.size() << std::endl;
  }

  OpenCloseData data = OpenCloseData(data_vector);
  SigmaParameter sigmas_slow = SigmaParameter(sigma_hats_slow, dt);
  SigmaParameter sigmas_fast = SigmaParameter(sigma_hats_fast, dt);

  double theta_hat_fast_mean = 1.0/(10*60*1000);
  double theta_hat_fast_std_dev = theta_hat_fast_mean/10.0;
  double theta_hat_slow_mean = 1.0/(6.5*60*60*1000);
  double theta_hat_slow_std = theta_hat_slow_mean/10.0;

  // ========================================
  double VV = 0.116; // VIX on the log(sigma) scale
  double tau_square_hat_fast_mean = VV * theta_hat_fast_mean;
  double tau_square_hat_fast_std_dev = tau_square_hat_fast_mean / 10.0;
  double tau_square_hat_slow_mean = VV * theta_hat_slow_mean;
  double tau_square_hat_slow_std = tau_square_hat_slow_mean / 10.0;
  // ========================================

  SVModelWithJumps* model = 
    new SVModelWithJumps(data, dt,
			 theta_hat_fast_mean,
			 theta_hat_fast_std_dev,
			 theta_hat_slow_mean,
			 theta_hat_slow_std,
			 tau_square_hat_fast_mean,
 			 tau_square_hat_fast_std_dev,
			 tau_square_hat_slow_mean,
			 tau_square_hat_slow_std);
  
  model->get_ou_model_fast()->set_rho(-0.2);
  model->get_ou_model_fast()->set_tau_square_hat(tau_square_hat_fast_mean);
  model->get_ou_model_fast()->set_theta_hat(theta_hat_fast_mean);
  model->get_ou_model_fast()->set_sigmas(sigmas_fast);

  model->get_ou_model_slow()->set_rho(0.0);
  model->get_ou_model_slow()->set_tau_square_hat(tau_square_hat_slow_mean);
  model->get_ou_model_slow()->set_theta_hat(theta_hat_slow_mean);
  model->get_ou_model_slow()->set_sigmas(sigmas_slow);

  gsl_matrix * proposal_covariance_matrix_ptr = gsl_matrix_alloc(3,3);
  gsl_matrix_set_zero(proposal_covariance_matrix_ptr);
  std::vector<double> proposal_sds {1e-04, 1e+00, 4e-1*0.5};
  std::vector<double> proposal_correlations 
  {1.00000000,  0.0, 0.0,
      0.0,  1.00000000, -0.1,
      0.0, -0.1,  1.00000000};
  
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      int correlation_index = (i*3) + j;
      gsl_matrix_set(proposal_covariance_matrix_ptr, i,j, 
  		     proposal_sds[i]*proposal_sds[j]*proposal_correlations[correlation_index]);
      std::cout << gsl_matrix_get(proposal_covariance_matrix_ptr, i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
  
  SVWithJumpsPosteriorSampler sampler = 
    SVWithJumpsPosteriorSampler(model, 
				r, 
				proposal_covariance_matrix_ptr);

  gsl_matrix_free(proposal_covariance_matrix_ptr);

  // RECORD THE RESULTS
  std::ofstream results ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-1/bid-ask-noise/rho-log-likelihood.csv");

  // header
  results << "rho.ll.full, rho.ll.fast.model, rho\n";

  double drho = 1e-3;
  double rho_min = -0.15;
  double rho_max = 0.0;
  double rho_ll = -1.0*HUGE_VAL;
  double rho_ll_max = rho_ll;
  double rho_max_ll = rho_min;
  // MIX OVER THE GAMMAS
  int burn = 100;
  for (int i=0; i<burn; ++i) {
    if (i % 10 == 0) {
      std::vector<double> out = sampler.rho_mle();
      double rho_MLE = out[0];
      double rho_current_now = model->
	get_ou_model_fast()->get_rho().
	get_continuous_time_parameter();

      double dr = 0.001;
      model->get_ou_model_fast()->set_rho(rho_MLE);
      double ll = model->log_likelihood();
      model->get_ou_model_fast()->set_rho(rho_MLE+dr);
      double ll_plus_drho = model->log_likelihood();
      model->get_ou_model_fast()->set_rho(rho_MLE-dr);
      double ll_minus_drho = model->log_likelihood();
      model->get_ou_model_fast()->set_rho(rho_current_now);
      std::cout << "ddrho = " << (ll_plus_drho - 2*ll + ll_minus_drho)/(dr*dr)
		<< "\n";
      std::cout << "rho_MLE_var = " << -1.0*(dr*dr)/(ll_plus_drho - 2*ll + ll_minus_drho)
		<< "\n";
            
      std::cout << "rho_MLE = " << rho_MLE << "\n";
      std::cout << "on i=" << i << std::endl;
    }

    sampler.draw_gammas_gsl();
    sampler.draw_sigmas();
    sampler.draw_filtered_log_prices();
  }
  
  // int MM = 200;
  // for (int i=0; i<MM; ++i) {
    double rho_current = rho_min - drho;
    while (rho_current <= rho_max) {
      model->get_ou_model_fast()->set_rho(rho_current);
      rho_ll = model->log_likelihood();

      if (rho_ll > rho_ll_max) {
	rho_ll_max = rho_ll;
	rho_max_ll = rho_current;
      }
      std::cout << "rho_current = " 
      		<< rho_current 
      		<< "; ll = " << model->get_ou_model_fast()->log_likelihood_rho(rho_current)
		<< "; deriv = " << 
	model->get_ou_model_fast()->
	rho_deriv_numeric_nominal_scale(rho_current,
					model->
					get_ou_model_fast()->
					get_theta().
					get_discrete_time_parameter(model->
								    get_delta_t()),
					model->
					get_ou_model_fast()->
					get_tau_square().
					get_discrete_time_parameter(model->
								    get_delta_t()),
								    1e-5)
		<< std::endl;
      
      results << rho_ll << ",";
      results << model->get_ou_model_fast()->log_likelihood_rho(rho_current) << ",";
      results << rho_current << "\n";
      
      rho_current = rho_current + drho;
    }
  // }
  results.close();

  std::cout << "rho_max_ll = " << rho_max_ll << "\n";
  std::vector<double> out = sampler.rho_mle();
  drho = 0.01;
  std::cout << "rho_max_ll_alg = " << out[0] << "\n";
  std::cout << "numeric deriv at MLE dt=1e-4 = " 
	    << model->get_ou_model_fast()->
    rho_deriv_numeric_nominal_scale(out[0],
				    model->
				    get_ou_model_fast()->
				    get_theta().
				    get_discrete_time_parameter(model->
								get_delta_t()),
				    model->
				    get_ou_model_fast()->
				    get_tau_square().
				    get_discrete_time_parameter(model->
								get_delta_t()),
				    drho = 1e-4) 
	    << "\n";
  std::cout << "numeric double deriv at MLE dt=1e-4 = " 
	    << model->get_ou_model_fast()->
    rho_double_deriv_numeric_nominal_scale(out[0],
					   model->
					   get_ou_model_fast()->
					   get_theta().
					   get_discrete_time_parameter(model->
								       get_delta_t()),
					   model->
					   get_ou_model_fast()->
					   get_tau_square().
					   get_discrete_time_parameter(model->
								    get_delta_t()),
					   drho = 1e-4) 
	    << "\n";
  sampler.draw_rho();
  
  std::vector<double> tilde_mle =  model->get_ou_model_fast()->rho_theta_tau_square_tilde_MLE();
  
  double rho_tilde = tilde_mle[0];
  double rho = 2.0*exp(rho_tilde)/(1.0 + exp(rho_tilde)) - 1.0;
  
  double theta_tilde = tilde_mle[1];
  double theta = exp(theta_tilde)/(1.0 + exp(theta_tilde));

  double tau_square_tilde = tilde_mle[2];
  double tau_square = exp(tau_square_tilde);

  std::cout << "rho = " << rho 
	    << "; theta = " << theta
	    << "; tau_square = " << tau_square
	    << std::endl;

  std::cout << "dll/dr (numeric) = " << model->get_ou_model_fast()->
    rho_deriv_numeric_nominal_scale(rho,
				    theta,
				    tau_square,
				    1e-5)
	    << "\n"
	    << "dll/dr (analytic) = " << model->get_ou_model_fast()->
    rho_deriv_analytic_nominal_scale(rho,
				     theta,
				     tau_square)
	    << "\n";

  std::cout << "dll/dr (tilde, numeric) = " << model->get_ou_model_fast()->
    rho_deriv_numeric_tilde_scale(rho_tilde,
				  theta_tilde,
				  tau_square_tilde,
				  1e-6)
	    << "\n"
	    << "dll/dr (tilde, analytic) = " << model->get_ou_model_fast()->
    rho_deriv_analytic_tilde_scale(rho_tilde,
				   theta_tilde,
				   tau_square_tilde)
	    << "\n";

  double a11 = -1.0 * model->get_ou_model_fast()->
    rho_double_deriv_numeric_tilde_scale(rho_tilde,
					 theta_tilde,
					 tau_square_tilde,
					 1e-5);
  double a12 = -1.0 * model->get_ou_model_fast()->
    rho_theta_deriv_numeric_tilde_scale(rho_tilde,
					theta_tilde,
					tau_square_tilde,
					1e-5,
					1e-5);

  double a13 = -1.0 * model->get_ou_model_fast()->
    rho_tau_square_deriv_numeric_tilde_scale(rho_tilde,
					     theta_tilde,
					     tau_square_tilde,
					     1e-5,
					     1e-5);
  double a22 = -1.0 * model->get_ou_model_fast()->
    theta_double_deriv_numeric_tilde_scale(rho_tilde,
					   theta_tilde,
					   tau_square_tilde,
					   1e-5);
  double a23 = -1.0 * model->get_ou_model_fast()->
    theta_tau_square_deriv_numeric_tilde_scale(rho_tilde,
					       theta_tilde,
					       tau_square_tilde,
					       1e-5,
					       1e-5);
  double a33 = -1.0 * model->get_ou_model_fast()->
    tau_square_double_deriv_numeric_tilde_scale(rho_tilde,
						theta_tilde,
						tau_square_tilde,
						1e-5);
  arma::mat inv_info_mat = { {a11, a12, a13},
			     {a12, a22, a23},
			     {a13, a23, a33} };

  std::cout << "inv_info_mat = \n"
	    << inv_info_mat
	    << std::endl;

  std::cout << "info_mat = \n"
	    << inv_sympd(inv_info_mat)
	    << std::endl;
  gsl_rng_free(r);
  return 0;
}
