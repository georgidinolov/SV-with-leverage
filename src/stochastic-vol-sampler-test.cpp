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

  std::ifstream file ("/home/gdinolov/Research/SV-with-leverage/simulated-data/simulation-1/no-noise/simulated-prices-and-returns-no-noise.csv");

  double dt = 1*60*1000;

  // READ IN THE DATA
  std::vector<OpenCloseDatum> data_vector = std::vector<OpenCloseDatum> ();
  std::vector<double> sigma_hats = std::vector<double> ();

  std::string value;
  if (file.is_open()) {
    std::cout << "file opened" << std::endl;

    std::getline(file, value, ' ');
    double previous_price = std::stod(value);

    std::getline(file, value);
    double log_sigma_hat = std::stod(value);
    sigma_hats.push_back(exp(log_sigma_hat));

    double previous_time = 0;

    //    std::cout << previous_price << "," << log_sigma_hat << std::endl;
    while ( std::getline(file, value, ' ') ) {

      double closing_price = std::stod(value);

      std::getline(file, value);
      log_sigma_hat = std::stod(value);

      //      std::cout << closing_price << "," << log_sigma_hat << std::endl;
      sigma_hats.push_back(exp(log_sigma_hat));

      OpenCloseDatum datum = OpenCloseDatum(previous_price, 
    					    closing_price,
    					    dt,
    					    previous_time);
      previous_price = closing_price;
      previous_time = previous_time + dt;
      data_vector.push_back(datum);
    }

    std::cout << "data length = " << data_vector.size() << std::endl;
    std::cout << "sigma length = " << sigma_hats.size() << std::endl;
  }

  OpenCloseData data = OpenCloseData(data_vector);
  SigmaParameter sigmas = SigmaParameter(sigma_hats, dt);

  StochasticVolatilityModel* model = 
    new StochasticVolatilityModel(data, dt);

  model->get_ou_model()->set_sigmas(sigmas);
  model->get_ou_model()->set_rho(-0.5);
  gsl_matrix * proposal_covariance_matrix_ptr = gsl_matrix_alloc(3,3);
  gsl_matrix_set_zero(proposal_covariance_matrix_ptr);
  std::vector<double> proposal_sds {1e-01, 2e-1, 1.8e-01};
  std::vector<double> proposal_correlations 
  {1.00000000,  0.0,  0.0,
      0.0,  1.00000000, 0.0,
      0.0,  0.0,  1.00000000};
  
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
  
  StochasticVolatilityPosteriorSampler sampler = 
    StochasticVolatilityPosteriorSampler(model, r,
					 proposal_covariance_matrix_ptr);
  gsl_matrix_free(proposal_covariance_matrix_ptr);

  // RECORD THE RESULTS
  std::ofstream results ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-10003/bid-ask-noise/simulated-prices-and-returns-bid-ask-noise-9-26-14-36-35-results.csv");


  std::ofstream log_sigmas ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-10003/bid-ask-noise/log-sigmas.csv");

  std::ofstream log_filtered_prices ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-10003/bid-ask-noise/log-filtered-prices.csv");

  // header
  results << "alpha_hat, tau_sq_hat, theta_hat, mu_hat, rho, xi_square\n";

  int burn_in = 0;
  int M = 5000;
  for (int i=0; i<M+burn_in; ++i) {
    std::cout << i << std::endl;
    sampler.draw();
    if (i > burn_in-1) {
      results 
	<< model->get_ou_model()->get_alpha().get_continuous_time_parameter() << ",";
      results 
	<< model->get_ou_model()->get_tau_square().get_continuous_time_parameter() << ",";
      results 
	<< model->get_ou_model()->get_theta().get_continuous_time_parameter() << ",";
      results 
	<< model->get_const_vol_model()->get_mu().get_continuous_time_parameter() << ",";
      results 
	<< model->get_ou_model()->get_rho().get_continuous_time_parameter() << ",";
      results 
	<< model->get_observational_model()->
	get_xi_square().get_continuous_time_parameter() << "\n";

      for (unsigned j=0; j<model->data_length(); ++j) {
  	log_sigmas << log(model->
			  get_ou_model()->
			  get_sigmas().get_sigmas()[j].get_continuous_time_parameter())
  		   << ",";

  	log_filtered_prices 
  	  << model->get_const_vol_model()->get_filtered_log_prices()[j]
  	  << ",";
      }
      log_sigmas 
  	<< log(model->
	       get_ou_model()->
	       get_sigmas().get_sigmas()[model->data_length()].get_continuous_time_parameter())
  	<< "\n";

      log_filtered_prices 
  	<< model->get_const_vol_model()->get_filtered_log_prices()[model->data_length()]
  	<< "\n";
    }
  }
  results.close();
  log_sigmas.close();
  log_filtered_prices.close();
  
  gsl_rng_free(r);
  return 0;
}
