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

  double dt = 5*60*1000;
  // READ IN THE DATA
  std::vector<OpenCloseDatum> data_vector = std::vector<OpenCloseDatum> ();
  std::vector<double> sigma_hats_slow = std::vector<double> ();
  std::vector<double> sigma_hats_fast = std::vector<double> ();

  std::string value;
  if (file.is_open()) {
    std::cout << "file opened" << std::endl;

    std::getline(file, value, ' ');
    double previous_price = std::stod(value);

    std::getline(file, value, ' ');
    double log_sigma_hat = std::stod(value);
    sigma_hats_slow.push_back(exp(log_sigma_hat));

    std::getline(file, value);
    log_sigma_hat = std::stod(value);
    sigma_hats_fast.push_back(exp(log_sigma_hat));

    double previous_time = 0;

    //    std::cout << previous_price << "," << log_sigma_hat << std::endl;
    while ( std::getline(file, value, ' ') ) {

      double closing_price = std::stod(value);

      std::getline(file, value, ' ');
      log_sigma_hat = std::stod(value);

      //      std::cout << closing_price << "," << log_sigma_hat << std::endl;
      sigma_hats_slow.push_back(exp(log_sigma_hat));

      std::getline(file, value);
      log_sigma_hat = std::stod(value);
      sigma_hats_fast.push_back(exp(log_sigma_hat));

      OpenCloseDatum datum = OpenCloseDatum(previous_price, 
    					    closing_price,
    					    dt,
    					    previous_time);
      previous_price = closing_price;
      previous_time = previous_time + dt;
      data_vector.push_back(datum);
    }

    std::cout << "data length = " << data_vector.size() << std::endl;
    std::cout << "sigma length = " << sigma_hats_slow.size() << std::endl;
  }

  OpenCloseData data = OpenCloseData(data_vector);
  SigmaParameter sigmas_slow = SigmaParameter(sigma_hats_slow, dt);
  SigmaParameter sigmas_fast = SigmaParameter(sigma_hats_fast, dt);

  double theta_hat_fast_mean = 1.0/(10.0*60*1000);
  double theta_hat_fast_std_dev = theta_hat_fast_mean/10.0;
  double theta_hat_slow_mean = 1.0/(1*3.25*60*60*1000);
  double theta_hat_slow_std = theta_hat_slow_mean/10.0;

  double tau_square_hat_fast_mean = 1.3e-7*25;
  double tau_square_hat_fast_std_dev = 1.3e-7*25*10;
  double tau_square_hat_slow_mean = 1.3e-7;
  double tau_square_hat_slow_std = 1e-6;

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
  
  model->get_ou_model_fast()->set_rho(-0.1);
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
  std::ofstream results ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-1/no-noise/theta-slow-log-likelihood.csv");

  // header
  results << "theta.ll.full, theta.ll.slow.model, theta.hat\n";

  double dtheta_hat = 1e-9;
  double theta_hat_min = 1e-10;
  double theta_hat_max = 1e-6;
  double theta_hat_current = theta_hat_min - dtheta_hat;
  double ll = 0;

  // MIX OVER THE GAMMAS
  int MM = 10000;
  for (int i=0; i<MM; ++i) {
    if (i % 1000 == 0) {
      std::cout << "on i=" << i << std::endl;;
    }
    //    sampler.draw_gammas_gsl();
  }

  sampler.draw_theta_hat_slow();

  while (theta_hat_current <= theta_hat_max) {
    std::cout << "theta_hat_current = " 
	      << theta_hat_current 
	      << "; ll = " << ll
	      << std::endl;
    model->get_ou_model_slow()->set_theta_hat(theta_hat_current);
    ll = model->log_likelihood();
    results << ll << ",";
    results << model->get_ou_model_slow()->log_likelihood() << ",";
    results << theta_hat_current << "\n";
    
    theta_hat_current = theta_hat_current + dtheta_hat;
  }
  results.close();
  
  gsl_rng_free(r);
  return 0;
}
