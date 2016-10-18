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

  std::ifstream file ("/home/gdinolov/Research/SV-with-leverage/simulated-data/simulation-10003/bid-ask-noise/simulated-prices-and-returns-bid-ask-noise-9-26-14-36-35.csv");

  double dt = 1*1*10;
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

  double theta_hat_fast_mean = 1.0/(10.0*60*1000);
  double theta_hat_fast_std_dev = theta_hat_fast_mean/10.0;
  double theta_hat_slow_mean = 1.0/(1*3.25*60*60*1000);
  double theta_hat_slow_std = theta_hat_slow_mean/10.0;

  MultifactorStochasticVolatilityModel* model = 
    new MultifactorStochasticVolatilityModel(data, dt,
					     theta_hat_fast_mean,
					     theta_hat_fast_std_dev,
					     theta_hat_slow_mean,
					     theta_hat_slow_std);
  model->get_ou_model_fast()->set_rho(-0.1);
  model->get_ou_model_slow()->set_rho(0.0);

  std::cout << "theta_slow_mean=" << model->get_ou_model_slow()->
    get_theta_prior().get_theta_mean() << std::endl;
  std::cout << "theta_slow_sd=" << model->get_ou_model_slow()->
    get_theta_prior().get_theta_std_dev() << std::endl;

  std::cout << "theta_fast_mean=" << model->get_ou_model_fast()->
    get_theta_prior().get_theta_mean() << std::endl;
  std::cout << "theta_fast_sd=" << model->get_ou_model_fast()->
    get_theta_prior().get_theta_std_dev() << std::endl;


  model->generate_data(5*6.5*60*60*1000,r);
}
