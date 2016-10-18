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
  // unsigned long int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  // gsl_rng_set(r, seed);

  printf ("generator type: %s\n", gsl_rng_name (r));
  printf ("seed = %lu\n", gsl_rng_default_seed);
  printf ("first value = %lu\n", gsl_rng_get (r));

  std::ifstream file ("/home/gdinolov/Research/SV-with-leverage/simulated-data/simulation-10003/bid-ask-noise/simulated-prices-and-returns-bid-ask-noise-9-26-14-36-35.csv");

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

  ObservationalModel* observational_model = 
    new ObservationalModel(data,
			   dt);
  
  ConstantVolatilityModel* const_vol_model = 
    new ConstantVolatilityModel(observational_model,
				dt);

  observational_model->set_const_vol_model(const_vol_model);

  OUModel * ou_model = new OUModel(const_vol_model,
				   dt);
  const_vol_model->set_ou_model(ou_model);
  ou_model->set_sigmas(sigmas);

  ObservationalPosteriorSampler observational_sampler = 
    ObservationalPosteriorSampler(observational_model, r);
  ConstantVolatilityPosteriorSampler const_vol_sampler = 
    ConstantVolatilityPosteriorSampler(const_vol_model,r);

  for (unsigned i=0; i<observational_model->data_length()+1; ++i) {
    observational_sampler.draw_xi_square();
    const_vol_sampler.draw_mu_hat();

    std::cout << "xi^2 = "
	      << observational_model->get_xi_square().get_continuous_time_parameter()
	      << "; mu_hat = "
	      << const_vol_model->get_mu().get_continuous_time_parameter()
    	      << std::endl;
  }
  
  delete ou_model;
  delete const_vol_model;
  delete observational_model;
  return 0;
}
