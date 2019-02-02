#include <algorithm>
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

int main (int argc, char *argv[])
{
  if(argc < 4) {
    printf("You must provide at least one argument\n");
    printf("The order of inputs is: \n sampling (in seconds),\n data-file, \n save-location\n");
    exit(0);
  }
  // report settings
  for (int i=0;i<argc;i++) {
    printf("Argument %d:%s\n",i,argv[i]);
  }

  // GSL stuff
  gsl_rng_env_setup();
  const gsl_rng_type * T = gsl_rng_default;
  gsl_rng * r = gsl_rng_alloc (T);
  unsigned long int seed = 12345542343;
  // std::chrono::high_resolution_clock::now().time_since_epoch().count();
  gsl_rng_set(r, seed);

  printf ("generator type: %s\n", gsl_rng_name (r));
  printf ("seed = %lu\n", gsl_rng_default_seed);
  printf ("first value = %lu\n", gsl_rng_get (r));

  std::ifstream file (argv[2]);

  double dt = std::stod(std::string(argv[1]))*1000;
  int dt_int = dt;
  int dt_simulation = 100;

  int burn_in = 10000;
  int M = 100000;
  int number_paths = 100;

  // READ IN THE DATA
  std::vector<OpenCloseDatum> data_vector = std::vector<OpenCloseDatum> ();
  std::vector<double> sigma_hats_slow = std::vector<double> ();
  std::vector<double> sigma_hats_fast = std::vector<double> ();

  double previous_price_true = 0;
  double previous_price = 0;
  double log_sigma_hat = 0;

  std::string value;
  if (file.is_open()) {
    std::cout << "file opened" << std::endl;

    // price true
    std::getline(file, value, ',');
    std::cout << value << std::endl;
    previous_price_true = std::stod(value);
    
    // price 
    std::getline(file, value, ',');
    std::cout << value << std::endl;
    previous_price = std::stod(value);

    // log.sigma.hat.slow
    std::getline(file, value, ',');
    std::cout << value << std::endl;
    log_sigma_hat = std::stod(value);
    sigma_hats_slow.push_back(exp(log_sigma_hat));

    // log.sigma.hat.fast
    std::getline(file, value, ',');
    std::cout << value << std::endl;
    log_sigma_hat = std::stod(value);
    sigma_hats_fast.push_back(exp(log_sigma_hat));

    // jump
    std::getline(file, value);
    std::cout << value << std::endl;

    double previous_time = 0;
    long int index = 0;
    long int number_rows = 0;
    double log_sigma_hat_slow = 0;
    double log_sigma_hat_fast = 0;

    while ( std::getline(file, value, ',') ) {
      index = index+1;
      number_rows = number_rows + 1;
      double closing_price_true = std::stod(value);

      std::getline(file, value, ',');
      double closing_price = std::stod(value);

      std::getline(file, value, ',');
      log_sigma_hat_slow = std::stod(value);

      std::getline(file, value, ',');
      log_sigma_hat_fast = std::stod(value);

      // jump
      std::getline(file, value);

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
    std::cout << "number_rows = " << number_rows << std::endl;
    std::cout << "index = " << index << std::endl;
    std::cout << "data length = " << data_vector.size() << std::endl;
    std::cout << "sigma length = " << sigma_hats_slow.size() << std::endl;
  }

  OpenCloseData data = OpenCloseData(data_vector);
  SigmaParameter sigmas_slow = SigmaParameter(sigma_hats_slow, dt);
  SigmaParameter sigmas_fast = SigmaParameter(sigma_hats_fast, dt);

  double theta_hat_slow_mean = 1.0/(3.5*60*60*1000);
  double theta_hat_slow_std_dev = theta_hat_slow_mean / 10.0;
  double theta_hat_fast_mean = 1.0/(10*60*1000);
  double theta_hat_fast_std_dev = theta_hat_fast_mean / 10.0;

  // ============================================
  double VV = 0.116; // VIX on the log(sigma) scale
  double tau_square_hat_fast_mean = VV * theta_hat_fast_mean;
  double tau_square_hat_fast_std_dev = tau_square_hat_fast_mean * 10.0;
  double tau_square_hat_slow_mean = VV * theta_hat_slow_mean;
  double tau_square_hat_slow_std_dev = tau_square_hat_slow_mean * 10.0;
  // ===========================================

  SVModelWithJumps* model = 
    new SVModelWithJumps(data, dt,
			 theta_hat_fast_mean,
			 theta_hat_fast_std_dev,
			 theta_hat_slow_mean,
			 theta_hat_slow_std_dev,
			 tau_square_hat_fast_mean,
			 tau_square_hat_fast_std_dev,
			 tau_square_hat_slow_mean,
			 tau_square_hat_slow_std_dev);

  model->get_observational_model()->set_nu(20);
  model->get_observational_model()->set_xi_square(6.25e-8); // was 6.25e-8

  model->get_ou_model_fast()->set_rho(0.0);
  model->get_ou_model_fast()->set_tau_square_hat(tau_square_hat_fast_mean);
  model->get_ou_model_fast()->set_theta_hat(theta_hat_fast_mean);
  model->get_ou_model_fast()->set_sigmas(sigmas_fast);
  model->get_ou_model_fast()->get_theta_prior().set_theta_hat_ub(1.0/(5.0*60*1000));

  model->get_ou_model_slow()->set_rho(0.0);
  model->get_ou_model_slow()->set_tau_square_hat(tau_square_hat_slow_mean);
  model->get_ou_model_slow()->set_theta_hat(theta_hat_slow_mean);
  model->get_ou_model_slow()->set_sigmas(sigmas_slow);
  // model->get_ou_model_slow()->get_theta_prior().set_theta_hat_ub(1.0/(5.0*60*1000));

  std::cout << "theta_hat_fast = "
	    << model->get_ou_model_slow()->get_theta().get_continuous_time_parameter()
	    << std::endl;

  // check to see if data was loaded correctly
  const std::vector<double>& filtered_data =  
    model->get_constant_vol_model()->get_filtered_log_prices();
  for (const double& datum : filtered_data) {
    std::cout << datum << std::endl;
  }

  gsl_matrix * proposal_covariance_matrix_ptr = gsl_matrix_alloc(3,3);
  gsl_matrix_set_zero(proposal_covariance_matrix_ptr);
  double cc = 0.1;
  std::vector<double> proposal_sds 
  {0.1*cc,0.1*cc,1.0e-10*cc};
     // {0.2, 0.1, 0.02};
     // {0.02, 0.07, 0.01};

  std::vector<double> proposal_correlations 
  {1.0, -0.05, -0.02,
      -0.05, 1.0, 0.01,
      -0.02, 0.01, 1.0};
  // {1, 0, 0,
  //     0.00000,  1.00000, -0.15651,
  //     0.00000, -0.15651,  1.00000};
    
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

  gsl_matrix * proposal_covariance_matrix_all_ptr = gsl_matrix_alloc(5,5);
  gsl_matrix_set_zero(proposal_covariance_matrix_all_ptr);

  // alpha, theta_slow, tau2_slow, rho, theta_fast, tau2_fast
  // double c = 9;
  double c = 1.5;
  std::vector<double> proposal_sds_all 
  // {0.01*c,0.01*c,0.01*c,0.01*c,0.01*c,0.01*c};
  // {0.0239*c,0.16762*c,0.07222*c,0.07558*c,0.15949*c,0.08596*c,};
  {0.02883*c,0.25498*c,0.30276*c,0.38199*c,0.36508*c};

  std::vector<double> proposal_correlations_all
  {1,0,0,0,0,
      0,1,0,0,0,
      0,0,1,0,0,
      0,0,0,1,0,
      0,0,0,0,1}; 
  // {1.00000,0.11005,0.00000,-0.14755,0.00000,
  //     0.11005,1.00000,-0.71960,0.00000,0.00000,
  //     0.00000,-0.71960,1.00000,-0.19499,-0.10487,
  //     -0.14755,0.00000,-0.19499,1.00000,-0.51690,
  //     0.00000,0.00000,-0.10487,-0.51690,1.00000};

  for (int i=0; i<5; ++i) {
    for (int j=0; j<5; ++j) {
      int correlation_index = (i*5) + j;
      gsl_matrix_set(proposal_covariance_matrix_all_ptr, i,j, 
  		     proposal_sds_all[i]*proposal_sds_all[j]
		     *proposal_correlations_all[correlation_index]);
      std::cout << gsl_matrix_get(proposal_covariance_matrix_all_ptr, i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;

  
  SVWithJumpsPosteriorSampler sampler = 
    SVWithJumpsPosteriorSampler(model, 
				r, 
				proposal_covariance_matrix_ptr,
				proposal_covariance_matrix_all_ptr);
  
  std::vector<int> new_gammas = 
    std::vector<int> {4,1,2,1,5,5,1,3,4,5,5,3,8,4,1,2,3,3,4,3,4,4,6,5,6,5,5,5,3,3,2,5,5,6,4,3,4,6,5,4,3,3,5,2,6,7,5,5,6,5,3,4,5,4,3,2,3,4,3,3,5,8,2,3,3,2,6,4,4,2,5,4,3,3,3,5,8,0,5,7,7,1,0,0,3,2,6,2,3,4,5,3,5,4,4,5,3,6,3,6,2,3,4,2,5,2,2,1,4,3,2,7,4,4,4,4,5,1,5,4,3,7,3,7,4,5,3,2,4,6,3,3,6,8,3,4,4,6,2,2,5,2,3,1,4,5,2,5,1,3,5,6,2,3,6,1,3,4,3,3,1,2,5,6,7,7,4,2,3,5,1,3,3,4,1,7,1,3,4,4,3,4,5,1,0,5,4,2,3,2,4,1,7,2,3,7,6,2,5,3,1,5,3,4,5,4,1,5,3,4,6,5,4,4,5,4,6,3,1,4,2,5,4,3,1,3,2,3,6,2,2,5,6,2,5,2,5,3,3,5,7,4,5,0,2,4,1,4,4,4,6,3,3,3,4,2,4,6,7,3,6,1,4,2,3,2,8,6,3,6,3,3,0,6,6,4,4,4,6,4,2,1,5,0,2,3,4,1,2,3,3,6,6,5,4,6,7,2,3,6,7,4,4,3,4,2,5,2,1,3,2,5,8,6,1,1,3,8,4,2,3,2,7,4,3,7,1,3,4,4,2,2,1,1,4,1,3,4,4,3,3,3,3,7,1,3,5,4,1,5,5,7,6,0,3,5,7,6,4,4,3,5,3,1,5,3,1,4,2,7,2,5,5,5,3,7,7,2,2,4,3,4,3,7,2,4,3,4,1,9};

  for (unsigned ii=0; ii < model->data_length(); ++ii) {
    if (ii >= new_gammas.size()-1) {
      model->get_constant_vol_model()->set_gamma_element(ii,3);
    } else {
      model->get_constant_vol_model()->set_gamma_element(ii,new_gammas[ii]);
    }
  }
  gsl_matrix_free(proposal_covariance_matrix_ptr);

  // RECORD THE RESULTS
  std::string save_location = std::string(argv[3]);
  std::string sampling_period = std::to_string(dt_int);

  std::string results_name = 
    "simulated-prices-and-returns-bid-ask-noise-results-" + 
    sampling_period + ".csv";

  std::string log_sigma_hat_slow_name = 
    "log-sigma-hats-slow-" + sampling_period + ".csv";

  std::string log_sigma_hat_fast_name = 
    "log-sigma-hats-fast-" + sampling_period + ".csv";

  std::string log_filtered_prices_name = 
    "log-filtered-prices-" + sampling_period + ".csv";

  std::ofstream results (save_location + results_name);

  std::ofstream log_sigma_hats_slow (save_location + log_sigma_hat_slow_name);
  
  std::ofstream log_sigma_hats_fast (save_location + log_sigma_hat_fast_name);

  std::ofstream log_filtered_prices (save_location + log_filtered_prices_name);

  // header
  results << "alpha.hat, tau.sq.hat.slow, theta.hat.slow, tau.sq.hat.fast, theta.hat.fast, mu.hat, rho, xi.square, jump.size.mean, jump.size.variance, jump.rate, nu\n";

  // std::vector<std::vector<double>>* log_sigmas_slow_posterior = 
  //   new std::vector<std::vector<double>> (model->data_length()+1, 
  // 					  std::vector<double> (M));

  // std::vector<std::vector<double>>* log_sigmas_fast_posterior = 
  //   new std::vector<std::vector<double>> (model->data_length()+1,
  // 					  std::vector<double> (M));

  // std::vector<std::vector<double>>* log_filtered_prices_posterior = 
  //   new std::vector<std::vector<double>> (model->data_length()+1,
  // 					  std::vector<double> (M));

  // std::vector<std::vector<double>>* jump_indicators_posterior = 
  //   new std::vector<std::vector<double>> (model->data_length()+1,
  // 					  std::vector<double> (M));

  // std::vector<std::vector<double>>* jump_sizes_posterior = 
  //   new std::vector<std::vector<double>> (model->data_length()+1,
  // 					  std::vector<double> (M));

  // std::vector<std::vector<double>>* deltas_posterior = 
  //   new std::vector<std::vector<double>> (model->data_length()+1,
  // 					  std::vector<double> (M));

  // for (unsigned i=0; i<log_sigmas_slow_posterior->size(); ++i) {
  //   //    log_sigmas_slow_posterior->operator[](i) = std::vector<double> (M);
  //   log_sigmas_fast_posterior->operator[](i) = std::vector<double> (M);
  //   log_filtered_prices_posterior->operator[](i) = std::vector<double> (M);
  //   jump_indicators_posterior->operator[](i) = std::vector<double> (M);
  //   jump_sizes_posterior->operator[](i) = std::vector<double> (M); 
  //   deltas_posterior->operator[](i) = std::vector<double> (M);
  // }

  number_paths = std::min(number_paths, M);
  double path_indeces[number_paths], indeces[M];
  for (int i=0; i<M; ++i) {
    indeces[i] = (double) i;
  }
  gsl_ran_choose(r, path_indeces, number_paths, indeces, M, sizeof (double));  
  int current_path_index=0;

  // \sum_{i=1}^M x_i^2 and \sum_{i=1}^M x_i for each time point
  std::vector<double> log_filtered_prices_sum_square (model->data_length()+1, 0);
  std::vector<double> log_filtered_prices_sum (model->data_length()+1, 0);

  std::vector<double> log_sigma_hats_slow_sum_square (model->data_length()+1, 0);
  std::vector<double> log_sigma_hats_slow_sum (model->data_length()+1, 0);

  std::vector<double> log_sigma_hats_fast_sum_square (model->data_length()+1, 0);
  std::vector<double> log_sigma_hats_fast_sum (model->data_length()+1, 0);

  for (int i=0; i<M+burn_in; ++i) {
    if (i % 10 == 0) {
      std::cout << "on iteration " << i << "\n";
      std::cout << "rho = " 
		<< model->get_ou_model_fast()->get_rho().get_continuous_time_parameter()
		<< "; ";
      std::cout << "tau2_fast = " 
		<< model->get_ou_model_fast()->get_tau_square().get_continuous_time_parameter()
		<< "; ";
      std::cout << "theta_hat_fast = " 
		<< model->get_ou_model_fast()->get_theta().get_continuous_time_parameter()
		<< "; ";
      std::cout << "xi^2 = " 
		<< model->get_observational_model()->
	get_xi_square().get_continuous_time_parameter()
		<< "; ";
      std::cout << "mu.hat = " 
		<< model->get_constant_vol_model()->
	get_mu().get_continuous_time_parameter()
		<< "; ";
      std::cout << "alpha_hat = " 
		<< model->get_ou_model_fast()->
	get_alpha().get_continuous_time_parameter()
		<< "; ";
      std::cout << "nu = " 
		<< model->get_observational_model()->get_nu()
		<< "; ";
      std::cout << "dt = " << dt
		<< std::endl;

      // const std::vector<SigmaSingletonParameter>& sigmas_slow = 
      // 	model->get_ou_model_slow()->
      // 	get_sigmas().get_sigmas();

      // const std::vector<SigmaSingletonParameter>& sigmas_fast = 
      // 	model->get_ou_model_fast()->
      // 	get_sigmas().get_sigmas();
      
      // const std::vector<double>& log_filtered_prices_sample = 
      // 	model->get_constant_vol_model()->get_filtered_log_prices();	
	
      // 	  std::ofstream log_vol_fast ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-1/bid-ask-noise/problematic-fast-log-vol.csv");
      // 	  std::ofstream log_vol_slow ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-1/bid-ask-noise/problematic-slow-log-vol.csv");
      // 	  std::ofstream log_prices ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-1/bid-ask-noise/problematic-log-vol-prices.csv");
	  
      // 	for (unsigned ii=0; ii<model->data_length()+1; ++ii) {
      // 	  log_vol_fast << log(sigmas_fast[ii].
      // 			      get_discrete_time_parameter(model->
      // 							  get_delta_t()))
      // 		       << "\n";
      // 	  log_vol_slow << log(sigmas_slow[ii].
      // 			      get_discrete_time_parameter(model->
      // 							  get_delta_t()))
      // 		       << "\n";
      // 	  log_prices << log_filtered_prices_sample[ii]
      // 		     << "\n";
      // 	}
      // 	log_vol_fast.close();
      // 	log_vol_slow.close();
      // 	log_prices.close();
    }

    sampler.draw();
    if (i > burn_in-1) {
      results << model->get_ou_model_fast()->
    	get_alpha().get_continuous_time_parameter() << ",";
      results << model->get_ou_model_slow()->
    	get_tau_square().get_continuous_time_parameter() << ",";
      results << model->get_ou_model_slow()->
    	get_theta().get_continuous_time_parameter() << ",";
      results << model->get_ou_model_fast()->
    	get_tau_square().get_continuous_time_parameter() << ",";
      results << model->get_ou_model_fast()->
    	get_theta().get_continuous_time_parameter() << ",";
      results << model->get_constant_vol_model()->
    	get_mu().get_continuous_time_parameter() << ",";
      results << model->get_ou_model_fast()->
    	get_rho().get_continuous_time_parameter() << ",";
      results << model->get_observational_model()->
    	get_xi_square().get_continuous_time_parameter() << ",";
      results << model->get_constant_vol_model()->
    	get_jump_size_mean().get_continuous_time_parameter() << ",";
      results << model->get_constant_vol_model()->
    	get_jump_size_variance().get_sigma_square() << ",";
      results << model->get_constant_vol_model()->
    	get_jump_rate().get_lambda() << ",";
      results << model->get_observational_model()->
    	get_nu() << "\n";

      const std::vector<SigmaSingletonParameter>& sigmas_slow = 
    	model->get_ou_model_slow()->
    	get_sigmas().get_sigmas();

      const std::vector<SigmaSingletonParameter>& sigmas_fast = 
    	model->get_ou_model_fast()->
    	get_sigmas().get_sigmas();
      
      // const std::vector<bool>& jump_Is = 
      // 	model->get_constant_vol_model()->
      // 	get_jump_indicators();

      // const std::vector<double>& jumps = 
      // 	model->get_constant_vol_model()->
      // 	get_jump_sizes();

      // const std::vector<double>& deltas_sample = 
      // 	model->get_observational_model()->
      // 	get_deltas();

      const std::vector<double>& log_filtered_prices_sample = 
    	model->get_constant_vol_model()->get_filtered_log_prices();	

      for (unsigned j=0; j<model->data_length(); ++j) {
    	// log_sigma_hats_slow << 
    	//   (log(sigmas_slow[j].get_continuous_time_parameter())) << ",";

    	// log_sigma_hats_fast << 
    	//   (log(sigmas_fast[j].get_continuous_time_parameter())) << ",";

    	// log_filtered_prices <<
    	//   (log_filtered_prices_sample[j]) << ",";

    	log_sigma_hats_slow_sum_square[j] = log_sigma_hats_slow_sum_square[j] +
    	  (log(sigmas_slow[j].get_continuous_time_parameter()))*
    	  (log(sigmas_slow[j].get_continuous_time_parameter()));
    	log_sigma_hats_slow_sum[j] = log_sigma_hats_slow_sum[j] +
    	  (log(sigmas_slow[j].get_continuous_time_parameter()));

    	log_sigma_hats_fast_sum_square[j] = log_sigma_hats_fast_sum_square[j] +
    	  (log(sigmas_fast[j].get_continuous_time_parameter()))*
    	  (log(sigmas_fast[j].get_continuous_time_parameter()));
    	log_sigma_hats_fast_sum[j] = log_sigma_hats_fast_sum[j] +
    	  (log(sigmas_fast[j].get_continuous_time_parameter()));

    	log_filtered_prices_sum_square[j] = log_filtered_prices_sum_square[j] +
    	  (log_filtered_prices_sample[j])*
    	  (log_filtered_prices_sample[j]);
    	log_filtered_prices_sum[j] = log_filtered_prices_sum[j] +
    	  (log_filtered_prices_sample[j]);
	
    	// log_sigmas_slow_posterior->operator[](j)[i-(burn_in-1)] = 
    	//   (log(sigmas_slow[j].get_continuous_time_parameter()));
	
    	// log_sigmas_fast_posterior->operator[](j)[i-(burn_in-1)] = 
    	//   (log(sigmas_fast[j].get_continuous_time_parameter()));
	
    	// jump_indicators_posterior->operator[](j)[i-(burn_in-1)] = jump_Is[j];
    	// jump_sizes_posterior->operator[](j)[i-(burn_in-1)] = jumps[j];
    	// deltas_posterior->operator[](j)[i-(burn_in-1)] = deltas_sample[j];
	
    	// log_filtered_prices_posterior->operator[](j)[i-(burn_in-1)] =
    	//   (log_filtered_prices_sample[j]);
      }
	
      // log_sigma_hats_slow << 
      // 	(log(sigmas_slow[model->data_length()].get_continuous_time_parameter())) << "\n";

      // log_sigma_hats_fast << 
      // 	(log(sigmas_fast[model->data_length()].get_continuous_time_parameter())) << "\n";

      // log_filtered_prices << 
      // 	(log_filtered_prices_sample[model->data_length()]) << "\n";

      log_sigma_hats_slow_sum_square[model->data_length()] = 
    	log_sigma_hats_slow_sum_square[model->data_length()] +
    	  (log(sigmas_slow[model->data_length()].get_continuous_time_parameter()))*
    	  (log(sigmas_slow[model->data_length()].get_continuous_time_parameter()));
    	log_sigma_hats_slow_sum[model->data_length()] = 
    	  log_sigma_hats_slow_sum[model->data_length()] +
    	  (log(sigmas_slow[model->data_length()].get_continuous_time_parameter()));

      log_sigma_hats_fast_sum_square[model->data_length()] = 
    	log_sigma_hats_fast_sum_square[model->data_length()] +
    	  (log(sigmas_slow[model->data_length()].get_continuous_time_parameter()))*
    	  (log(sigmas_slow[model->data_length()].get_continuous_time_parameter()));
    	log_sigma_hats_fast_sum[model->data_length()] = 
    	  log_sigma_hats_fast_sum[model->data_length()] +
    	  (log(sigmas_slow[model->data_length()].get_continuous_time_parameter()));
      
      log_filtered_prices_sum_square[model->data_length()] = 
    	log_filtered_prices_sum_square[model->data_length()] +
    	(log_filtered_prices_sample[model->data_length()])*
    	(log_filtered_prices_sample[model->data_length()]);
    	log_filtered_prices_sum[model->data_length()] = 
    	  log_filtered_prices_sum[model->data_length()] +
    	  (log_filtered_prices_sample[model->data_length()]);
      
      // log_sigmas_slow_posterior->operator[](model->data_length())[i-(burn_in-1)] =
      // 	(log(sigmas_slow[model->data_length()].get_continuous_time_parameter()));
      
      // log_sigmas_fast_posterior->operator[](model->data_length())[i-(burn_in-1)] = 
      // 	(log(sigmas_fast[model->data_length()].get_continuous_time_parameter()));
      
      // log_filtered_prices_posterior->operator[](model->data_length())[i-(burn_in-1)] = 
      // 	(log_filtered_prices_sample[model->data_length()]);
      
      // jump_indicators_posterior->operator[](model->data_length())[i-(burn_in-1)] = 
      // 	jump_Is[model->data_length()];
      // jump_sizes_posterior->operator[](model->data_length())[i-(burn_in-1)] = 
      // 	jumps[model->data_length()];
      // deltas_posterior->operator[](model->data_length())[i-(burn_in-1)] = 
      // 	deltas_sample[model->data_length()];
      
    }
  }
  results.close();
  log_sigma_hats_slow.close();
  log_sigma_hats_fast.close();
  log_filtered_prices.close();

  std::string log_filtered_prices_quantiles_name = 
    "log-filtered-prices-quantiles-" + sampling_period + ".csv";
  std::string log_sigma_hat_slow_quantiles_name = 
    "log-sigma-hats-slow-quantiles-" + sampling_period + ".csv";
  std::string log_sigma_hat_fast_quantiles_name = 
    "log-sigma-hats-fast-quantiles-" + sampling_period + ".csv";

  std::ofstream log_filtered_prices_quantiles (save_location + log_filtered_prices_quantiles_name);
  std::ofstream log_sigma_hats_slow_quantiles (save_location + log_sigma_hat_slow_quantiles_name);
  std::ofstream log_sigma_hats_fast_quantiles (save_location + log_sigma_hat_fast_quantiles_name);

  for (unsigned i=0; i<log_sigma_hats_fast_sum.size(); ++i) {
    if (i == log_sigma_hats_fast_sum.size()-1) {
      // prices
      double mean_prices = log_filtered_prices_sum[i]/((double) M);
      double var_prices = 
  	(log_filtered_prices_sum_square[i] / ((double) M) -
  	 mean_prices*mean_prices);

      log_filtered_prices_quantiles << 
  	mean_prices - 1.96*sqrt(var_prices) << "," <<
  	mean_prices << "," <<
  	mean_prices + 1.96*sqrt(var_prices) << "\n";

      // slow
      double mean_slow = log_sigma_hats_slow_sum[i]/((double) M);
      double var_slow = 
  	(log_sigma_hats_slow_sum_square[i] / ((double) M) -
  	 mean_slow*mean_slow);

      std::cout << "var = " << var_slow << std::endl;

      log_sigma_hats_slow_quantiles << 
  	mean_slow - 1.96*sqrt(var_slow) << "," <<
  	mean_slow << "," <<
  	mean_slow + 1.96*sqrt(var_slow) << "\n";

      // fast
      double mean_fast = log_sigma_hats_fast_sum[i]/((double) M);
      double var_fast = 
  	(log_sigma_hats_fast_sum_square[i] / ((double) M) -
  	 mean_fast*mean_fast);

      log_sigma_hats_fast_quantiles << 
  	mean_fast - 1.96*sqrt(var_fast) << "," <<
  	mean_fast << "," <<
  	mean_fast + 1.96*sqrt(var_fast) << "\n";
    } else {
      // prices
      double mean_prices = log_filtered_prices_sum[i]/((double) M);
      double var_prices = 
  	(log_filtered_prices_sum_square[i] / ((double) M) -
  	 mean_prices*mean_prices);

      log_filtered_prices_quantiles << 
  	mean_prices - 1.96*sqrt(var_prices) << "," <<
  	mean_prices << "," <<
  	mean_prices + 1.96*sqrt(var_prices) << "\n";

      // slow
      double mean_slow = log_sigma_hats_slow_sum[i]/((double) M);
      double var_slow = 
  	(log_sigma_hats_slow_sum_square[i] / ((double) M) -
  	 mean_slow*mean_slow);

      log_sigma_hats_slow_quantiles << 
  	mean_slow - 1.96*sqrt(var_slow) << "," <<
  	mean_slow << "," <<
  	mean_slow + 1.96*sqrt(var_slow) << "\n";

      // fast
      double mean_fast = log_sigma_hats_fast_sum[i]/((double) M);
      double var_fast = 
  	(log_sigma_hats_fast_sum_square[i] / ((double) M) -
  	 mean_fast*mean_fast);

      log_sigma_hats_fast_quantiles << 
  	mean_fast - 1.96*sqrt(var_fast) << "," <<
  	mean_fast << "," <<
  	mean_fast + 1.96*sqrt(var_fast) << "\n";
    }
  }
  
  // // std::ofstream log_sigmas_slow_quantiles ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-1/bid-ask-noise/log-sigmas-slow-quantiles.csv");
  // // std::ofstream log_sigmas_fast_quantiles ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-1/bid-ask-noise/log-sigmas-fast-quantiles.csv");
  // // std::ofstream log_filtered_prices_quantiles ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-1/bid-ask-noise/log-filtered-prices-quantiles.csv");

  // // std::ofstream jump_indicators_quantiles ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-1/bid-ask-noise/jump-indicators-quantiles.csv");
  // // std::ofstream jump_sizes_quantiles ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-1/bid-ask-noise/jump-sizes-quantiles.csv");
  // // std::ofstream deltas_quantiles ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-1/bid-ask-noise/deltas-quantiles.csv");

  // // for (unsigned j=0; j<log_sigmas_fast_posterior->size(); ++j) {
  // //   int lower_quantile = log_sigmas_slow_posterior->operator[](j).size() * 0.025;
  // //   int median_index = log_sigmas_slow_posterior->operator[](j).size() * 0.5;
  // //   int upper_quantile = log_sigmas_slow_posterior->operator[](j).size() * 0.975;
    
  // //   std::sort (log_sigmas_slow_posterior->operator[](j).begin(), 
  // // 	       log_sigmas_slow_posterior->operator[](j).end());
    
  // //   log_sigmas_slow_quantiles << log_sigmas_slow_posterior->
  // //     operator[](j)[lower_quantile]
  // // 			      << ",";
  // //   log_sigmas_slow_quantiles << log_sigmas_slow_posterior->
  // //     operator[](j)[median_index]
  // // 			      << ",";
  // //   log_sigmas_slow_quantiles << log_sigmas_slow_posterior->
  // //     operator[](j)[upper_quantile]
  // // 			      << "\n";
    
  // //   std::sort (log_sigmas_fast_posterior->operator[](j).begin(), 
  // // 	       log_sigmas_fast_posterior->operator[](j).end());

  // //   log_sigmas_fast_quantiles << log_sigmas_fast_posterior->
  // //     operator[](j)[lower_quantile]
  // // 			      << ",";
  // //   log_sigmas_fast_quantiles << log_sigmas_fast_posterior->
  // //     operator[](j)[median_index]
  // // 			      << ",";
  // //   log_sigmas_fast_quantiles << log_sigmas_fast_posterior->
  // //     operator[](j)[upper_quantile]
  // // 			      << "\n";

  // //   std::sort (log_filtered_prices_posterior->operator[](j).begin(),
  // // 	       log_filtered_prices_posterior->operator[](j).end());

  // //   log_filtered_prices_quantiles << log_filtered_prices_posterior->operator[](j)[lower_quantile]
  // // 				  << ",";
  // //   log_filtered_prices_quantiles << log_filtered_prices_posterior->operator[](j)[median_index]
  // // 				  << ",";
  // //   log_filtered_prices_quantiles << log_filtered_prices_posterior->operator[](j)[upper_quantile]
  // // 				  << "\n";

  // //   // std::sort (jump_indicators_posterior->operator[](j).begin(),
  // //   // 	       jump_indicators_posterior->operator[](j).end());

  // //   // jump_indicators_quantiles << jump_indicators_posterior->operator[](j)[lower_quantile]
  // //   // 			      << ",";
  // //   // jump_indicators_quantiles << jump_indicators_posterior->operator[](j)[median_index]
  // //   // 				  << ",";
  // //   // jump_indicators_quantiles << jump_indicators_posterior->operator[](j)[upper_quantile]
  // //   // 				  << "\n";

  // //   // std::sort (jump_sizes_posterior->operator[](j).begin(),
  // //   // 	       jump_sizes_posterior->operator[](j).end());

  // //   // jump_sizes_quantiles << jump_sizes_posterior->operator[](j)[lower_quantile]
  // //   // 				  << ",";
  // //   // jump_sizes_quantiles << jump_sizes_posterior->operator[](j)[median_index]
  // //   // 				  << ",";
  // //   // jump_sizes_quantiles << jump_sizes_posterior->operator[](j)[upper_quantile]
  // //   // 				  << "\n";

  // //   // std::sort (deltas_posterior->operator[](j).begin(),
  // //   // 	       deltas_posterior->operator[](j).end());

  // //   // deltas_quantiles << deltas_posterior->operator[](j)[lower_quantile]
  // //   // 				  << ",";
  // //   // deltas_quantiles << deltas_posterior->operator[](j)[median_index]
  // //   // 				  << ",";
  // //   // deltas_quantiles << deltas_posterior->operator[](j)[upper_quantile]
  // //   // 				  << "\n";
    
  // // }
  // // std::cout << std::endl;

  log_sigma_hats_slow_quantiles.close();
  log_sigma_hats_fast_quantiles.close();
  log_filtered_prices_quantiles.close();
  // // jump_indicators_quantiles.close();
  // // jump_sizes_quantiles.close();
  // // deltas_quantiles.close();

  // // delete log_sigmas_slow_posterior;
  // // delete log_sigmas_fast_posterior;
  // // delete log_filtered_prices_posterior;
  // // // delete jump_indicators_posterior;
  // // // delete jump_sizes_posterior;
  // // // delete deltas_posterior;

  gsl_rng_free(r);
  delete model;
  return 0;
}
