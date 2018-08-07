#include <algorithm>
#include "src/armadillo-7.600.2/usr/include/armadillo"
#include <cmath>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <limits>
#include "MultivariateNormal.hpp"
#define MATHLIB_STANDALONE
#include <Rmath.h>
#include "StochasticVolatilityPosteriorSampler.hpp"

using namespace arma;

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

  inline double logit(double p) {
    return log(p/(1.0-p));
  }

  inline double rtruncnorm(gsl_rng * r, double a, double b, double mean, double std_dev) {
    double sample = gsl_ran_flat(r,
				 pnorm(a,mean,std_dev,1,0),
				 pnorm(b,mean,std_dev,1,0));
    double out = mean + 
      std_dev * qnorm(sample,
		      0.0, 1.0, 1, 0);

    return out;
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

// ======== OBSERVATIONAL POSTERIOR SAMPELR ==========================
ObservationalPosteriorSampler::
ObservationalPosteriorSampler(ObservationalModel * obs_mod,
			      gsl_rng * rng)
  : observational_model_(obs_mod),
    rng_(rng)
{}

void ObservationalPosteriorSampler::draw_xi_square()
{
  double alpha = 
    observational_model_->get_xi_square_prior().get_xi_square_shape() + 
    observational_model_->data_length()/2.0;
  
  double beta =
     observational_model_->get_xi_square_prior().get_xi_square_scale();
  const std::vector<double>& deltas =
    observational_model_->get_deltas();
  
  const std::vector<double>& filtered_log_prices =
    observational_model_->get_filtered_log_prices();
  
  beta = beta +
    (0.5*square((observational_model_->get_data_element(0).get_open())
		   - filtered_log_prices[0]));

  for (unsigned i=0; i<observational_model_->data_length(); ++i) {
    beta = beta + 
      (0.5*deltas[i]*square((observational_model_->get_data_element(i).get_close())
				- filtered_log_prices[i+1]));
  }

  // std::cout << "alpha = " << alpha << "; " << "beta = " << beta << "\n";
  double xi_square_inv = gsl_ran_gamma(rng_, alpha, 1.0/beta);
  double xi_square = 1.0/xi_square_inv;
  observational_model_->set_xi_square(xi_square);
}

void ObservationalPosteriorSampler::draw_deltas()
{
  double nu = observational_model_->get_nu();
  const std::vector<double>& filtered_log_prices = 
    observational_model_->get_filtered_log_prices();
  double xi_square = observational_model_->get_xi_square().
    get_continuous_time_parameter();

  double alpha = 0.5*(nu+1.0);
  double beta = 0;
  double log_price_i = 0;
  double delta_i = 0;
  for (unsigned i=0; i<observational_model_->data_length(); ++i) {
    log_price_i = (observational_model_->get_data_element(i).get_close());
    beta = 0.5 * (nu + 
		  square(log_price_i - filtered_log_prices[i+1])/xi_square);
    delta_i = gsl_ran_gamma_knuth(rng_,
				  alpha,
				  1.0/beta);
    observational_model_->set_delta_element(i, delta_i);
  }
}

void ObservationalPosteriorSampler::draw_nu()
{
  double nu_current = observational_model_->get_nu();
  const NuPrior& nu_prior = observational_model_->get_nu_prior();
  
  long unsigned int number_nus = 
    nu_prior.get_nu_max() -
    nu_prior.get_nu_min() + 1;

  double nu_proposal = 0;
  if (nu_current == nu_prior.get_nu_min()) {
    nu_proposal = nu_current + 
      gsl_rng_uniform_int(rng_, 2);
  } else if (nu_current == nu_prior.get_nu_max()) {
    nu_proposal = nu_current + 
      (gsl_rng_uniform_int(rng_, 2) - 1);
  } else {
    nu_proposal = nu_current + 
      (gsl_rng_uniform_int(rng_, 3) - 1);
  }

  const std::vector<double>& deltas = observational_model_->get_deltas();
  double ll_current = 0;
  double ll_proposal = 0;

  for (unsigned i=0; i<observational_model_->data_length(); ++i) {
    ll_current = ll_current +
      dgamma(deltas[i], nu_current/2.0, 2.0/nu_current, 1);
    ll_proposal = ll_proposal +
      dgamma(deltas[i], nu_proposal/2.0, 2.0/nu_proposal, 1);
  }
  ll_current = ll_current + 
    nu_prior.log_likelihood(nu_current);
  ll_proposal = ll_proposal + 
    nu_prior.log_likelihood(nu_proposal);

  double log_a_acceptance = ll_proposal -
    ll_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    observational_model_->set_nu(nu_proposal);
    // std::cout << "nu accepted: " << nu_proposal << "\n";
  } else {
    // std::cout << "nu NOT accepted: " << nu_current << "\n";
    observational_model_->set_nu(nu_current);
  }
}



// ================== CONST VOL SAMPLER ==========================
ConstantVolatilityPosteriorSampler::
ConstantVolatilityPosteriorSampler(ConstantVolatilityModel * const_vol_mod,
				   gsl_rng * rng)
  : const_vol_model_(const_vol_mod),
    rng_(rng)
{}

void ConstantVolatilityPosteriorSampler::draw_mu_hat()
{
  // we need the close for each period, and the opening for the frist period
  double variance_inv = 0;
  double mean_over_variance = 0;
  const std::vector<SigmaSingletonParameter>& sigmas = 
    const_vol_model_->get_ou_model()->get_sigmas().get_sigmas();

  for (unsigned i=0; i<const_vol_model_->data_length(); ++i) {
    double S_j_minus_1 = 
      exp(const_vol_model_->
	  get_observational_model()->
	  get_data_element(i).get_open());

    double S_j = 
      exp(const_vol_model_->
	  get_observational_model()->
	  get_data_element(i).get_close());

    if (i>0) {
      S_j_minus_1 = 
	exp(const_vol_model_->
	    get_observational_model()->
	    get_data_element(i-1).get_close());
      S_j = 
	exp(const_vol_model_->
	    get_observational_model()->
	    get_data_element(i).get_close());
    }

    double sigma_j = 
      sigmas[i+1].get_discrete_time_parameter(const_vol_model_->get_delta_t());
    
    variance_inv = variance_inv + 1.0 / square(sigma_j);
    mean_over_variance = mean_over_variance + 
      (log(S_j) - log(S_j_minus_1)) / (sigma_j*sigma_j);
  }

  double likelihood_variance = 1.0 / variance_inv;
  double likelihood_mean = mean_over_variance * likelihood_variance;
  //  std::cout << "mm = " << likelihood_mean << "; vv=" << likelihood_variance << "\n";

  double prior_mean = const_vol_model_->get_mu_prior().get_mu_mean();
  double prior_std_dev = const_vol_model_->get_mu_prior().get_mu_std_dev();
  double prior_var = prior_std_dev * prior_std_dev;

  double var = 1.0 / (1.0/likelihood_variance + 1.0/prior_var);
  double mean = (prior_mean / prior_var + likelihood_mean / likelihood_variance) * var;

  double mu_sample = mean + sqrt(var)*gsl_ran_gaussian(rng_, 1.0);
  double mu_hat_sample = mu_sample / const_vol_model_->get_delta_t();
  const_vol_model_->set_mu_hat(mu_hat_sample);
  const_vol_model_->set_y_star_ds();
}

// ================= CONST VOL SAMPELR WITH JUMPS  ==============
ConstantVolatilityWithJumpsPosteriorSampler::
ConstantVolatilityWithJumpsPosteriorSampler(
   ConstantMultifactorVolatilityModelWithJumps * const_vol_mod,
   gsl_rng * rng)
  : const_vol_model_(const_vol_mod),
    rng_(rng)
{}

void ConstantVolatilityWithJumpsPosteriorSampler::draw_mu_hat() 
{
  // we need the close for each period, and the opening for the frist period
  double variance_inv = 0;
  double mean_over_variance = 0;
  const std::vector<SigmaSingletonParameter>& sigmas_slow = 
    const_vol_model_->get_ou_model_slow()->get_sigmas().get_sigmas();

  const std::vector<SigmaSingletonParameter>& sigmas_fast = 
    const_vol_model_->get_ou_model_fast()->get_sigmas().get_sigmas();

  const std::vector<double>& jump_sizes =
    const_vol_model_->get_jump_sizes();
  
  const std::vector<double>& filtered_log_prices =
    const_vol_model_->get_filtered_log_prices();

  const std::vector<double>& h_fast = 
    const_vol_model_->get_ou_model_fast()->
    get_sigmas().get_discrete_time_log_sigmas();
  double tau_square_fast = const_vol_model_->
    get_ou_model_fast()->get_tau_square().
    get_discrete_time_parameter(const_vol_model_->
				get_ou_model_fast()->get_delta_t());
  double rho = const_vol_model_->get_ou_model_fast()->
    get_rho().get_continuous_time_parameter();

  double alpha = const_vol_model_->get_ou_model_fast()->get_alpha().
    get_discrete_time_parameter(const_vol_model_->
				get_ou_model_fast()->get_delta_t());
  double theta_fast = const_vol_model_->get_ou_model_fast()->get_theta().
    get_discrete_time_parameter(const_vol_model_->
				get_ou_model_fast()->get_delta_t());

  for (unsigned i=0; i<const_vol_model_->data_length(); ++i) {
    double log_S_j = filtered_log_prices[i+1];
    double log_S_j_minus_1 = filtered_log_prices[i];

    double eta_t_2 = 
      ((h_fast[i+1] - alpha) - theta_fast*(h_fast[i] - alpha))
      / sqrt(tau_square_fast);

    double sigma_j_slow = 
      sigmas_slow[i].get_discrete_time_parameter(const_vol_model_->get_delta_t());
    double sigma_j_fast = 
      sigmas_fast[i].get_discrete_time_parameter(const_vol_model_->get_delta_t());
    
    variance_inv = variance_inv + 1.0 / (sigma_j_slow*sigma_j_fast*(1-square(rho)));
    mean_over_variance = mean_over_variance + 
      (log_S_j - log_S_j_minus_1 - 
       jump_sizes[i] - 
       sqrt(sigma_j_slow*sigma_j_fast)*rho*eta_t_2)
      / (sigma_j_slow*sigma_j_fast*(1-square(rho)));
  }

  double likelihood_variance = 1.0 / variance_inv;
  double likelihood_mean = mean_over_variance * likelihood_variance;
  //  std::cout << "mm = " << likelihood_mean << "; vv=" << likelihood_variance << "\n";

  double prior_mean = const_vol_model_->get_mu_prior().get_mu_mean();
  double prior_std_dev = const_vol_model_->get_mu_prior().get_mu_std_dev();
  double prior_var = prior_std_dev * prior_std_dev;

  double var = 1.0 / (1.0/likelihood_variance + 1.0/prior_var);
  double mean = (prior_mean / prior_var + likelihood_mean / likelihood_variance) * var;

  double mu_sample = mean + sqrt(var)*gsl_ran_gaussian(rng_, 1.0);
  double mu_hat_sample = mu_sample / const_vol_model_->get_delta_t();
  const_vol_model_->set_mu_hat(mu_hat_sample);
  //  std::cout << "sampled MU; variance_inv = " << variance_inv << "\n";
  const_vol_model_->set_y_star_ds();
}

void ConstantVolatilityWithJumpsPosteriorSampler::draw_jump_size_mean() 
{
  // we need the close for each period, and the opening for the frist period

  const std::vector<SigmaSingletonParameter>& sigmas_slow = 
    const_vol_model_->get_ou_model_slow()->get_sigmas().get_sigmas();

  const std::vector<SigmaSingletonParameter>& sigmas_fast = 
    const_vol_model_->get_ou_model_fast()->get_sigmas().get_sigmas();

  const std::vector<double>& filtered_log_prices =
    const_vol_model_->get_filtered_log_prices();

  double jump_size_prior_mean = const_vol_model_->get_jump_size_prior().
    get_mu_hat_mean();
  double jump_size_prior_std_dev = const_vol_model_->get_jump_size_prior().
    get_mu_hat_std_dev();
  double mu = const_vol_model_->get_mu().
    get_discrete_time_parameter(const_vol_model_->get_delta_t());
  
  const std::vector<double>& h_fast = 
    const_vol_model_->get_ou_model_fast()->get_sigmas().get_discrete_time_log_sigmas();
  double tau_square_fast = const_vol_model_->get_ou_model_fast()->get_tau_square().
    get_discrete_time_parameter(const_vol_model_->get_ou_model_fast()->get_delta_t());
  double rho = const_vol_model_->get_ou_model_fast()->
    get_rho().get_continuous_time_parameter();
  double alpha = const_vol_model_->get_ou_model_fast()->get_alpha().
    get_discrete_time_parameter(const_vol_model_->get_ou_model_fast()->get_delta_t());
  double theta_fast = const_vol_model_->get_ou_model_fast()->get_theta().
    get_discrete_time_parameter(const_vol_model_->get_ou_model_fast()->get_delta_t());

  double variance_inv = 1.0/square(jump_size_prior_std_dev);
  double mean_over_variance = jump_size_prior_mean/
    square(jump_size_prior_std_dev);

  const std::vector<bool>& jump_indicators = 
    const_vol_model_->get_jump_indicators();
  for (unsigned i=0; i<const_vol_model_->data_length(); ++i) {
    if (jump_indicators[i]) {
      double log_S_j = filtered_log_prices[i+1];
      double log_S_j_minus_1 = filtered_log_prices[i];
      
      double eta_t_2 = 
	((h_fast[i+1] - alpha) - theta_fast*(h_fast[i] - alpha))
	/ sqrt(tau_square_fast);

      double sigma_j_slow = 
	sigmas_slow[i].
	get_discrete_time_parameter(const_vol_model_->get_delta_t());
      double sigma_j_fast = 
	sigmas_fast[i].
	get_discrete_time_parameter(const_vol_model_->get_delta_t());
      
      variance_inv = variance_inv + 1.0 / (sigma_j_slow*sigma_j_fast*(1-square(rho)));
      mean_over_variance = mean_over_variance + 
	(log_S_j - log_S_j_minus_1 - mu - 
	 sqrt(sigma_j_slow*sigma_j_fast)*rho*eta_t_2)
	/ (sigma_j_slow*sigma_j_fast*(1-square(rho)));
    }
  }
  
  double variance = 1.0 / variance_inv;
  double mean = mean_over_variance * variance;
  // std::cout << "mm = " << mean << "; vv=" << variance << "\n";

  double sample = mean + sqrt(variance)*gsl_ran_gaussian(rng_, 1.0);

  const_vol_model_->set_jump_size_mean(sample);
}

void ConstantVolatilityWithJumpsPosteriorSampler::draw_jump_size_variance() 
{
  // we need the close for each period, and the opening for the frist period

  double jump_size_mean = const_vol_model_->
    get_jump_size_prior().get_mu_hat_mean();

  double jump_size_variance_prior_alpha = const_vol_model_->
    get_jump_size_variance_prior().get_sigma_square_alpha();

  double jump_size_variance_prior_beta = const_vol_model_->
    get_jump_size_variance_prior().get_sigma_square_beta();

  double posterior_alpha = jump_size_variance_prior_alpha;
  double posterior_beta = jump_size_variance_prior_beta;

  const std::vector<bool>& jump_indicators = 
    const_vol_model_->get_jump_indicators();

  const std::vector<double>& jump_sizes = 
    const_vol_model_->get_jump_sizes();

  for (unsigned i=0; i<const_vol_model_->data_length(); ++i) {
    if (jump_indicators[i]) {
      
      double jump_size = jump_sizes[i];
      posterior_alpha = posterior_alpha + 1;
      posterior_beta = posterior_beta + 0.5*square(jump_size-jump_size_mean);
    }
  }
  

  double var_inv = gsl_ran_gamma(rng_, 
				 posterior_alpha,
				 1.0/posterior_beta);
  double var = 1.0/var_inv;
  const_vol_model_->set_jump_size_variance(var);
}

void ConstantVolatilityWithJumpsPosteriorSampler::draw_jump_rate() 
{
  double T = const_vol_model_->get_observational_model()->
    get_data_element(const_vol_model_->data_length()-1).get_t_close();

  long unsigned N_jumps = const_vol_model_->get_number_jumps();
  //  std::cout << "N_jumps = " << N_jumps << "\n";
  
  double alpha_posterior = 
    const_vol_model_->get_jump_rate_prior().get_lambda_alpha() + 
    N_jumps;

  double beta_posterior = 
    const_vol_model_->get_jump_rate_prior().get_lambda_beta() + 
    T;

  // std::cout << "posterior_mean = " << alpha_posterior / beta_posterior
  // 	    << "\n";

  double lambda = gsl_ran_gamma(rng_, 
				alpha_posterior,
				1.0/beta_posterior);
  const_vol_model_->set_jump_rate(lambda);
}

void ConstantVolatilityWithJumpsPosteriorSampler::draw_jump_indicators() 
{
  const std::vector<SigmaSingletonParameter>& sigmas_slow = 
    const_vol_model_->get_ou_model_slow()->get_sigmas().get_sigmas();

  const std::vector<SigmaSingletonParameter>& sigmas_fast = 
    const_vol_model_->get_ou_model_fast()->get_sigmas().get_sigmas();

  double mu = const_vol_model_->get_mu().
    get_discrete_time_parameter(const_vol_model_->get_delta_t());

  const std::vector<double>& h_fast = 
    const_vol_model_->get_ou_model_fast()->get_sigmas().get_discrete_time_log_sigmas();

  double tau_square_fast = const_vol_model_->
    get_ou_model_fast()->
    get_tau_square().
    get_discrete_time_parameter(const_vol_model_->
				get_ou_model_fast()->
				get_delta_t());

  double rho = const_vol_model_->
    get_ou_model_fast()->
    get_rho().get_continuous_time_parameter();

  double alpha = const_vol_model_->
    get_ou_model_fast()->get_alpha().
    get_discrete_time_parameter(const_vol_model_->
				get_ou_model_fast()->
				get_delta_t());

  double theta_fast = const_vol_model_->
    get_ou_model_fast()->
    get_theta().
    get_discrete_time_parameter(const_vol_model_->
				get_ou_model_fast()->
				get_delta_t());

  double jump_size_mean = const_vol_model_->
    get_jump_size_mean().
    get_continuous_time_parameter();
  double jump_size_var = const_vol_model_->
    get_jump_size_variance().
    get_sigma_square();
  double jump_rate = const_vol_model_->
    get_jump_rate().
    get_lambda();

  const std::vector<double>& filtered_log_prices =
    const_vol_model_->get_filtered_log_prices();

  double *P;
  long unsigned number_jumps = 0;

  for (unsigned i=0; i<const_vol_model_->data_length(); ++i) {
    double log_S_j = filtered_log_prices[i+1];
    double log_S_j_minus_1 = filtered_log_prices[i];

    double sigma_j_slow = 
      sigmas_slow[i].get_discrete_time_parameter(const_vol_model_->get_delta_t());
    double sigma_j_fast = 
      sigmas_fast[i].get_discrete_time_parameter(const_vol_model_->get_delta_t());

    double eta_t_2 = 
      ((h_fast[i+1] - alpha) - theta_fast*(h_fast[i] - alpha))
      / sqrt(tau_square_fast);

    double ll_jump = dnorm(log_S_j - log_S_j_minus_1 - 
			   sqrt(sigma_j_fast*sigma_j_slow)*rho*eta_t_2,
			   jump_size_mean + mu,
			   sqrt(jump_size_var 
				+ sigma_j_fast*sigma_j_slow*(1-square(rho))),
			   1)
      + log(1.0 - exp(-1.0*jump_rate*const_vol_model_->get_delta_t()));
    double ll_no_jump = dnorm(log_S_j - log_S_j_minus_1 - 
			      sqrt(sigma_j_fast*sigma_j_slow)*rho*eta_t_2,
			      mu,
			      sqrt(sigma_j_fast*sigma_j_slow*(1-square(rho))),
			      1) +
      (-1.0*jump_rate*const_vol_model_->get_delta_t());

    if (ll_jump >= ll_no_jump) {
      ll_no_jump = ll_no_jump - ll_jump;
      ll_jump = 0.0;
    } else {
      ll_jump = ll_jump - ll_no_jump;
      ll_no_jump = 0.0;
    }

    double normalized_posterior_probabilities[2];
    normalized_posterior_probabilities[0] = exp(ll_no_jump);
    normalized_posterior_probabilities[1] = exp(ll_jump);
    
    P = normalized_posterior_probabilities;
    gsl_ran_discrete_t * g = 
      gsl_ran_discrete_preproc(2,P);
    int J = gsl_ran_discrete (rng_,g);
    gsl_ran_discrete_free(g);
    if (J==0) {
      const_vol_model_->set_jump_indicator(i, false);
      const_vol_model_->set_jump_size(i, 0.0);
    } else {
      number_jumps = number_jumps + 1;
      const_vol_model_->set_jump_indicator(i, true);
      
      double posterior_jump_size_var = 1.0/
	(1.0/(sigma_j_slow*sigma_j_fast*(1-square(rho))) + 1.0/jump_size_var);
      double posterior_jump_size_mean = posterior_jump_size_var * 
	((log_S_j - 
	  log_S_j_minus_1 - mu - 
	  sqrt(sigma_j_fast*sigma_j_slow)*rho*eta_t_2)
	 /(sigma_j_slow*sigma_j_fast*(1-square(rho))) +
	 jump_size_mean / jump_size_var);
	
      double jump_size = posterior_jump_size_mean + 
	sqrt(posterior_jump_size_var) * gsl_ran_gaussian(rng_, 1.0);
      
      if (std::abs(jump_size - (log_S_j - log_S_j_minus_1 - mu)) <= 1e-16) {
	std::cout << "WARNING: jumps occurred when it shouldn't have " 
		  << "at index " << i << "\n"; 
	std::cout << "log(sigma_j_slow) = " << log(sigma_j_slow) << "\n";
	std::cout << "log(sigma_j_fast) = " << log(sigma_j_fast) << "\n";
	std::cout << "P(jump) = " 
		  << 1.0 - exp(-1.0*jump_rate*const_vol_model_->get_delta_t())
		  << "\n";
	std::cout << "ll_jump = " << dnorm(log_S_j - log_S_j_minus_1,
					   jump_size_mean + mu,
					   sqrt(jump_size_var + sigma_j_fast*sigma_j_slow),
					   1) + 
	  log(1.0 - exp(-1.0*jump_rate*const_vol_model_->get_delta_t())) << "\n";
	std::cout << "ll_no_jump = " << dnorm(log_S_j - log_S_j_minus_1,
					      mu,
					      sqrt(sigma_j_fast*sigma_j_slow),
					      1) +
	  (-1.0*jump_rate*const_vol_model_->get_delta_t()) << "\n";
	
	//   std::ofstream log_vol_fast ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-1/bid-ask-noise/problematic-fast-log-vol.csv");
	//   std::ofstream log_vol_slow ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-1/bid-ask-noise/problematic-slow-log-vol.csv");
	//   std::ofstream log_prices ("/home/gdinolov/Research/SV-with-leverage/results-simulated-data/simulation-1/bid-ask-noise/problematic-log-vol-prices.csv");
	// for (unsigned ii=0; ii<const_vol_model_->data_length()+1; ++ii) {
	//   log_vol_fast << log(sigmas_fast[ii].
	// 		      get_discrete_time_parameter(const_vol_model_->
	// 						  get_delta_t()))
	// 	       << "\n";
	//   log_vol_slow << log(sigmas_slow[ii].
	// 		      get_discrete_time_parameter(const_vol_model_->
	// 						  get_delta_t()))
	// 	       << "\n";
	//   log_prices << filtered_log_prices[ii]
	// 	     << "\n";
	// }
	// log_vol_fast.close();
	// log_vol_slow.close();
	// log_prices.close();
      }
      const_vol_model_->set_jump_size(i, jump_size);
    }
  }
  const_vol_model_->set_number_jumps(number_jumps);
  const_vol_model_->set_y_star_ds();  
}

// ============== OU POSTERIOR SAMPLER ================================
OUPosteriorSampler::OUPosteriorSampler(OUModel * ou_model,
				       gsl_rng * rng,
				       const gsl_matrix * proposal_covariance_matrix)
 : ou_model_(ou_model),
   rng_(rng),
   theta_tau_square_rho_proposal_(ThetaTauSquareRhoProposal(proposal_covariance_matrix))
{}

OUPosteriorSampler::~OUPosteriorSampler()
{}

void OUPosteriorSampler::draw_alpha_hat()
{
  std::vector<double> post_mean_var = ou_model_->alpha_posterior_mean_var();
  double likelihood_mean = post_mean_var[0];
  double likelihood_var = post_mean_var[1];

  double prior_mean = ou_model_->get_alpha_prior().get_alpha_mean();
  double prior_var = square(ou_model_->get_alpha_prior().get_alpha_std_dev());

  double posterior_var = 1.0/(1.0/likelihood_var + 1.0/prior_var);
  double posterior_mean = posterior_var * (prior_mean/prior_var + likelihood_mean/likelihood_var);

  double alpha = posterior_mean + 
    sqrt(posterior_var)*gsl_ran_gaussian(rng_, 1.0);
  double alpha_hat = alpha - 0.5*log(ou_model_->get_delta_t());

  ou_model_->set_alpha_hat(alpha_hat);
}

void OUPosteriorSampler::draw_rho_tnorm() 
{
  std::vector<double> rho_mean_var = 
    ou_model_-> rho_posterior_mean_var();

  double rho_mean = rho_mean_var[0];
  double rho_var = rho_mean_var[1];

  double rho_proposal = rtruncnorm(rng_, -1, 1, rho_mean, sqrt(rho_var));
  double rho_current = ou_model_->get_rho().get_continuous_time_parameter();

  double q_proposal_given_current = dtruncnorm(rho_proposal, -1, 1, 
					       rho_mean, sqrt(rho_var),
					       true);

  double q_current_given_proposal = dtruncnorm(rho_current, -1, 1, 
					       rho_mean, sqrt(rho_var),
					       true);

  ou_model_->set_rho(rho_proposal);
  double ll_proposal = 
    ou_model_->log_likelihood() + 
    ou_model_->get_rho_prior().log_likelihood(rho_proposal) +
    q_current_given_proposal;
    
  ou_model_->set_rho(rho_current);
  double ll_current = 
    ou_model_->log_likelihood() + 
    ou_model_->get_rho_prior().log_likelihood(rho_current) +
    q_proposal_given_current;

  double log_a_acceptance = 
    ll_proposal - ll_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    //    std::cout << "; MOVE ACCEPTED ";
    ou_model_->set_rho(rho_proposal);
  } else {
    ou_model_->set_rho(rho_current);
  }  
}

void OUPosteriorSampler::draw_rho_norm() 
{
  double rho_current = ou_model_->get_rho().get_continuous_time_parameter();
  double rho_tilde_current = logit((rho_current + 1.0)/2.0);

  std::vector<double> rho_mean_var = 
    ou_model_-> rho_posterior_mean_var();

  double rho_mean = rho_mean_var[0];
  double rho_var = rho_mean_var[1];

  double rho_tilde_mean = logit((rho_mean+1.0)/2.0);
  double rho_tilde_var =
    (-2.0/(square(rho_mean)-1.0) * rho_var);

  double rho_tilde_proposal = 
    rho_tilde_mean + sqrt(rho_tilde_var)*gsl_ran_gaussian(rng_, 1.0);
  double rho_proposal = 
    2 * exp(rho_tilde_proposal) / (exp(rho_tilde_proposal)+1) - 1;

  double q_proposal_given_current = dnorm(rho_tilde_proposal, 
					  rho_tilde_mean,
					  sqrt(rho_tilde_var), 1);

  double q_current_given_proposal = dnorm(rho_tilde_current, 
					  rho_tilde_mean,
					  sqrt(rho_tilde_var), 1);

  ou_model_->set_rho(rho_current);
  double ll_current = 
    ou_model_->log_likelihood() + 
    ou_model_->get_rho_prior().log_likelihood(rho_current) +
    square(rho_tilde_current) - 2.0*log(exp(rho_tilde_current)+1) + 
    q_proposal_given_current;

  ou_model_->set_rho(rho_proposal);
  double ll_proposal = 
    ou_model_->log_likelihood() + 
    ou_model_->get_rho_prior().log_likelihood(rho_proposal) +
    square(rho_tilde_proposal) - 2.0*log(exp(rho_tilde_proposal)+1) + 
    q_current_given_proposal;

  double log_a_acceptance = 
    ll_proposal - ll_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    //    std::cout << "; MOVE ACCEPTED RHO =" << rho_proposal << std::endl;
    ou_model_->set_rho(rho_proposal);
  } else {
    ou_model_->set_rho(rho_current);
  }  
}

void OUPosteriorSampler::draw_rho_norm_walk() 
{
  double rho_current = ou_model_->get_rho().get_continuous_time_parameter();
  double rho_tilde_current = logit((rho_current + 1.0)/2.0);

  std::vector<double> rho_mean_var = 
    ou_model_-> rho_posterior_mean_var();

  double rho_mean = rho_mean_var[0];
  double rho_var = rho_mean_var[1];
  // INCORPORATING CURRENT RHO INTO MEAN AND VAR
  double rho_mean_over_var = (rho_mean / rho_var) / (1-square(rho_current));
  rho_var = 1.0 / ((1.0/rho_var) / (1-square(rho_current)));
  rho_mean = rho_var * rho_mean_over_var;

  double rho_tilde_mean = logit((rho_mean+1.0)/2.0);
  double rho_tilde_var =
    (-2.0/(square(rho_mean)-1.0) * rho_var);

  double rho_tilde_proposal = 
    rho_tilde_mean + sqrt(rho_tilde_var)*gsl_ran_gaussian(rng_, 1.0);
  double rho_proposal = 
    2 * exp(rho_tilde_proposal) / (exp(rho_tilde_proposal)+1) - 1;

  double q_proposal_given_current = dnorm(rho_tilde_proposal, 
					  rho_tilde_mean,
					  sqrt(rho_tilde_var), 1);

  // CALCULATIONS FOR PROPOSAL
  ou_model_->set_rho(rho_proposal);
  rho_mean_var = ou_model_-> rho_posterior_mean_var();
  rho_mean = rho_mean_var[0];
  rho_var = rho_mean_var[1];
  // INCORPORATING PROPOSAL RHO
  rho_mean_over_var = (rho_mean / rho_var) / (1-square(rho_proposal));
  rho_var = 1.0 / ((1.0/rho_var) / (1-square(rho_proposal)));
  rho_mean = rho_var * rho_mean_over_var;
  rho_tilde_mean = logit((rho_mean+1.0)/2.0);
  rho_tilde_var = (-2.0/(square(rho_mean)-1.0) * rho_var);

  double q_current_given_proposal = dnorm(rho_tilde_current, 
					  rho_tilde_mean,
					  sqrt(rho_tilde_var), 1);

  ou_model_->set_rho(rho_current);
  double ll_current = 
    ou_model_->log_likelihood() + 
    ou_model_->get_rho_prior().log_likelihood(rho_current) +
    square(rho_tilde_current) - 2.0*log(exp(rho_tilde_current)+1) + 
    q_proposal_given_current;

  ou_model_->set_rho(rho_proposal);
  double ll_proposal = 
    ou_model_->log_likelihood() + 
    ou_model_->get_rho_prior().log_likelihood(rho_proposal) +
    square(rho_tilde_proposal) - 2.0*log(exp(rho_tilde_proposal)+1) + 
    q_current_given_proposal;

  double log_a_acceptance = 
    ll_proposal - ll_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    //    std::cout << "; MOVE ACCEPTED ";
    ou_model_->set_rho(rho_proposal);
  } else {
    ou_model_->set_rho(rho_current);
  }  
}

void OUPosteriorSampler::draw_theta_hat() 
{
  double prior_mean = 
    ou_model_->get_theta_prior().get_theta_mean();
  double prior_var = 
    square(ou_model_->get_theta_prior().get_theta_std_dev());

  const std::vector<double>& h = 
    ou_model_->get_sigmas().get_discrete_time_log_sigmas();
  const std::vector<int> ds = 
    ou_model_->get_const_vol_model()->get_ds();
  double alpha = 
    ou_model_->get_alpha().get_discrete_time_parameter(ou_model_->get_delta_t());
  double rho = 
    ou_model_->get_rho().get_continuous_time_parameter();
  double tau_square = 
    ou_model_->get_tau_square().get_discrete_time_parameter(ou_model_->get_delta_t());
  const std::vector<int> gammas = 
    ou_model_->get_const_vol_model()->get_gammas().get_gammas();
  
  const std::vector<double>& m = 
    ou_model_->get_const_vol_model()->get_gammas().get_mixture_means();

  const std::vector<double>& as = 
    ou_model_->get_const_vol_model()->get_as();
  const std::vector<double>& bs = 
    ou_model_->get_const_vol_model()->get_bs();
  const std::vector<double>& y_star =
    ou_model_->get_const_vol_model()->get_y_star();

  double mean_over_variance = 0;
  double variance_inv = 0;

  for (unsigned i=0; i<ou_model_->data_length(); ++i) {
    double Ft = sqrt(tau_square)*
      ds[i] * rho * 2.0 * bs[gammas[i]] * exp(m[gammas[i]]/2.0);
    double Mt = sqrt(tau_square) * ds[i] * rho * exp(m[gammas[i]]/2.0) * 
      ( as[gammas[i]] + bs[gammas[i]] * 2.0 * (y_star[i] - m[gammas[i]]/2.0));

    mean_over_variance = mean_over_variance + 
      ( (h[i] - alpha) * (h[i+1] - alpha - Mt + Ft*h[i]) ) / 
      (tau_square * (1-square(rho)));

    variance_inv = variance_inv + 
      square(h[i] - alpha) / (tau_square * (1-square(rho)));
  }

  double likelihood_var = 1.0/variance_inv;
  double likelihood_mean = likelihood_var * mean_over_variance;

  double posterior_var = 
    1.0 / (1.0/likelihood_var + 1.0/prior_var);
  double posterior_mean = 
    posterior_var * (likelihood_mean / likelihood_var + prior_mean / prior_var);
  double theta_sample = rtruncnorm(rng_, 0.0, 1.0, 
				   posterior_mean, sqrt(posterior_var));
  double theta_hat_sample = 
    -1.0*log(theta_sample) / ou_model_->get_delta_t();

  // std::cout << "likelihood_mean = " << likelihood_mean << "\n";
  // std::cout << "theta_hat_sample SLOW = " << theta_hat_sample << "\n";

  double theta_current = 
    ou_model_->get_theta().get_discrete_time_parameter(ou_model_->get_delta_t());

  double theta_hat_current = 
    ou_model_->get_theta().get_continuous_time_parameter();

  // double log_a_theta = 
  //   0.5 * (log(1-square(theta_sample)) - log(1-square(theta_current))) 
  //   - 1.0/(2.0*tau_square) * 
  //   ( (1-square(theta_sample)) - (1-square(theta_current)) ) * 
  //   square(h[0] - alpha);

  // if (log(gsl_ran_flat(rng_,0,1)) < log_a_theta) {
  //   ou_model_->set_theta_hat(theta_hat_sample);
  //   std::cout << "SLOW THETA ACCEPTED\n";
  // }

  ou_model_->set_theta_hat(theta_hat_current);
  double ll_current = ou_model_->log_likelihood();
  double q_proposal_given_current = dtruncnorm(theta_sample,
  					       0.0, 1.0,
  					       posterior_mean, sqrt(posterior_var), 1);

  ou_model_->set_theta_hat(theta_hat_sample);
  double ll_proposal = ou_model_->log_likelihood();
  double q_current_given_proposal = dtruncnorm(theta_current,
  					       0.0, 1.0,
  					       posterior_mean, sqrt(posterior_var), 1);

  // std::cout << "ll_current = " << ll_current << "; "
  // 	    << "ll_proposal = " << ll_proposal << "\n";
  // std::cout << "theta_current = " << theta_current << "; "
  // 	    << "theta_proposal = " << theta_sample << "\n";
  double log_a_theta = 
    (ll_proposal + 
     ou_model_->get_theta_prior().log_likelihood(theta_sample) + 
     q_current_given_proposal)
    -
    (ll_current + 
     ou_model_->get_theta_prior().log_likelihood(theta_current) + 
     q_proposal_given_current);

  if (log(gsl_ran_flat(rng_,0,1)) < log_a_theta) {
    ou_model_->set_theta_hat(theta_hat_sample);
    //    std::cout << "SLOW THETA ACCEPTED\n";
  } else {
    ou_model_->set_theta_hat(theta_hat_current);
    //    std::cout << "slow theta NOT accepted\n";
  }

}

void OUPosteriorSampler::draw_tau_square_hat()
{
  std::vector<double> proposal_alpha_beta = ou_model_->
    tau_square_posterior_shape_rate();

  double proposal_alpha = proposal_alpha_beta[0];
  double proposal_beta = proposal_alpha_beta[1];

  double tau_square_proposal_inv = gsl_ran_gamma(rng_, 
						 proposal_alpha,
						 1.0/proposal_beta);

  double tau_square_proposal = 1.0/tau_square_proposal_inv;
  double tau_square_current = ou_model_->
    get_tau_square().get_discrete_time_parameter(ou_model_->get_delta_t());

  double theta_hat = ou_model_->
    get_theta().get_continuous_time_parameter();

  double tau_square_hat_proposal = 
    tau_square_proposal / 
    ((1.0 - exp(-2.0*theta_hat*ou_model_->get_delta_t()))/(2.0*theta_hat));
  double tau_square_hat_current = 
    ou_model_->
    get_tau_square().get_continuous_time_parameter();

  ou_model_->set_tau_square_hat(tau_square_hat_proposal);
  double log_likelihood_proposal = ou_model_->log_likelihood();

  ou_model_->set_tau_square_hat(tau_square_hat_current);
  double log_likelihood_current = ou_model_->log_likelihood();

  double q_current_given_proposal = 
    dgamma(1.0/tau_square_current, proposal_alpha, 1.0/proposal_beta, 1);

  double q_proposal_given_current = 
    dgamma(tau_square_proposal_inv, proposal_alpha, 1.0/proposal_beta, 1);
  
  double log_a_acceptance = 
    (log_likelihood_proposal
     + ou_model_->get_tau_square_prior().log_likelihood(tau_square_proposal) 
     + q_current_given_proposal)
    -
    (log_likelihood_current
     + ou_model_->get_tau_square_prior().log_likelihood(tau_square_current)
     + q_proposal_given_current);
  
  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    ou_model_->set_tau_square_hat(tau_square_hat_proposal);
  } else {
    ou_model_->set_tau_square_hat(tau_square_hat_current);
  }

  // std::cout << "tau_sq_hat SLOW= " 
  // 	    << ou_model_->get_tau_square().get_continuous_time_parameter()
  // 	    << std::endl;

}

void OUPosteriorSampler::draw_theta_tau_square_rho()
{ 
  std::vector<double> mean {ou_model_->get_rho().get_continuous_time_parameter(),
      ou_model_->get_theta().get_discrete_time_parameter(ou_model_->get_delta_t()),
      ou_model_->get_tau_square().get_discrete_time_parameter(ou_model_->get_delta_t())};

  std::vector<double> proposal = 
    theta_tau_square_rho_proposal_.propose_parameters(rng_, mean);

  double rho_proposal = proposal[0];
  double theta_proposal = proposal[1];
  double tau_square_proposal = proposal[2];

  double rho_current = ou_model_->get_rho().get_continuous_time_parameter();
  
  double theta_hat_proposal = -1.0*log(theta_proposal) / 
    ou_model_->get_delta_t();
  double theta_hat_current = ou_model_->get_theta().get_continuous_time_parameter();
  double theta_current =  ou_model_->get_theta().
    get_discrete_time_parameter(ou_model_->get_delta_t());

  double tau_square_hat_proposal = 
    tau_square_proposal / 
    ((1.0 - exp(-2.0*theta_hat_proposal*ou_model_->get_delta_t()))/(2.0*theta_hat_proposal));
  double tau_square_hat_current = ou_model_->get_tau_square().get_continuous_time_parameter();
  double tau_square_current = ou_model_->get_tau_square().
    get_discrete_time_parameter(ou_model_->get_delta_t());

  ou_model_->set_tau_square_hat(tau_square_hat_proposal);
  ou_model_->set_theta_hat(theta_hat_proposal);
  ou_model_->set_rho(rho_proposal);

  double log_likelihood_proposal = ou_model_->log_likelihood() + 
    ou_model_->get_rho_prior().log_likelihood(rho_proposal) + 
    ou_model_->get_theta_prior().log_likelihood(theta_proposal) +
    ou_model_->get_tau_square_prior().log_likelihood(tau_square_proposal) + 
    log(rho_proposal + 1) + log((1-rho_proposal)/2.0) + 
    log(theta_proposal) + log(1-theta_proposal) + 
    log(tau_square_proposal);

  ou_model_->set_tau_square_hat(tau_square_hat_current);
  ou_model_->set_theta_hat(theta_hat_current);
  ou_model_->set_rho(rho_current);
  double log_likelihood_current = ou_model_->log_likelihood() +
    ou_model_->get_rho_prior().log_likelihood(rho_current) + 
    ou_model_->get_theta_prior().log_likelihood(theta_current) + 
    ou_model_->get_tau_square_prior().log_likelihood(tau_square_current) + 
    log(rho_current + 1) + log((1-rho_current)/2.0) + 
    log(theta_current) + log(1-theta_current) + 
    log(tau_square_current);
  
  // std::cout << rho_current << " "
  // 	    << theta_hat_current << " "
  // 	    << tau_square_hat_current << " " 
  // 	    << std::endl;

  // std::cout << rho_proposal << " "
  // 	    << theta_hat_proposal << " "
  // 	    << tau_square_hat_proposal << " ";
	 

  // std::cout << "; ll_current = " << log_likelihood_current 
  // 	    << "; ll_proposal = " << log_likelihood_proposal;
  // std::cout << std::endl;

  double log_a_acceptance = log_likelihood_proposal -
    log_likelihood_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    //    std::cout << "; MOVE ACCEPTED ";
    ou_model_->set_tau_square_hat(tau_square_hat_proposal);
    ou_model_->set_theta_hat(theta_hat_proposal);
    ou_model_->set_rho(rho_proposal);
    acceptance_ratio_ = (acceptance_ratio_*number_iterations_ + 1) / 
      (number_iterations_ + 1);
  }

}

void OUPosteriorSampler::draw_theta_tau_square()
{ 
  std::vector<double> mean {ou_model_->get_rho().get_continuous_time_parameter(),
      ou_model_->get_theta().get_discrete_time_parameter(ou_model_->get_delta_t()),
      ou_model_->get_tau_square().get_discrete_time_parameter(ou_model_->get_delta_t())};

  std::vector<double> proposal = 
    theta_tau_square_rho_proposal_.propose_parameters(rng_, mean);

  double rho_current = ou_model_->get_rho().get_continuous_time_parameter();

  // HERE RHO IS KEPT AS BEFORE, SO WE ARE ONLY ACCEPTING/REJECTING THETA AND TAU^2
  double rho_proposal = rho_current;
  double theta_proposal = proposal[1];
  double tau_square_proposal = proposal[2];

  double theta_hat_proposal = -1.0*log(theta_proposal) / 
    ou_model_->get_delta_t();
  double theta_hat_current = ou_model_->get_theta().get_continuous_time_parameter();
  double theta_current =  ou_model_->get_theta().
    get_discrete_time_parameter(ou_model_->get_delta_t());

  double tau_square_hat_proposal = 
    tau_square_proposal / 
    ((1.0 - exp(-2.0*theta_hat_proposal*ou_model_->get_delta_t()))/(2.0*theta_hat_proposal));
  double tau_square_hat_current = ou_model_->get_tau_square().get_continuous_time_parameter();
  double tau_square_current = ou_model_->get_tau_square().
    get_discrete_time_parameter(ou_model_->get_delta_t());

  ou_model_->set_tau_square_hat(tau_square_hat_proposal);
  ou_model_->set_theta_hat(theta_hat_proposal);
  ou_model_->set_rho(rho_proposal);

  double rho_tilde_proposal = logit((rho_proposal+1.0)/2.0);
  double theta_tilde_proposal = logit(theta_proposal);
  double tau_square_tilde_proposal = log(tau_square_proposal);

  double rho_tilde_current = logit((rho_current+1.0)/2.0);
  double theta_tilde_current = logit(theta_current);
  double tau_square_tilde_current = log(tau_square_current);

  double log_likelihood_proposal = ou_model_->log_likelihood() +
    ou_model_->get_rho_prior().log_likelihood(rho_proposal) + 
    ou_model_->get_theta_prior().log_likelihood(theta_proposal) +
    ou_model_->get_tau_square_prior().log_likelihood(tau_square_proposal)
    + log(2.0) + rho_tilde_proposal -2*log(exp(rho_tilde_proposal) + 1.0)
    + theta_tilde_proposal - 2.0*log(exp(theta_tilde_proposal) + 1.0)
    + tau_square_tilde_proposal;

  ou_model_->set_tau_square_hat(tau_square_hat_current);
  ou_model_->set_theta_hat(theta_hat_current);
  ou_model_->set_rho(rho_current);
  double log_likelihood_current = ou_model_->log_likelihood() +
    ou_model_->get_rho_prior().log_likelihood(rho_current) + 
    ou_model_->get_theta_prior().log_likelihood(theta_current) + 
    ou_model_->get_tau_square_prior().log_likelihood(tau_square_current)
    + log(2.0) + rho_tilde_current -2*log(exp(rho_tilde_current) + 1.0)
    + theta_tilde_current - 2.0*log(exp(theta_tilde_current) + 1.0)
    + tau_square_tilde_current;
  
  // std::cout << rho_current << " "
  // 	    << theta_hat_current << " "
  // 	    << tau_square_hat_current << " " 
  // 	    << std::endl;

  // std::cout << rho_proposal << " "
  // 	    << theta_hat_proposal << " "
  // 	    << tau_square_hat_proposal << " ";
	 

  // std::cout << "; ll_current = " << log_likelihood_current 
  // 	    << "; ll_proposal = " << log_likelihood_proposal;

  double log_a_acceptance = log_likelihood_proposal -
    log_likelihood_current;


  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    //    std::cout << "; move accepted ";
    ou_model_->set_tau_square_hat(tau_square_hat_proposal);
    ou_model_->set_theta_hat(theta_hat_proposal);
    ou_model_->set_rho(rho_proposal);
    acceptance_ratio_ = (acceptance_ratio_*number_iterations_ + 1) / 
      (number_iterations_ + 1);
  }
}

void OUPosteriorSampler::set_theta_lower_bound(double lb)
{
  theta_tau_square_rho_proposal_.set_theta_lower_bound(lb);
}
void OUPosteriorSampler::set_theta_upper_bound(double ub)
{
  theta_tau_square_rho_proposal_.set_theta_upper_bound(ub);
}

// ======================= FAST OU SAMPLER =========================
// Doesn't sample alpha; that's the job of the sampler holding this
// sampler

FastOUPosteriorSampler::
FastOUPosteriorSampler(FastOUModel * ou_model_fast,
		       gsl_rng * rng,
		       const gsl_matrix * proposal_covariance_matrix)
  : ou_model_fast_(ou_model_fast),
    rng_(rng),
    theta_tau_square_rho_proposal_(ThetaTauSquareRhoProposal(proposal_covariance_matrix))
{}

void FastOUPosteriorSampler::
draw_theta_tau_square_rho()
{
  std::vector<double> mean {
    ou_model_fast_->get_rho().get_continuous_time_parameter(),
      ou_model_fast_->get_theta().get_discrete_time_parameter(ou_model_fast_->get_delta_t()),
      ou_model_fast_->get_tau_square().get_discrete_time_parameter(ou_model_fast_->get_delta_t())};

  std::vector<double> proposal = 
    theta_tau_square_rho_proposal_.propose_parameters(rng_, mean);

  double rho_proposal = proposal[0];
  double theta_proposal = proposal[1];
  double tau_square_proposal = proposal[2];

  double rho_current = ou_model_fast_->get_rho().get_continuous_time_parameter();

  double theta_hat_proposal = -1.0*log(theta_proposal) / 
    ou_model_fast_->get_delta_t();
  double theta_hat_current = ou_model_fast_->get_theta().get_continuous_time_parameter();
  double theta_current =  ou_model_fast_->get_theta().
    get_discrete_time_parameter(ou_model_fast_->get_delta_t());

  double tau_square_hat_proposal = 
    tau_square_proposal / 
    ((1.0 - exp(-2.0*theta_hat_proposal*ou_model_fast_->get_delta_t()))/(2.0*theta_hat_proposal));
  double tau_square_hat_current = ou_model_fast_->get_tau_square().get_continuous_time_parameter();
  double tau_square_current = ou_model_fast_->get_tau_square().
    get_discrete_time_parameter(ou_model_fast_->get_delta_t());

  double rho_tilde_proposal = logit((rho_proposal+1.0)/2.0);
  double theta_tilde_proposal = logit(theta_proposal);
  double tau_square_tilde_proposal = log(tau_square_proposal);

  double log_likelihood_proposal = 
    // ll
    ou_model_fast_->log_likelihood(rho_proposal,
				   theta_proposal,
				   tau_square_proposal)
    // priors
    + ou_model_fast_->get_rho_prior().log_likelihood(rho_proposal)
    + ou_model_fast_->get_theta_prior().log_likelihood(theta_proposal)
    + ou_model_fast_->get_tau_square_prior().log_likelihood(tau_square_proposal)
    // transformation determinant
    + log(2.0) + rho_tilde_proposal -2*log(exp(rho_tilde_proposal) + 1.0)
    + theta_tilde_proposal - 2.0*log(exp(theta_tilde_proposal) + 1.0)
    + tau_square_tilde_proposal;


  double rho_tilde_current = logit((rho_current+1.0)/2.0);
  double theta_tilde_current = logit(theta_current);
  double tau_square_tilde_current = log(tau_square_current);

  double log_likelihood_current = 
    // ll
    ou_model_fast_->log_likelihood(rho_current,
				   theta_current,
				   tau_square_current)
    // priors
    + ou_model_fast_->get_rho_prior().log_likelihood(rho_current)
    + ou_model_fast_->get_theta_prior().log_likelihood(theta_current)
    + ou_model_fast_->get_tau_square_prior().log_likelihood(tau_square_current)
    // transformation determinant
    + log(2.0) + rho_tilde_current -2*log(exp(rho_tilde_current) + 1.0)
    + theta_tilde_current - 2.0*log(exp(theta_tilde_current) + 1.0)
    + tau_square_tilde_current;
  
  // std::cout << rho_current << " "
  // 	    << theta_hat_current << " "
  // 	    << tau_square_hat_current << " " 
  // 	    << std::endl;

  // std::cout << rho_proposal << " "
  // 	    << theta_hat_proposal << " "
  // 	    << tau_square_hat_proposal << " ";
	 

  // std::cout << "; ll_current = " << log_likelihood_current 
  // 	    << "; ll_proposal = " << log_likelihood_proposal;

  double log_a_acceptance = log_likelihood_proposal -
    log_likelihood_current;


  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    //    std::cout << "; move accepted ";
    ou_model_fast_->set_tau_square_hat(tau_square_hat_proposal);
    ou_model_fast_->set_theta_hat(theta_hat_proposal);
    ou_model_fast_->set_rho(rho_proposal);
  } else {
    ou_model_fast_->set_tau_square_hat(tau_square_hat_current);
    ou_model_fast_->set_theta_hat(theta_hat_current);
    ou_model_fast_->set_rho(rho_current);
  }
}

void FastOUPosteriorSampler::
draw_theta_tau_square_rho_MLE()
{
  double rho_current = ou_model_fast_->
    get_rho().get_continuous_time_parameter();
  double rho_tilde_current = log( ( (rho_current+1.0)/2.0 ) /
				  ( 1.0 - (rho_current+1.0)/2.0 ) );

  double theta_current = ou_model_fast_->
    get_theta().get_discrete_time_parameter(ou_model_fast_->get_delta_t());
  double theta_tilde_current = log( theta_current / (1.0 - theta_current) );

  double tau_square_current = ou_model_fast_->
    get_tau_square().get_discrete_time_parameter(ou_model_fast_->get_delta_t());
  double tau_square_tilde_current = 
    log(tau_square_current);

  std::vector<double> tilde_current {rho_tilde_current,
      theta_tilde_current,
      tau_square_tilde_current};

  std::vector<double> mean_tilde = ou_model_fast_->rho_theta_tau_square_tilde_MLE();
  double rho_tilde = mean_tilde[0];
  double theta_tilde = mean_tilde[1];
  double tau_square_tilde = mean_tilde[2];
  double dd = 1e-5;

  double a11 = -1.0 * ou_model_fast_->
    rho_double_deriv_numeric_tilde_scale(rho_tilde,
					 theta_tilde,
					 tau_square_tilde,
					 dd);
  double a12 = -1.0 * ou_model_fast_->
    rho_theta_deriv_numeric_tilde_scale(rho_tilde,
					theta_tilde,
					tau_square_tilde,
					dd,
					dd);

  double a13 = -1.0 * ou_model_fast_->
    rho_tau_square_deriv_numeric_tilde_scale(rho_tilde,
					     theta_tilde,
					     tau_square_tilde,
					     dd,
					     dd);
  double a22 = -1.0 * ou_model_fast_->
    theta_double_deriv_numeric_tilde_scale(rho_tilde,
					   theta_tilde,
					   tau_square_tilde,
					   dd);
  double a23 = -1.0 * ou_model_fast_->
    theta_tau_square_deriv_numeric_tilde_scale(rho_tilde,
					       theta_tilde,
					       tau_square_tilde,
					       dd,
					       dd);
  double a33 = -1.0 * ou_model_fast_->
    tau_square_double_deriv_numeric_tilde_scale(rho_tilde,
						theta_tilde,
						tau_square_tilde,
						dd);
  arma::mat inv_info_mat = { {a11, a12, a13},
			     {a12, a22, a23},
			     {a13, a23, a33} };
  arma::mat info_mat = inv_sympd(inv_info_mat);

  gsl_vector * proposal_tilde_mean_ptr = gsl_vector_alloc(3);
  gsl_matrix * proposal_tilde_covariance_matrix_ptr = gsl_matrix_alloc(3,3);

  for (int i=0; i<3; ++i) {
    gsl_vector_set(proposal_tilde_mean_ptr, i, mean_tilde[i]);
    for (int j=i; j<3; ++j) {
      gsl_matrix_set(proposal_tilde_covariance_matrix_ptr, i,j,
  		     info_mat(i,j));
      if (i != j) {
	gsl_matrix_set(proposal_tilde_covariance_matrix_ptr, j,i, 
		       info_mat(i,j));
      }
    }
  }
  
  RhoThetaTauSquareTildeProposal tilde_proposal_object =
    RhoThetaTauSquareTildeProposal(proposal_tilde_mean_ptr,
				   proposal_tilde_covariance_matrix_ptr,
				   1);

  ThetaTauSquareRhoProposal rho_theta_tau_square_tilde_proposal =
    ThetaTauSquareRhoProposal(proposal_tilde_covariance_matrix_ptr);
  
  std::vector<double> tilde_proposed =
    tilde_proposal_object.propose_parameters(rng_);

  double rho_tilde_proposal = tilde_proposed[0];
  double rho_proposal = 2.0 
    * (exp(rho_tilde_proposal)/(exp(rho_tilde_proposal)+1.0)) - 1.0;

  double theta_tilde_proposal = tilde_proposed[1];
  double theta_proposal = (exp(theta_tilde_proposal)
			   / (exp(theta_tilde_proposal)+1.0));

  double tau_square_tilde_proposal = tilde_proposed[2];
  double tau_square_proposal = exp(tau_square_tilde_proposal);

  double theta_hat_proposal = 
    -1.0*log(theta_proposal) / ou_model_fast_->get_delta_t();

  double theta_hat_current = 
    ou_model_fast_->get_theta().get_continuous_time_parameter();

  double tau_square_hat_proposal = 
    tau_square_proposal / 
    ((1.0 - exp(-2.0*theta_hat_proposal*ou_model_fast_->get_delta_t()))
     / (2.0*theta_hat_proposal));
  double tau_square_hat_current = ou_model_fast_->get_tau_square().
    get_continuous_time_parameter();

  // ou_model_fast_->set_tau_square_hat(tau_square_hat_proposal);
  // ou_model_fast_->set_theta_hat(theta_hat_proposal);
  // ou_model_fast_->set_rho(rho_proposal);

  //  std::cout << "theta_proposal=" << theta_proposal << std::endl;
  double log_likelihood_proposal = 
    ou_model_fast_->log_likelihood_tilde(rho_tilde_proposal,
					 theta_tilde_proposal,
					 tau_square_tilde_proposal) +
    ou_model_fast_->get_rho_prior().log_likelihood(rho_proposal) +
    ou_model_fast_->get_theta_prior().log_likelihood(theta_proposal) + 
    ou_model_fast_->get_tau_square_prior().log_likelihood(tau_square_proposal) + 
    tilde_proposal_object.q_log_likelihood(tilde_current);
  
  // // std::cout << "theta_current=" << theta_current << std::endl;
  // ou_model_fast_->set_tau_square_hat(tau_square_hat_current);
  // ou_model_fast_->set_theta_hat(theta_hat_current);
  // ou_model_fast_->set_rho(rho_current);
  double log_likelihood_current = 
    ou_model_fast_->log_likelihood_tilde(rho_tilde_current,
					 theta_tilde_current,
					 tau_square_tilde_current) +
    ou_model_fast_->get_rho_prior().log_likelihood(rho_current) +
    ou_model_fast_->get_theta_prior().log_likelihood(theta_current) + 
    ou_model_fast_->get_tau_square_prior().log_likelihood(tau_square_current) +
    tilde_proposal_object.q_log_likelihood(tilde_proposed);
  
  // std::cout << rho_current << " "
  // 	    << theta_hat_current << " "
  // 	    << tau_square_hat_current << " " 
  // 	    << std::endl;

  // std::cout << rho_proposal << " "
  // 	    << theta_hat_proposal << " "
  // 	    << tau_square_hat_proposal << " ";
	 

  // std::cout << "; ll_current = " << log_likelihood_current 
  // 	    << "; ll_proposal = " << log_likelihood_proposal;

  double log_a_acceptance = log_likelihood_proposal -
    log_likelihood_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    //    std::cout << "; move accepted ";
    ou_model_fast_->set_tau_square_hat(tau_square_hat_proposal);
    ou_model_fast_->set_theta_hat(theta_hat_proposal);
    ou_model_fast_->set_rho(rho_proposal);
  } else {
    ou_model_fast_->set_tau_square_hat(tau_square_hat_current);
    ou_model_fast_->set_theta_hat(theta_hat_current);
    ou_model_fast_->set_rho(rho_current);
  }

  gsl_vector_free(proposal_tilde_mean_ptr);
  gsl_matrix_free(proposal_tilde_covariance_matrix_ptr);
}

void FastOUPosteriorSampler::draw_theta_hat() 
{
  double prior_mean = 
    ou_model_fast_->get_theta_prior().get_theta_mean();
  double prior_var = 
    square(ou_model_fast_->get_theta_prior().get_theta_std_dev());

  const std::vector<double>& h_fast = 
    ou_model_fast_->get_sigmas().get_discrete_time_log_sigmas();
  const std::vector<double>& h_slow = 
    ou_model_fast_->get_ou_model_slow()->
    get_sigmas().get_discrete_time_log_sigmas();

  const std::vector<int>& ds = 
    ou_model_fast_->get_const_vol_model()->get_ds();
  double alpha = 
    ou_model_fast_->get_alpha().
    get_discrete_time_parameter(ou_model_fast_->get_delta_t());
  double rho = 
    ou_model_fast_->get_rho().get_continuous_time_parameter();
  double tau_square = 
    ou_model_fast_->get_tau_square().
    get_discrete_time_parameter(ou_model_fast_->get_delta_t());
  const std::vector<int>& gammas = 
    ou_model_fast_->get_const_vol_model()->get_gammas().get_gammas();
  
  const std::vector<double>& m = 
    ou_model_fast_->get_const_vol_model()->get_gammas().get_mixture_means();

  const std::vector<double>& as = 
    ou_model_fast_->get_const_vol_model()->get_as();
  const std::vector<double>& bs = 
    ou_model_fast_->get_const_vol_model()->get_bs();
  const std::vector<double>& y_star =
    ou_model_fast_->get_const_vol_model()->get_y_star();

  double likelihood_mean_over_var = 0;
  double likelihood_var_inv = 0;

  for (unsigned i=0; i<ou_model_fast_->data_length(); ++i) {
    double Ft_fast = 
      sqrt(tau_square) 
      * ds[i] * rho * bs[gammas[i]] 
      * exp(m[gammas[i]]/2.0);

    double Ft_slow = Ft_fast;

    double Mt = 
      sqrt(tau_square) 
      * ds[i] * rho * exp(m[gammas[i]]/2.0) 
      * ( as[gammas[i]] + bs[gammas[i]] * 2.0 * (y_star[i] - m[gammas[i]]/2.0));

    // std::cout << "Ft=" << Ft 
    // 	      << "; Mt=" << Mt 
    // 	      << "; ds[i]=" << ds[i] 
    // 	      << "; rho=" << rho
    // 	      << "; alpha=" << alpha
    // 	      << "; h[i]=" << h[i] 
    // 	      << "; h[i+1]=" << h[i+1]
    // 	      << std::endl;

    likelihood_var_inv = likelihood_var_inv + 
      square(h_fast[i] - alpha) / (tau_square*(1-square(rho)));

    likelihood_mean_over_var = likelihood_mean_over_var + 
      (h_fast[i] - alpha) * (h_fast[i+1] - alpha - Mt +
			     h_fast[i]*Ft_fast + 
			     h_slow[i]*Ft_slow) / 
      ( tau_square*(1-square(rho)));
  }
  double likelihood_var = 1.0/likelihood_var_inv;
  double likelihood_mean = likelihood_var * likelihood_mean_over_var;

  double posterior_var = 
    1.0 / (1.0/likelihood_var + 1.0/prior_var);
  double posterior_mean = 
    posterior_var * (likelihood_mean / likelihood_var + prior_mean / prior_var);

  double theta_sample = rtruncnorm(rng_, 0.0, 1.0, 
				   posterior_mean, sqrt(posterior_var));

  // std::cout << "posterior_mean = " << posterior_mean << "\n";
  // std::cout << "posterior_var = " << posterior_var << "\n";
  // std::cout << "theta_sample = " << theta_sample << "\n";
  double theta_hat_sample = 
    -1.0*log(theta_sample) / ou_model_fast_->get_delta_t();
  //  std::cout << "theta_hat_sample = " << theta_hat_sample << "\n";  
  double theta_current = 
    ou_model_fast_->get_theta().
    get_discrete_time_parameter(ou_model_fast_->get_delta_t());

  double theta_hat_current = 
    ou_model_fast_->get_theta().get_continuous_time_parameter();

  ou_model_fast_->set_theta_hat(theta_hat_current);
  double ll_current = ou_model_fast_->log_likelihood();
  double q_proposal_given_current = dtruncnorm(theta_sample,
					       0.0, 1.0,
					       posterior_mean, 
					       sqrt(posterior_var), 1);

  ou_model_fast_->set_theta_hat(theta_hat_sample);
  double ll_proposal = ou_model_fast_->log_likelihood();
  double q_current_given_proposal = dtruncnorm(theta_current,
					       0.0, 1.0,
					       posterior_mean, 
					       sqrt(posterior_var), 1);

  double log_a_theta = 
    (ll_proposal + 
     ou_model_fast_->get_theta_prior().log_likelihood(theta_sample) + 
     q_current_given_proposal)
    -
    (ll_current + 
     ou_model_fast_->get_theta_prior().log_likelihood(theta_current) + 
     q_proposal_given_current);

  // double log_a_theta = 
  //   0.5 * (log(1-square(theta_sample)) - log(1-square(theta_current))) 
  //   - 1.0/(2.0*tau_square) * 
  //   ( (1-square(theta_sample)) - (1-square(theta_current)) ) * 
  //   square(h_fast[0] - alpha);

  if (log(gsl_ran_flat(rng_,0,1)) < log_a_theta) {
    ou_model_fast_->set_theta_hat(theta_hat_sample);
  }

  // std::cout << "theta_hat FAST = " 
  // 	    << ou_model_fast_->get_theta().get_continuous_time_parameter()
  // 	    << std::endl;
}

void FastOUPosteriorSampler::draw_theta_hat_norm() 
{
  double prior_mean = 
    ou_model_fast_->get_theta_prior().get_theta_mean();
  double prior_var = 
    square(ou_model_fast_->get_theta_prior().get_theta_std_dev());

  const std::vector<double>& h_fast = 
    ou_model_fast_->get_sigmas().get_discrete_time_log_sigmas();
  const std::vector<double>& h_slow = 
    ou_model_fast_->get_ou_model_slow()->
    get_sigmas().get_discrete_time_log_sigmas();

  const std::vector<int>& ds = 
    ou_model_fast_->get_const_vol_model()->get_ds();
  double alpha = 
    ou_model_fast_->get_alpha().
    get_discrete_time_parameter(ou_model_fast_->get_delta_t());
  double rho = 
    ou_model_fast_->get_rho().get_continuous_time_parameter();
  double tau_square = 
    ou_model_fast_->get_tau_square().
    get_discrete_time_parameter(ou_model_fast_->get_delta_t());
  const std::vector<int>& gammas = 
    ou_model_fast_->get_const_vol_model()->get_gammas().get_gammas();
  
  const std::vector<double>& m = 
    ou_model_fast_->get_const_vol_model()->get_gammas().get_mixture_means();

  const std::vector<double>& as = 
    ou_model_fast_->get_const_vol_model()->get_as();
  const std::vector<double>& bs = 
    ou_model_fast_->get_const_vol_model()->get_bs();
  const std::vector<double>& y_star =
    ou_model_fast_->get_const_vol_model()->get_y_star();

  double mean_over_variance = 0;
  double variance_inv = 0;

  for (unsigned i=0; i<ou_model_fast_->data_length(); ++i) {
    double Ft_fast = sqrt(tau_square)*
      ds[i] * rho * bs[gammas[i]] * exp(m[gammas[i]]/2.0);
    double Ft_slow = Ft_fast;
    double Mt = sqrt(tau_square) * ds[i] * rho * exp(m[gammas[i]]/2.0) * 
      ( as[gammas[i]] + bs[gammas[i]] * 2.0 * (y_star[i] - m[gammas[i]]/2.0));

    // std::cout << "Ft=" << Ft 
    // 	      << "; Mt=" << Mt 
    // 	      << "; ds[i]=" << ds[i] 
    // 	      << "; rho=" << rho
    // 	      << "; alpha=" << alpha
    // 	      << "; h[i]=" << h[i] 
    // 	      << "; h[i+1]=" << h[i+1]
    // 	      << std::endl;

    // B = B + 
    //   square(-1.0*h_fast[i] + alpha);
    // b = b + 
    //   -1.0 * ( (-1.0*h_fast[i] + alpha) * 
    // 	       (h_fast[i+1] - alpha - Mt +
    // 		Ft_slow*h_slow[i] + Ft_fast*h_fast[i]) );

    mean_over_variance = mean_over_variance + 
      -1.0 * ( (-1.0*h_fast[i] + alpha) * 
	       (h_fast[i+1] - alpha - Mt +
     		Ft_slow*h_slow[i] + Ft_fast*h_fast[i]) ) / 
      (tau_square * (1-square(rho)));

    variance_inv = variance_inv + 
      square(-1.0*h_fast[i] + alpha) / (tau_square * (1-square(rho)));
  }

  double likelihood_var = 1.0/variance_inv;
  double likelihood_mean = likelihood_var * mean_over_variance;

  double posterior_var = 
    1.0 / (1.0/likelihood_var + 1.0/prior_var);
  double posterior_mean = 
    posterior_var * (likelihood_mean / likelihood_var + prior_mean / prior_var);

  // std::cout << "posterior_mean = " << posterior_mean << "\n";
  // std::cout << "posterior_var = " << posterior_var << "\n";

  double theta_sample = rtruncnorm(rng_, 0.0, 1.0, 
				   posterior_mean, sqrt(posterior_var));
  // std::cout << "theta_sample = " << theta_sample << "\n";

  double theta_hat_sample = 
    -1.0*log(theta_sample) / ou_model_fast_->get_delta_t();
  //  std::cout << "theta_hat_sample FAST = " << theta_hat_sample << "\n";  
  double theta_current = 
    ou_model_fast_->get_theta().
    get_discrete_time_parameter(ou_model_fast_->get_delta_t());

  double theta_hat_current = 
    ou_model_fast_->get_theta().get_continuous_time_parameter();

  double ll_proposal = ou_model_fast_->log_likelihood(rho,
						      theta_sample,
						      tau_square);
  double q_current_given_proposal = dtruncnorm(theta_current,
					       0.0, 1.0,
					       posterior_mean, sqrt(posterior_var), 1);

  double ll_current = ou_model_fast_->log_likelihood(rho,
						     theta_current,
						     tau_square);
  double q_proposal_given_current = dtruncnorm(theta_sample,
					       0.0, 1.0,
					       posterior_mean, sqrt(posterior_var), 1);

  double log_a_theta = 
    (ll_proposal + 
     ou_model_fast_->get_theta_prior().log_likelihood(theta_sample) + 
     q_current_given_proposal)
    -
    (ll_current + 
     ou_model_fast_->get_theta_prior().log_likelihood(theta_current) + 
     q_proposal_given_current);

  // double log_a_theta = 
  //   0.5 * (log(1-square(theta_sample)) - log(1-square(theta_current))) 
  //   - 1.0/(2.0*tau_square) * 
  //   ( (1-square(theta_sample)) - (1-square(theta_current)) ) * 
  //   square(h_fast[0] - alpha);

  if (log(gsl_ran_flat(rng_,0,1)) < log_a_theta) {
    ou_model_fast_->set_theta_hat(theta_hat_sample);
  } else {
    ou_model_fast_->set_theta_hat(theta_hat_current);
  }

  // std::cout << "theta_hat FAST = " 
  // 	    << ou_model_fast_->get_theta().get_continuous_time_parameter()
  // 	    << std::endl;
}

void FastOUPosteriorSampler::draw_tau_square_hat()
{
  std::vector<double> proposal_alpha_beta_current = 
    ou_model_fast_->tau_square_posterior_shape_rate();

  double proposal_alpha = proposal_alpha_beta_current[0];
  double proposal_beta = proposal_alpha_beta_current[1];

  double tau_square_proposal_inv = gsl_ran_gamma(rng_, 
						 proposal_alpha,
						 1.0/proposal_beta);

  double tau_square_proposal = 1.0/tau_square_proposal_inv;

  double theta_hat = ou_model_fast_->
    get_theta().get_continuous_time_parameter();

  double tau_square_hat_proposal = 
    tau_square_proposal / 
    ((1.0 - exp(-2.0*theta_hat*ou_model_fast_->get_delta_t()))/(2.0*theta_hat));

  double tau_square_current = ou_model_fast_->
    get_tau_square().get_discrete_time_parameter(ou_model_fast_->get_delta_t());

  double tau_square_hat_current = 
    ou_model_fast_->
    get_tau_square().get_continuous_time_parameter();

  ou_model_fast_->set_tau_square_hat(tau_square_hat_proposal);
  double log_likelihood_proposal = ou_model_fast_->log_likelihood();

  std::vector<double> proposal_alpha_beta_proposal = ou_model_fast_->
    tau_square_posterior_shape_rate();
  double current_alpha = proposal_alpha_beta_proposal[0];
  double current_beta = proposal_alpha_beta_proposal[1];

  ou_model_fast_->set_tau_square_hat(tau_square_hat_current);
  double log_likelihood_current = ou_model_fast_->log_likelihood();

  double q_current_given_proposal = 
    dgamma(1.0/tau_square_current, current_alpha, 1.0/current_beta, 1);

  double q_proposal_given_current = 
    dgamma(tau_square_proposal_inv, proposal_alpha, 1.0/proposal_beta, 1);
  
  double log_a_acceptance = 
    (log_likelihood_proposal
     + ou_model_fast_->get_tau_square_prior().log_likelihood(tau_square_proposal) 
     + q_current_given_proposal)
    -
    (log_likelihood_current
     + ou_model_fast_->get_tau_square_prior().log_likelihood(tau_square_current)
     + q_proposal_given_current);
  
  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    ou_model_fast_->set_tau_square_hat(tau_square_hat_proposal);
  } else {
    ou_model_fast_->set_tau_square_hat(tau_square_hat_current);
  }

  // std::cout << "tau_sq_hat FAST = " 
  // 	    << ou_model_fast_->get_tau_square().get_continuous_time_parameter()
  // 	    << "\n";
  // std::cout << "theta_hat FAST = " 
  // 	    << ou_model_fast_->get_theta().get_continuous_time_parameter()
  // 	    << "\n";
  // std::cout << "tau_sq FAST = " 
  // 	    << ou_model_fast_->
  //   get_tau_square().get_discrete_time_parameter(ou_model_fast_->get_delta_t())
  // 	    << "\n";

}

void FastOUPosteriorSampler::draw_tau_square_hat_norm_MLE()
{
  std::vector<double> proposal_mean_var_tilde = 
    ou_model_fast_->tau_square_MLE_mean_variance_tilde_scale();

  double proposal_mean_tilde = proposal_mean_var_tilde[0];
  double proposal_var_tilde = proposal_mean_var_tilde[1];

  double tau_square_tilde_proposal = 
    proposal_mean_tilde + 
    sqrt(proposal_var_tilde)*gsl_ran_gaussian(rng_,1.0);

  double tau_square_current = ou_model_fast_->
    get_tau_square().get_discrete_time_parameter(ou_model_fast_->get_delta_t());
  double tau_square_tilde_current = log(tau_square_current);

  double log_likelihood_proposal = 
    ou_model_fast_->log_likelihood_tau_square(exp(tau_square_tilde_proposal));

  double log_likelihood_current = 
    ou_model_fast_->log_likelihood_tau_square(tau_square_current);

  double q_current_given_proposal = 
    dnorm(tau_square_tilde_current,
	  proposal_mean_tilde,
	  sqrt(proposal_var_tilde),
	  1);

  double q_proposal_given_current = 
    dnorm(tau_square_tilde_proposal,
	  proposal_mean_tilde,
	  sqrt(proposal_var_tilde),
	  1);
  
  double log_a_acceptance = 
    (log_likelihood_proposal
     + ou_model_fast_->get_tau_square_prior().
     log_likelihood(exp(tau_square_tilde_proposal))
     + tau_square_tilde_proposal
     + q_current_given_proposal)
    -
    (log_likelihood_current
     + ou_model_fast_->get_tau_square_prior().
     log_likelihood(tau_square_current)
     + tau_square_tilde_current +
     + q_proposal_given_current);
  
  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    double theta_hat = ou_model_fast_->get_theta().get_continuous_time_parameter();
    double tau_square_hat_proposal = exp(tau_square_tilde_proposal) / 
      ((1 - exp(-2*theta_hat*ou_model_fast_->get_delta_t()))/(2*theta_hat));
    ou_model_fast_->set_tau_square_hat(tau_square_hat_proposal);
  }

  // std::cout << "tau_sq_hat FAST = " 
  // 	    << ou_model_fast_->get_tau_square().get_continuous_time_parameter()
  // 	    << "\n";
  // std::cout << "theta_hat FAST = " 
  // 	    << ou_model_fast_->get_theta().get_continuous_time_parameter()
  // 	    << "\n";
  // std::cout << "tau_sq FAST = " 
  // 	    << ou_model_fast_->
  //   get_tau_square().get_discrete_time_parameter(ou_model_fast_->get_delta_t())
  // 	    << "\n";

}

void FastOUPosteriorSampler::draw_rho_norm() 
{
  double rho_current = ou_model_fast_->get_rho().get_continuous_time_parameter();
  double rho_tilde_current = logit((rho_current + 1.0)/2.0);

  std::vector<double> rho_mean_var = 
    ou_model_fast_-> rho_posterior_mean_var();

  double rho_mean = rho_mean_var[0];
  double rho_var = rho_mean_var[1];

  double rho_tilde_mean = logit((rho_mean+1.0)/2.0);
  double rho_tilde_var =
    (-2.0/(square(rho_mean)-1.0) * rho_var);

  double rho_tilde_proposal = 
    rho_tilde_mean + sqrt(rho_tilde_var)*gsl_ran_gaussian(rng_, 1.0);
  double rho_proposal = 
    2 * exp(rho_tilde_proposal) / (exp(rho_tilde_proposal)+1) - 1;

  double q_proposal_given_current = dnorm(rho_tilde_proposal, 
					  rho_tilde_mean,
					  sqrt(rho_tilde_var), 1);

  double q_current_given_proposal = dnorm(rho_tilde_current, 
					  rho_tilde_mean,
					  sqrt(rho_tilde_var), 1);

  ou_model_fast_->set_rho(rho_current);
  double ll_current = 
    ou_model_fast_->get_const_vol_model()->log_likelihood() + 
    ou_model_fast_->log_likelihood() + 
    ou_model_fast_->get_rho_prior().log_likelihood(rho_current) +
    square(rho_tilde_current) - 2.0*log(exp(rho_tilde_current)+1) + 
    q_proposal_given_current;

  ou_model_fast_->set_rho(rho_proposal);
  double ll_proposal = 
    ou_model_fast_->get_const_vol_model()->log_likelihood() + 
    ou_model_fast_->log_likelihood() + 
    ou_model_fast_->get_rho_prior().log_likelihood(rho_proposal) +
    square(rho_tilde_proposal) - 2.0*log(exp(rho_tilde_proposal)+1) + 
    q_current_given_proposal;

  double log_a_acceptance = 
    ll_proposal - ll_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    // std::cout << "ll_proposal = " << ll_proposal << "; ll_current = " 
    // 	      << ll_current << "\n";
    // std::cout << "; ACCEPTED RHO =" << rho_proposal << "\n";
    ou_model_fast_->set_rho(rho_proposal);
  } else {
    // std::cout << "ll_proposal = " << ll_proposal << "; ll_current = " 
    // 	      << ll_current << "\n";
    // std::cout << ";UNACCEPTED RHO=" << rho_current << "\n";
    ou_model_fast_->set_rho(rho_current);
  }  
}

void FastOUPosteriorSampler::draw_rho_tnorm() 
{
  double rho_current = ou_model_fast_->get_rho().
    get_discrete_time_parameter(ou_model_fast_->get_delta_t());

  std::vector<double> rho_mean_var = 
    ou_model_fast_-> rho_posterior_mean_var();

  double rho_mean = rho_mean_var[0];
  double rho_var = rho_mean_var[1];

  double rho_proposal = 
    rtruncnorm(rng_, -1.0, 1.0, rho_mean, sqrt(rho_var));

  double q_proposal_given_current = dtruncnorm(rho_proposal, -1.0, 1.0,
					       rho_mean, sqrt(rho_var),
					       true);

  double q_current_given_proposal = dtruncnorm(rho_current, -1.0, 1.0,
					       rho_mean, sqrt(rho_var),
					       true);

  double ll_current = 
    ou_model_fast_->log_likelihood_rho(rho_current) + 
    ou_model_fast_->get_rho_prior().log_likelihood(rho_current) +
    q_proposal_given_current;

  double ll_proposal = 
    ou_model_fast_->log_likelihood_rho(rho_proposal) + 
    ou_model_fast_->get_rho_prior().log_likelihood(rho_proposal) +
    q_current_given_proposal;

  double log_a_acceptance = 
    ll_proposal - ll_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    // std::cout << "ll_proposal = " << ll_proposal << "; ll_current = " 
    // 	      << ll_current << "\n";
    // std::cout << "; ACCEPTED RHO =" << rho_proposal << "\n";
    ou_model_fast_->set_rho(rho_proposal);
  } else {
    // std::cout << "ll_proposal = " << ll_proposal << "; ll_current = " 
    // 	      << ll_current << "\n";
    // std::cout << ";UNACCEPTED RHO=" << rho_current << "\n";
    ou_model_fast_->set_rho(rho_current);
  }  
}

void FastOUPosteriorSampler::draw_rho_student() 
{
  double nu = 1.0;
  double rho_current = ou_model_fast_->get_rho().get_continuous_time_parameter();
  double rho_tilde_current = logit((rho_current + 1.0)/2.0);

  double theta_current =  ou_model_fast_->
    get_theta().get_discrete_time_parameter(ou_model_fast_->get_delta_t());

  double tau_square_current = ou_model_fast_->
    get_tau_square().get_discrete_time_parameter(ou_model_fast_->get_delta_t());

  std::vector<double> rho_mean_var = 
    ou_model_fast_->rho_posterior_mean_var();

  double rho_mean = rho_mean_var[0];
  double rho_var = rho_mean_var[1];

  double tilde_first_moment = logit((rho_mean+1.0)/2.0) 
    + 4*rho_mean/square(square(rho_mean) - 1.0)*rho_var;

  double tilde_second_moment = 
    square(logit((rho_mean+1.0)/2.0))
    + 8*(rho_mean*log(-(rho_mean+1)/(rho_mean-1)) + 1)/
    square(square(rho_mean)-1.0) * rho_var;

  double rho_tilde_mean = tilde_first_moment;
  double rho_tilde_var = tilde_second_moment - square(tilde_first_moment);

  // std::cout << "rho_mean = " << rho_mean << "\n";
  // std::cout << "rho_var = " << rho_var << "\n";
  // std::cout << "tilde_first_moment = " << tilde_first_moment << "\n";
  // std::cout << "tilde_second_moment = " << tilde_second_moment << "\n";
  // std::cout << "rho_tilde_mean = " << rho_tilde_mean << "\n";
  // std::cout << "rho_tilde_var = " << rho_tilde_var << "\n";

  // std::cout << "rho_mean = " << rho_mean << "\n";
  // std::cout << "rho_var = " << rho_var << "\n";

  // double rho_tilde_mean = logit((rho_mean+1.0)/2.0);
  // double rho_tilde_var =
  //   (-2.0/(square(rho_mean)-1.0) * rho_var);

  // std::vector<double> rho_mean_var = 
  //   ou_model_fast_-> rho_MLE_mean_var_tilde();

  double rho_tilde_proposal =
    rho_tilde_mean + sqrt(rho_tilde_var)*gsl_ran_tdist(rng_, nu);
  double rho_proposal =
    2 * exp(rho_tilde_proposal) / (exp(rho_tilde_proposal)+1) - 1;

  double q_proposal_given_current =
    log(gsl_ran_tdist_pdf((rho_tilde_proposal - rho_tilde_mean)/sqrt(rho_tilde_var), 
			  nu));

  double q_current_given_proposal = 
    log(gsl_ran_tdist_pdf((rho_tilde_current - rho_tilde_mean)/sqrt(rho_tilde_var), 
			  nu));

  double ll_current = 
    ou_model_fast_->log_likelihood(rho_current,
				   theta_current,
				   tau_square_current) + 
    ou_model_fast_->get_rho_prior().log_likelihood(rho_current)
    + log(2.0) + rho_tilde_current - 2.0*log(exp(rho_tilde_current)+1.0)
    + q_proposal_given_current;

  double ll_proposal = 
    ou_model_fast_->log_likelihood(rho_proposal,
				   theta_current,
				   tau_square_current) + 
    ou_model_fast_->get_rho_prior().log_likelihood(rho_proposal)
    + log(2.0) + rho_tilde_proposal - 2.0*log(exp(rho_tilde_proposal)+1.0)
    + q_current_given_proposal;

  double log_a_acceptance = 
    ll_proposal - ll_current;

  // std::cout << "ll_proposal = " << ll_proposal 
  // 	    << "; ll_current = " << ll_current 
  // 	    << "; q_current_given_proposal = " << q_current_given_proposal
  // 	    << "; q_proposal_given_current = " << q_proposal_given_current
  // 	    << "\n";
  
  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    // std::cout << "; ACCEPTED RHO =" << rho_proposal << "\n";
    ou_model_fast_->set_rho(rho_proposal);
  } else {
    // std::cout << ";UNACCEPTED RHO=" << rho_proposal << "\n";
    ou_model_fast_->set_rho(rho_current);
  }
}

void FastOUPosteriorSampler::set_theta_lower_bound(double lb)
{
  theta_tau_square_rho_proposal_.set_theta_lower_bound(lb);
}
void FastOUPosteriorSampler::set_theta_upper_bound(double ub)
{
  theta_tau_square_rho_proposal_.set_theta_upper_bound(ub);
}

// ============== SV POSTERIOR SAMPLER ================================

StochasticVolatilityPosteriorSampler::
StochasticVolatilityPosteriorSampler(StochasticVolatilityModel* sv_model,
				     gsl_rng * rng,
				     const gsl_matrix * proposal_covariance_matrix)
  : sv_model_(sv_model),
    rng_(rng),
    observational_model_sampler_(ObservationalPosteriorSampler(sv_model_->
							       get_observational_model(),
							       rng)),
    constant_vol_sampler_(ConstantVolatilityPosteriorSampler(sv_model_->get_const_vol_model(),
							  rng)),
    ou_sampler_(OUPosteriorSampler(sv_model_->get_ou_model(),
				   rng,
				   proposal_covariance_matrix))
{}

void StochasticVolatilityPosteriorSampler::draw_gammas_gsl()
{
  sv_model_->
    get_const_vol_model()->
    set_y_star_ds();
  const std::vector<double>& h = 
    sv_model_->
    get_ou_model()->
    get_sigmas().get_discrete_time_log_sigmas();
  const std::vector<double>& y_star = 
    sv_model_->
    get_const_vol_model()->
    get_y_star();
  double tau_square = 
    sv_model_->
    get_ou_model()->
    get_tau_square().get_discrete_time_parameter(sv_model_->
						 get_delta_t());
  double rho = 
    sv_model_->
    get_ou_model()->
    get_rho().get_continuous_time_parameter();

  const std::vector<double>& m = 
    sv_model_->
    get_const_vol_model()->
    get_gammas().get_mixture_means();
  const std::vector<double>& v_square = 
    sv_model_->
    get_const_vol_model()->
    get_gammas().get_mixture_variances();
  const std::vector<double>& probs = 
    sv_model_->
    get_const_vol_model()->
    get_gammas().get_mixture_probabilities();

  std::vector<int> new_gamma (sv_model_->data_length());
  int J = sv_model_->get_const_vol_model()->get_gammas().get_J();

  double posterior_mixture_component_probability[J];    
  for (unsigned i=0; i<sv_model_->data_length(); ++i) {

    for (int j=0; j<J; ++j) {
      double element = 
	-1.0*log(sqrt(v_square[j])/2.0) +
	-1.0/(2.0*v_square[j]/4.0) * square(y_star[i]-h[i]-m[j]/2.0) + 
	-1.0/(2.0*tau_square*(1.0-square(rho))) * 
	square(h[i+1] - 
	       sv_model_->get_ou_model()->theta_j(i,j)*h[i] - 
	       sv_model_->get_ou_model()->alpha_j(i,j)) +
	log(probs[j]);

      posterior_mixture_component_probability[j] = exp(element);
    }
    
    double *P;
    P = posterior_mixture_component_probability;

    gsl_ran_discrete_t * g = 
      gsl_ran_discrete_preproc(J, P);
    int j = gsl_ran_discrete (rng_, g);
    new_gamma[i] = j;
    gsl_ran_discrete_free(g);
  }
  sv_model_->get_const_vol_model()->set_gammas(new_gamma);
}

void StochasticVolatilityPosteriorSampler::draw_gammas()
{
  sv_model_->
    get_const_vol_model()->
    set_y_star_ds();
  const std::vector<double>& h = 
    sv_model_->
    get_ou_model()->
    get_sigmas().get_discrete_time_log_sigmas();
  const std::vector<double>& y_star = 
    sv_model_->
    get_const_vol_model()->
    get_y_star();
  double tau_square = 
    sv_model_->
    get_ou_model()->
    get_tau_square().get_discrete_time_parameter(sv_model_->
						 get_delta_t());
  double rho = 
    sv_model_->
    get_ou_model()->
    get_rho().get_continuous_time_parameter();

  const std::vector<double>& m = 
    sv_model_->
    get_const_vol_model()->
    get_gammas().get_mixture_means();
  const std::vector<double>& v_square = 
    sv_model_->
    get_const_vol_model()->
    get_gammas().get_mixture_variances();
  const std::vector<double>& probs = 
    sv_model_->
    get_const_vol_model()->
    get_gammas().get_mixture_probabilities();

  std::vector<int> new_gamma (sv_model_->data_length());
  std::vector<double> 
    log_posterior_mixture_component_probability(sv_model_->
						get_const_vol_model()->
						get_gammas().get_J());
  std::vector<double> 
    normalized_posterior_mixture_component_probability(sv_model_->
						       get_const_vol_model()->
						       get_gammas().get_J());
  int J = sv_model_->get_const_vol_model()->get_gammas().get_J();
  for (unsigned i=0; i<sv_model_->data_length(); ++i) {
    double max_element = 0.0;
    
    for (int j=0; j<J; ++j) {
      double element = 
	-1.0*log(sqrt(v_square[j])/2.0) +
	-1.0/(2.0*v_square[j]/4.0) * square(y_star[i]-h[i]-m[j]/2.0) + 
	-1.0/(2.0*tau_square*(1.0-square(rho))) * 
	square(h[i+1] - 
	       sv_model_->get_ou_model()->theta_j(i,j)*h[i] - 
	       sv_model_->get_ou_model()->alpha_j(i,j)) +
	log(probs[j]);
      if (j==0) {
	max_element = element;
      } else {
	if (element >= max_element) {
	  max_element = element;
	}
      }
      log_posterior_mixture_component_probability[j] = element;
    }
    
    for (int j=0; j<J; ++j) {
      normalized_posterior_mixture_component_probability[j] =
	exp(log_posterior_mixture_component_probability[j] - max_element);
    }

    // std::discrete_distribution<int> 
    //   component_sampler(normalized_posterior_mixture_component_probability.begin(),
    //  			normalized_posterior_mixture_component_probability.end());
    
    double sum = 0;
    for (unsigned j=0; 
    	 j<normalized_posterior_mixture_component_probability.size();
    	 ++j) {
      sum = sum + normalized_posterior_mixture_component_probability[j];
    }

    unsigned j=0;    
    double running_sum = normalized_posterior_mixture_component_probability[j];
    double component_index = gsl_ran_flat(rng_, 0, sum);
    while (running_sum < component_index) {
      j=j+1;
      running_sum = running_sum + 
    	normalized_posterior_mixture_component_probability[j];
    }

    if (j > normalized_posterior_mixture_component_probability.size()-1) {
      std::cout << sum << " " << j << " " << component_index << std::endl;
    }

    new_gamma[i] = j;
    // new_gamma[i] = component_sampler(mt_);
  }
  sv_model_->get_const_vol_model()->set_gammas(new_gamma);
}

// void StochasticVolatilityPosteriorSampler::draw_tau_square()
// {
//   double tau_square_alpha = 
//     model_->get_tau_square_prior().get_tau_square_shape();
//   double tau_square_beta = 
//     model_->get_tau_square_prior().get_tau_square_scale();

//   double rho = 
//     model_->get_rho().get_continuous_time_parameter();
//   double alpha = 
//     model_->get_alpha().get_discrete_time_parameter(model_->get_delta_t());
//   double theta = 
//     model_->get_theta().get_discrete_time_parameter(model_->get_delta_t());
//   const std::vector<double> h = 
//     model_->get_sigmas().get_discrete_time_log_sigmas();

//   double proposal_alpha = tau_square_alpha + (model_->data_length())/2.0;
//   double proposal_beta  = tau_square_beta;

//   for (unsigned i=0; i<model_->data_length(); ++i) {
//     proposal_beta = proposal_beta + 
//       0.5/(1-square(rho))*square(h[i+1] - theta*h[i] - alpha*(1-theta));
//   }

//   proposal_beta = proposal_beta + 
//     (1-square(theta))/2.0 * square(h[0]-alpha);

//   double tau_square_proposal_inv = gsl_ran_gamma(rng_, 
// 						 proposal_alpha,
// 						 1.0/proposal_beta);

//   double tau_square_proposal = 1.0/tau_square_proposal_inv;
//   double tau_square_current = 
//     model_->get_tau_square().get_discrete_time_parameter(model_->get_delta_t());

//   double theta_hat = model_->get_theta().get_continuous_time_parameter();

//   double tau_square_hat_proposal = 
//     tau_square_proposal / 
//     ((1.0 - exp(-2.0*theta_hat*model_->get_delta_t()))/(2.0*theta_hat));
//   double tau_square_hat_current = 
//     tau_square_current / 
//     ((1.0 - exp(-2.0*theta_hat*model_->get_delta_t()))/(2.0*theta_hat));

//   model_->set_tau_square_hat(tau_square_hat_proposal);
//   double log_likelihood_proposal = model_->log_likelihood();

//   model_->set_tau_square_hat(tau_square_hat_current);
//   double log_likelihood_current = model_->log_likelihood();

//   double q_current_given_proposal = 
//     dgamma(1.0/tau_square_current, proposal_alpha, 1.0/proposal_beta, 1);

//   double q_proposal_given_current = 
//     dgamma(tau_square_proposal_inv, proposal_alpha, 1.0/proposal_beta, 1);
  
//   double log_a_acceptance = 
//     (log_likelihood_proposal 
//      + model_->get_tau_square_prior().log_likelihood(tau_square_proposal) 
//      + q_current_given_proposal)
//     -
//     (log_likelihood_current
//      + model_->get_tau_square_prior().log_likelihood(tau_square_current)
//      + q_proposal_given_current);
  
//   if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
//     std::cout << "tau sq accepted ";
//     model_->set_tau_square_hat(tau_square_hat_proposal);
//   }

//   std::cout << "tau_sq_hat = " 
//   	    << model_->get_tau_square().get_continuous_time_parameter()
//   	    << std::endl;
// }

// void StochasticVolatilityPosteriorSampler::draw_theta_hat()
// {
//   double prior_mean = 
//     model_->get_theta_prior().get_theta_mean();
//   double prior_var = 
//     square(model_->get_theta_prior().get_theta_std_dev());

//   const std::vector<double>& h = 
//     model_->get_sigmas().get_discrete_time_log_sigmas();
//   const std::vector<int> ds = 
//     model_->get_ds();
//   double alpha = 
//     model_->get_alpha().get_discrete_time_parameter(model_->get_delta_t());
//   double rho = 
//     model_->get_rho().get_continuous_time_parameter();
//   double tau_square = 
//     model_->get_tau_square().get_discrete_time_parameter(model_->get_delta_t());
//   const std::vector<int> gammas = 
//     model_->get_gammas().get_gammas();
  
//   const std::vector<double>& m = 
//     model_->get_gammas().get_mixture_means();

//   const std::vector<double>& as = 
//     model_->get_as();
//   const std::vector<double>& bs = 
//     model_->get_bs();
//   const std::vector<double>& y_star =
//     model_->get_y_star();

//   double B=0;
//   double b=0;

//   for (unsigned i=0; i<model_->data_length(); ++i) {
//     double Ft = ds[i] * rho * 2.0 * bs[gammas[i]] * exp(m[gammas[i]]/2.0);
//     double Mt = ds[i] * rho * exp(m[gammas[i]]/2.0) * 
//       ( as[gammas[i]] + bs[gammas[i]] * 2.0 * (y_star[i] - m[gammas[i]]/2.0));

//     // std::cout << "Ft=" << Ft 
//     // 	      << "; Mt=" << Mt 
//     // 	      << "; ds[i]=" << ds[i] 
//     // 	      << "; rho=" << rho
//     // 	      << "; alpha=" << alpha
//     // 	      << "; h[i]=" << h[i] 
//     // 	      << "; h[i+1]=" << h[i+1]
//     // 	      << std::endl;

//     B = B + 
//       square(-1.0*h[i] + alpha);
//     b = b + 
//       -1.0 * ( (-1.0*h[i] + alpha) * 
// 	       (h[i+1] + Ft*h[i] - alpha - Mt) );
//   }
  
//   double likelihood_mean = b / B;
//   double likelihood_var = tau_square*(1-square(rho)) / B;

//   double posterior_var = 
//     1.0 / (1.0/likelihood_var + 1.0/prior_var);
//   double posterior_mean = 
//     posterior_var * (likelihood_mean / likelihood_var + prior_mean / prior_var);

//   if (B < 1e-16) {
//     std::cout << "Less than EPS" << std::endl;
//     posterior_var = prior_var;
//     posterior_mean = prior_mean;
//   }
  
//   double theta_sample = 
//     posterior_mean + 
//     sqrt(posterior_var) *
//     qnorm(runif(pnorm(0.0,posterior_mean,sqrt(posterior_var),1,0),
// 		pnorm(1.0,posterior_mean,sqrt(posterior_var),1,0)),
// 	  0.0, 1.0, 1, 0);

//   double theta_hat_sample = 
//     -1.0*log(theta_sample) / model_->get_delta_t();
  
//   double theta_current = 
//     model_->get_theta().get_discrete_time_parameter(model_->get_delta_t());

//   // std::cout << "theta_hat_current = " 
//   // 	    << model_->get_theta().get_continuous_time_parameter()
//   // 	    << std::endl;

//   double log_a_theta = 
//     0.5 * (log(1-square(theta_sample)) - log(1-square(theta_current))) 
//     - 1.0/(2.0*tau_square) * 
//     ( (1-square(theta_sample)) - (1-square(theta_current)) ) * 
//     square(h[0] - alpha);

//   if (log(runif(0.0,1.0)) < log_a_theta) {
//     model_->set_theta_hat(theta_hat_sample);
//   }

//   // std::cout << "theta_hat = " 
//   // 	    << model_->get_theta().get_continuous_time_parameter()
//   // 	    << std::endl;
// }

// void StochasticVolatilityPosteriorSampler::draw_mu_hat()
// {
//   // we need the close for each period, and the opening for the frist period
//   double variance_inv = 0;
//   double mean_over_variance = 0;
//   const std::vector<SigmaSingletonParameter>& sigmas = 
//     model_->get_sigmas().get_sigmas();

//   for (unsigned i=0; i<model_->data_length(); ++i) {
//     double S_j_minus_1 = model_->get_data_element(i).get_open();
//     double S_j = model_->get_data_element(i).get_close();

//     if (i>0) {
//       S_j_minus_1 = model_->get_data_element(i-1).get_close();
//       S_j = model_->get_data_element(i).get_close();
//     }

//     double sigma_j = 
//       sigmas[i+1].get_discrete_time_parameter(model_->get_delta_t());
    
//     variance_inv = variance_inv + 1.0 / (sigma_j*sigma_j);
//     mean_over_variance = mean_over_variance + 
//       (log(S_j) - log(S_j_minus_1)) / (sigma_j*sigma_j);
//   }

//   double likelihood_variance = 1.0 / variance_inv;
//   double likelihood_mean = mean_over_variance * likelihood_variance;

//   double prior_mean = model_->get_mu_prior().get_mu_mean();
//   double prior_std_dev = model_->get_mu_prior().get_mu_std_dev();
//   double prior_var = prior_std_dev * prior_std_dev;

//   double var = 1.0 / (1.0/likelihood_variance + 1.0/prior_var);
//   double mean = (prior_mean / prior_var + likelihood_mean / likelihood_variance) * var;

//   double mu_sample = mean + sqrt(var)*gsl_ran_gaussian(rng_, 1.0);
//   double mu_hat_sample = mu_sample / model_->get_delta_t();
//   model_->set_mu_hat(mu_hat_sample);
//   model_->set_y_star_ds();
// }

void StochasticVolatilityPosteriorSampler::draw_sigmas()
{
  std::vector<double> taus_squared (sv_model_->data_length());
  std::vector<double> vs (sv_model_->data_length());

  const std::vector<double>& m = 
    sv_model_->get_const_vol_model()->get_gammas().get_mixture_means();
  const std::vector<double>& v_square = 
    sv_model_->get_const_vol_model()->get_gammas().get_mixture_variances();

  const std::vector<int>& gammas = 
    sv_model_->get_const_vol_model()->get_gammas().get_gammas();
  const std::vector<double>& y_star = sv_model_->get_const_vol_model()->get_y_star();
  double tau_square = 
    sv_model_->get_ou_model()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());
  double rho = 
    sv_model_->get_ou_model()->get_rho().get_continuous_time_parameter();

  // FORWARD FILTER
  // current mean level;
  double v_current = 
    sv_model_->get_ou_model()->
    get_alpha().get_discrete_time_parameter(sv_model_->get_delta_t());
  // current variance level;
  double tau_current_sq = 
    tau_square / 
    (1 - square(sv_model_->get_ou_model()->
		get_theta().get_discrete_time_parameter(sv_model_->get_delta_t())));

  vs[0] = v_current;
  taus_squared[0] = tau_current_sq;

  for (unsigned i=1; i<sv_model_->data_length(); ++i) {

    double theta_j_minus_one = sv_model_->get_ou_model()->
      theta_j(i-1,gammas[i-1]);
    double alpha_j_minus_one = sv_model_->get_ou_model()->
      alpha_j(i-1,gammas[i-1]);

    // s^2_{i+1} = s^2_{j}; j goes from 1 to n
    double s_square_current = 
      square(theta_j_minus_one)*tau_current_sq + 
      tau_square*(1-square(rho));

    // u_{i+1} = u_j
    double u_current = theta_j_minus_one*v_current + alpha_j_minus_one;
    
    tau_current_sq = 
      1.0/(1.0/(v_square[gammas[i]]/4) + 1.0/s_square_current);
      
      v_current = tau_current_sq * 
	((y_star[i]-m[gammas[i]]/2.0) / (v_square[gammas[i]]/4.0) 
	 + u_current / s_square_current);
      
      taus_squared[i] = tau_current_sq;
      vs[i] = v_current;
  }

  double theta_j_minus_one = 
    sv_model_->get_ou_model()->
    theta_j(sv_model_->data_length()-1,gammas[sv_model_->data_length()-1]);
  double alpha_j_minus_one = 
    sv_model_->get_ou_model()->
    alpha_j(sv_model_->data_length()-1,gammas[sv_model_->data_length()-1]);

  double s_square_current = 
    square(theta_j_minus_one)*tau_current_sq + 
    tau_square*(1-square(rho));
  
  double u_current = theta_j_minus_one*v_current + alpha_j_minus_one;
  
  // BACKWARD SMOOTHER
  double new_log_sigma = 
    u_current + sqrt(s_square_current) * gsl_ran_gaussian(rng_,1.0);

  sv_model_->get_ou_model()->
    set_sigmas_element(sv_model_->data_length(),
		       exp(new_log_sigma) / sqrt(sv_model_->get_delta_t()),
		       new_log_sigma);
  
  for (std::vector<int>::size_type i = sv_model_->data_length()-1;
       i != (std::vector<int>::size_type)-1; --i) {
    double alpha_j = sv_model_->get_ou_model()->alpha_j(i,gammas[i]);
    double theta_j = sv_model_->get_ou_model()->theta_j(i,gammas[i]);

    double omega_ts_sq = 
      1.0 / ( square(theta_j) / (tau_square*(1-square(rho))) +
	      1.0 / taus_squared[i]);

    double q_t = omega_ts_sq * 
      ( ((new_log_sigma - alpha_j)*theta_j)/ (tau_square*(1-square(rho)))
	+ vs[i]/taus_squared[i] );

    new_log_sigma = q_t + sqrt(omega_ts_sq)*gsl_ran_gaussian(rng_, 1.0);
    
    sv_model_->get_ou_model()->
      set_sigmas_element(i,
			 exp(new_log_sigma) / sqrt(sv_model_->get_delta_t()),
			 new_log_sigma);
  }
}

// void StochasticVolatilityPosteriorSampler::draw_filtered_log_prices()
// {
//   double mu =
//     model_->get_mu().get_discrete_time_parameter(model_->get_delta_t());
//   const std::vector<SigmaSingletonParameter>& sigmas = 
//     model_->get_sigmas().get_sigmas();

//   double xi_square = model_->get_xi_square().get_continuous_time_parameter();

//   double m_not = log(model_->get_data_element(0).get_open());
//   double C_not = square(m_not*10.0);
  
//   std::vector<double> vs (model_->data_length()+1);
//   std::vector<double> taus_sq (model_->data_length()+1);
  
//   double v_current = m_not;
//   double tau_current_sq = C_not;
  
//   vs[0] = v_current;
//   taus_sq[0] = tau_current_sq;

//   double u_current = 0.0;
//   double s_current_sq = 0.0;

//   for (unsigned i=0; i<model_->data_length(); ++i) {
//     u_current = mu + v_current;
//     s_current_sq = 
//       square(sigmas[i].get_discrete_time_parameter(model_->get_delta_t()))
//       + tau_current_sq;
    
//     tau_current_sq = 1.0 / 
//       (1.0/xi_square + 1.0/s_current_sq);
    
//     v_current = 
//       log(model_->get_data_element(i).get_close())/(1.0 + xi_square/s_current_sq) + 
//       u_current/(1.0 + s_current_sq/xi_square);

//     vs[i+1] = v_current;
//     taus_sq[i+1] = tau_current_sq;
//   }

//   std::vector<double> samples (model_->data_length()+1);

//   samples[model_->data_length()] = 
//     vs[model_->data_length()] + 
//     sqrt(taus_sq[model_->data_length()])*gsl_ran_gaussian(rng_, 1.0);

//   // BACKWARD SAMPLER
//   for (std::vector<int>::size_type i = model_->data_length()-1;
//        i != (std::vector<int>::size_type) -1; --i) {
 
//    double sigma2 = 
//      square(sigmas[i+1].get_discrete_time_parameter(model_->get_delta_t()));

//     double omega_current_sq = 1.0/
//       (1.0/sigma2 + 1.0/taus_sq[i]);
//     double q_current = 
//       (samples[i+1]-mu)/(1.0 + sigma2/taus_sq[i]) + vs[i]/(taus_sq[i]/sigma2 + 1.0);

//     samples[i] = q_current + sqrt(omega_current_sq)*gsl_ran_gaussian(rng_, 1.0);
//   }

//   model_->set_filtered_log_prices(samples);
//   model_->set_y_star_ds();
// }

// void StochasticVolatilityPosteriorSampler::draw_rho()
// {
//   const std::vector<double>& h = 
//     model_->get_sigmas().get_discrete_time_log_sigmas();
//   const std::vector<int> ds = 
//     model_->get_ds();
//   double alpha = 
//     model_->get_alpha().get_discrete_time_parameter(model_->get_delta_t());
//   double theta = 
//     model_->get_theta().get_discrete_time_parameter(model_->get_delta_t());
//   double rho_current = 
//     model_->get_rho().get_continuous_time_parameter();
//   double tau_square = 
//     model_->get_tau_square().get_discrete_time_parameter(model_->get_delta_t());
//   const std::vector<int>& gammas = 
//     model_->get_gammas().get_gammas();
//   const std::vector<double>& y_star = 
//     model_->get_y_star();

//   const std::vector<double>& bs = model_->get_bs();
//   const std::vector<double>& as = model_->get_as();

//   const std::vector<double>& ms = model_->get_gammas().get_mixture_means();

//   double proposal_mean_over_variance = 0.0;
//   double proposal_variance_inverse = 0.0;
  
//   for (unsigned i=0; i<model_->data_length(); ++i) {
    
//     double Ft = ds[i] * sqrt(tau_square) * bs[gammas[i]] * 
//       2.0 * exp(ms[gammas[i]]/2.0);
    
//     double Mt = ds[i] * sqrt(tau_square) * exp(ms[gammas[i]]/2.0) * 
//       (as[gammas[i]] + bs[gammas[i]] * 2.0 * (y_star[i] - ms[gammas[i]]/2.0));
    
//     // double theta_j = model_->theta_j(i,gammas[i]);
//     // double alpha_j = model_->alpha_j(i,gammas[i]);
    
//     // double R_one_j =  -1.0 * (theta_j - theta) / rho_current;
//     // double R_two_j = (alpha_j - alpha*(1.0-theta)) / rho_current;
    
//     // proposal_variance_inverse = proposal_variance_inverse + 
//     // 	square(R_one_j*h[i] + R_two_j) / tau_square;
    
//     // proposal_mean_over_variance = proposal_mean_over_variance + 
//     // 	-1.0 * (R_one_j*h[i] + R_two_j) * 
//     // 	(h[i+1] - theta*h[i] - alpha*(1-theta)) /
//     // 	tau_square;
    
//     proposal_variance_inverse = proposal_variance_inverse + 
//       square(Ft*h[i] - Mt);
    
//     proposal_mean_over_variance = proposal_mean_over_variance + 
//       -1.0 * (Ft*h[i] - Mt) * 
//       (h[i+1] - theta*h[i] - alpha*(1-theta));
//   }
//   double proposal_mean = proposal_mean_over_variance / proposal_variance_inverse;
//   double proposal_variance = tau_square/proposal_variance_inverse;


//   double rho_proposal = 
//     rtruncnorm(rng_, -1.0, 1.0, proposal_mean, sqrt(proposal_variance));

//   model_->set_rho(rho_proposal);  
//   double log_likelihood_proposal = model_->log_likelihood();

//   model_->set_rho(rho_current);
//   double log_likelihood_current = model_->log_likelihood();


//   double q_proposal_given_current = 
//     dtruncnorm(rho_proposal, -1.0, 1.0, 
// 	       proposal_mean, sqrt(proposal_variance),
// 	       1);
//   double q_current_given_proposal =  
//     dtruncnorm(rho_current, -1.0, 1.0, 
// 	       proposal_mean, sqrt(proposal_variance),
// 	       1);

//   // double q_current_given_proposal = 
//   //   dnorm(rho_tilde_current, rho_tilde_proposal, rho_tilde_proposal_std_dev, 1) + 
//   //   (log(2.0) - log(abs(rho_current+1)) - log(abs(1-rho_current)));

//   // double q_proposal_given_current = 
//   //   dnorm(rho_tilde_proposal, rho_tilde_current, rho_tilde_proposal_std_dev, 1) + 
//   //   (log(2.0) - log(rho_proposal+1) - log(1-rho_proposal));

//   double log_a_acceptance = 
//     (log_likelihood_proposal + model_->get_rho_prior().log_likelihood(rho_proposal) 
//      + q_current_given_proposal)
//     -
//     (log_likelihood_current + model_->get_rho_prior().log_likelihood(rho_current) + 
//      q_proposal_given_current);

//   // std::cout << "rho_current = " << rho_current
//   // 	    << " "
//   // 	    << "rho_proposal = " << rho_proposal
//   // 	    << " ";
//   // std::cout << "log_likelihood_current = " << log_likelihood_current 
//   // 	    << " "
//   // 	    << "log_likelihood_proposal = " << log_likelihood_proposal
//   // 	    << " ";
//   // std::cout << "q_current_given_proposal = " << q_current_given_proposal
//   // 	    << " " 
//   // 	    << "q_proposal_given_current = " << q_proposal_given_current
//   // 	    << " " << "rho_proposal prior="
//   // 	    << model_->get_rho_prior().log_likelihood(rho_proposal)
//   // 	    << " " << "rho_current prior="
//   // 	    << model_->get_rho_prior().log_likelihood(rho_current)
//   // 	    << " " << "log_a_acceptance = " << log_a_acceptance
//   // 	    << std::endl;

//   if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
//     std::cout << "rho accepted; rho=" << rho_proposal << std::endl;
//     model_->set_rho(rho_proposal);
//   }
// }

// void StochasticVolatilityPosteriorSampler::draw_xi_square()
// {
//   double alpha = 
//     model_->get_xi_square_prior().get_xi_square_shape() + 
//     model_->data_length()/2.0;

//   double beta = 
//     model_->get_xi_square_prior().get_xi_square_scale();

//   const std::vector<double>& filtered_log_prices = 
//     model_->get_filtered_log_prices();

//   for (unsigned i=0; i<model_->data_length(); ++i) {
//     beta = beta + 
//       0.5*square(log(model_->get_data_element(i).get_close())
// 		 - filtered_log_prices[i+1]);
//   }

//   double xi_square_inv = gsl_ran_gamma(rng_, alpha, 1.0/beta);
//   double xi_square = 1.0/xi_square_inv;
//   model_->set_xi_square(xi_square);
// }

// void StochasticVolatilityPosteriorSampler::draw_theta_tau_square_rho()
// { 
//   std::vector<double> mean {model_->get_rho().get_continuous_time_parameter(),
//       model_->get_theta().get_discrete_time_parameter(model_->get_delta_t()),
//       model_->get_tau_square().get_discrete_time_parameter(model_->get_delta_t())};

//   std::vector<double> proposal = 
//     theta_tau_square_rho_proposal_.propose_parameters(rng_, mean);

//   double rho_proposal = proposal[0];
//   double theta_proposal = proposal[1];
//   double tau_square_proposal = proposal[2];

//   double rho_current = model_->get_rho().get_continuous_time_parameter();

//   double theta_hat_proposal = -1.0*log(theta_proposal) / 
//     model_->get_delta_t();
//   double theta_hat_current = model_->get_theta().get_continuous_time_parameter();
//   double theta_current =  model_->get_theta().
//     get_discrete_time_parameter(model_->get_delta_t());

//   double tau_square_hat_proposal = 
//     tau_square_proposal / 
//     ((1.0 - exp(-2.0*theta_hat_proposal*model_->get_delta_t()))/(2.0*theta_hat_proposal));
//   double tau_square_hat_current = model_->get_tau_square().get_continuous_time_parameter();
//   double tau_square_current = model_->get_tau_square().
//     get_discrete_time_parameter(model_->get_delta_t());

//   model_->set_tau_square_hat(tau_square_hat_proposal);
//   model_->set_theta_hat(theta_hat_proposal);
//   model_->set_rho(rho_proposal);

//   double log_likelihood_proposal = model_->log_likelihood() + 
//     model_->get_rho_prior().log_likelihood(rho_proposal) + 
//     model_->get_theta_prior().log_likelihood(theta_proposal) +
//     model_->get_tau_square_prior().log_likelihood(tau_square_proposal) + 
//     log(rho_proposal + 1) + log((1-rho_proposal)/2.0) + 
//     log(theta_proposal) + log(1-theta_proposal) + 
//     log(tau_square_proposal);

//   model_->set_tau_square_hat(tau_square_hat_current);
//   model_->set_theta_hat(theta_hat_current);
//   model_->set_rho(rho_current);
//   double log_likelihood_current = model_->log_likelihood() +
//     model_->get_rho_prior().log_likelihood(rho_current) + 
//     model_->get_theta_prior().log_likelihood(theta_current) + 
//     model_->get_tau_square_prior().log_likelihood(tau_square_current) + 
//     log(rho_current + 1) + log((1-rho_current)/2.0) + 
//     log(theta_current) + log(1-theta_current) + 
//     log(tau_square_current);
  
//   std::cout << rho_current << " "
// 	    << theta_hat_current << " "
// 	    << tau_square_hat_current << " " 
// 	    << std::endl;

//   std::cout << rho_proposal << " "
// 	    << theta_hat_proposal << " "
// 	    << tau_square_hat_proposal << " ";
	 

//   std::cout << "; ll_current = " << log_likelihood_current 
// 	    << "; ll_proposal = " << log_likelihood_proposal;

//   double log_a_acceptance = log_likelihood_proposal -
//     log_likelihood_current;


//   if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
//     std::cout << "; move accepted ";
//     model_->set_tau_square_hat(tau_square_hat_proposal);
//     model_->set_theta_hat(theta_hat_proposal);
//     model_->set_rho(rho_proposal);
//     acceptance_ratio_ = (acceptance_ratio_*number_iterations_ + 1) / 
//       (number_iterations_ + 1);
//   }

//   number_iterations_ = number_iterations_ + 1;
//   std::cout << std::endl;
// }

// double StochasticVolatilityPosteriorSampler::get_acceptance_ratio() const
// {
//   return acceptance_ratio_;
// }

// ======================== SV MULTIFACTOR SAMPLER =======================
MultifactorStochasticVolatilityPosteriorSampler::
MultifactorStochasticVolatilityPosteriorSampler(MultifactorStochasticVolatilityModel *model,
						gsl_rng * rng,
						const gsl_matrix * proposal_covariance_matrix)
  : sv_model_(model),
    rng_(rng),
    observational_model_sampler_(ObservationalPosteriorSampler(sv_model_->
							       get_observational_model(),
							       rng)),
    constant_vol_sampler_(ConstantVolatilityPosteriorSampler(sv_model_->get_constant_vol_model(),
							     rng)),
    ou_sampler_slow_(OUPosteriorSampler(sv_model_->get_ou_model_slow(),
					rng,
					proposal_covariance_matrix)),
    ou_sampler_fast_(FastOUPosteriorSampler(sv_model_->get_ou_model_fast(),
					rng,
					proposal_covariance_matrix))
{}

MultifactorStochasticVolatilityPosteriorSampler::
~MultifactorStochasticVolatilityPosteriorSampler() 
{}

void MultifactorStochasticVolatilityPosteriorSampler::draw_gammas()
{
  sv_model_->get_constant_vol_model()->set_y_star_ds();
  const std::vector<double>& h_fast = 
    sv_model_->get_ou_model_fast()->get_sigmas().get_discrete_time_log_sigmas();
  const std::vector<double>& h_slow = 
    sv_model_->get_ou_model_slow()->get_sigmas().get_discrete_time_log_sigmas();

  const std::vector<double>& y_star = 
    sv_model_->get_constant_vol_model()->get_y_star();
  double tau_square_fast = 
    sv_model_->get_ou_model_fast()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());

  double rho = 
    sv_model_->get_ou_model_fast()->
    get_rho().get_discrete_time_parameter(sv_model_->get_delta_t());

  const std::vector<double>& m = 
    sv_model_->get_constant_vol_model()->get_gammas().get_mixture_means();
  const std::vector<double>& v_square = 
    sv_model_->get_constant_vol_model()->get_gammas().get_mixture_variances();
  const std::vector<double>& probs = 
    sv_model_->get_constant_vol_model()->get_gammas().get_mixture_probabilities();

  std::vector<int> new_gamma (sv_model_->data_length());
  std::vector<double> 
    log_posterior_mixture_component_probability(sv_model_->
						get_constant_vol_model()->
						get_gammas().get_J());
  std::vector<double> 
    normalized_posterior_mixture_component_probability(sv_model_->
						       get_constant_vol_model()->
						       get_gammas().get_J());

  for (unsigned i=0; i<sv_model_->data_length(); ++i) {
    double max_element = 0.0;
    
    for (int j=0; j<sv_model_->get_constant_vol_model()->get_gammas().get_J(); ++j) {
      double theta_j_one = 
	sv_model_->get_ou_model_fast()->theta_j_one(i,j);

      double theta_j_two = 
	sv_model_->get_ou_model_fast()->theta_j_two(i,j);

      double alpha_j = sv_model_->get_ou_model_fast()->alpha_j(i,j);

      double element = 
	-1.0*log(sqrt(v_square[j])/2.0) +
	-1.0/(2.0*v_square[j]/4.0) * square(y_star[i]-0.5*h_fast[i]-0.5*h_slow[i]-m[j]/2.0) + 
	-1.0/(2.0*tau_square_fast*(1.0-square(rho))) * 
	square(h_fast[i+1] - theta_j_one*h_slow[i] - theta_j_two*h_fast[i] - alpha_j) +
	log(probs[j]);
      if (j==0) {
	max_element = element;
      } else {
	if (element >= max_element) {
	  max_element = element;
	}
      }
      log_posterior_mixture_component_probability[j] = element;
    }
    
    for (int j=0; j<sv_model_->get_constant_vol_model()->get_gammas().get_J(); ++j) {
      normalized_posterior_mixture_component_probability[j] =
	exp(log_posterior_mixture_component_probability[j] - max_element);
    }

    // std::discrete_distribution<int> 
    //   component_sampler(normalized_posterior_mixture_component_probability.begin(),
    //  			normalized_posterior_mixture_component_probability.end());
    
    double sum = 0;
    for (unsigned j=0; 
    	 j<normalized_posterior_mixture_component_probability.size();
    	 ++j) {
      sum = sum + normalized_posterior_mixture_component_probability[j];
    }

    unsigned j=0;    
    double running_sum = normalized_posterior_mixture_component_probability[j];
    double component_index = gsl_ran_flat(rng_, 0, sum);
    while (running_sum < component_index) {
      j=j+1;
      running_sum = running_sum + 
    	normalized_posterior_mixture_component_probability[j];
    }

    if (j > normalized_posterior_mixture_component_probability.size()-1) {
      std::cout << sum << " " << j << " " << component_index << std::endl;
    }

    new_gamma[i] = j;
    // new_gamma[i] = component_sampler(mt_);
  }
  sv_model_->get_constant_vol_model()->set_gammas(new_gamma);
}

void MultifactorStochasticVolatilityPosteriorSampler::draw_gammas_gsl()
{
  sv_model_->
    get_constant_vol_model()->
    set_y_star_ds();
  const std::vector<double>& h_slow = 
    sv_model_->
    get_ou_model_slow()->
    get_sigmas().get_discrete_time_log_sigmas();

  const std::vector<double>& h_fast = 
    sv_model_->
    get_ou_model_fast()->
    get_sigmas().get_discrete_time_log_sigmas();

  const std::vector<double>& y_star = 
    sv_model_->
    get_constant_vol_model()->
    get_y_star();

  double tau_square_fast = 
    sv_model_->
    get_ou_model_fast()->
    get_tau_square().get_discrete_time_parameter(sv_model_->
						 get_delta_t());
  double rho = 
    sv_model_->
    get_ou_model_fast()->
    get_rho().get_continuous_time_parameter();

  const std::vector<double>& m = 
    sv_model_->
    get_constant_vol_model()->
    get_gammas().get_mixture_means();
  const std::vector<double>& v_square = 
    sv_model_->
    get_constant_vol_model()->
    get_gammas().get_mixture_variances();
  const std::vector<double>& probs = 
    sv_model_->
    get_constant_vol_model()->
    get_gammas().get_mixture_probabilities();

  std::vector<int> new_gamma (sv_model_->data_length());
  int J = sv_model_->get_constant_vol_model()->get_gammas().get_J();

  double posterior_mixture_component_probability[J];    
  for (unsigned i=0; i<sv_model_->data_length(); ++i) {

    for (int j=0; j<J; ++j) {
      double theta_j_slow = sv_model_->get_ou_model_fast()->
	theta_j_one(i,j);
      double theta_j_fast = sv_model_->get_ou_model_fast()->
	theta_j_two(i,j);
      double alpha_j =  sv_model_->get_ou_model_fast()->
	alpha_j(i,j);
            
      double element = 
	-1.0*log(sqrt(v_square[j])/2.0) +
	-1.0/(2.0*v_square[j]/4.0) * square(y_star[i]-0.5*h_slow[i]-0.5*h_fast[i]-m[j]/2.0) + 
	-1.0/(2.0*tau_square_fast*(1.0-square(rho))) * 
	square(h_fast[i+1] - 
	       theta_j_slow*h_slow[i] - 
	       theta_j_fast*h_fast[i] - 
	       alpha_j) +
	log(probs[j]);

      posterior_mixture_component_probability[j] = exp(element);
    }
    
    double *P;
    P = posterior_mixture_component_probability;

    gsl_ran_discrete_t * g = 
      gsl_ran_discrete_preproc(J, P);
    int j = gsl_ran_discrete (rng_, g);
    new_gamma[i] = j;
    gsl_ran_discrete_free(g);
  }

  sv_model_->get_constant_vol_model()->set_gammas(new_gamma);
}

void MultifactorStochasticVolatilityPosteriorSampler::draw_alpha_hat()
{
  std::vector<double> alpha_likelihood_mean_var_slow = 
    ou_sampler_slow_.get_ou_model()->alpha_posterior_mean_var();
  std::cout << "slow mean = " << alpha_likelihood_mean_var_slow[0] - 
    0.5*log(sv_model_->get_delta_t())
	    << "\n";

  std::vector<double> alpha_likelihood_mean_var_fast = 
    ou_sampler_fast_.get_ou_model_fast()->alpha_posterior_mean_var();

  std::cout << "fast mean = " << alpha_likelihood_mean_var_fast[0] -
    0.5*log(sv_model_->get_delta_t())
	    << "\n";

  double prior_mean = sv_model_->get_ou_model_fast()->get_alpha_prior().get_alpha_mean();
  double prior_var = square(sv_model_->get_ou_model_fast()->
			    get_alpha_prior().get_alpha_std_dev());

  std::cout << "prior mean = " << prior_mean
	    << "\n";

  double posterior_var = 1.0/ (1.0/alpha_likelihood_mean_var_slow[1] + 
			       1.0/alpha_likelihood_mean_var_fast[1] + 
			       1.0/prior_var);

  double posterior_mean = posterior_var * 
    ( alpha_likelihood_mean_var_slow[0]/alpha_likelihood_mean_var_slow[1] + 
      alpha_likelihood_mean_var_fast[0]/alpha_likelihood_mean_var_fast[1] + 
      prior_mean/prior_var);

  double alpha = posterior_mean + 
    sqrt(posterior_var)*gsl_ran_gaussian(rng_, 1.0);
  double alpha_hat = alpha - 0.5*log(sv_model_->get_delta_t());
  
  ou_sampler_slow_.get_ou_model()->set_alpha_hat(alpha_hat);
  ou_sampler_fast_.get_ou_model_fast()->set_alpha_hat(alpha_hat);
}

void MultifactorStochasticVolatilityPosteriorSampler::draw_sigmas()
{
  std::vector<arma::mat> taus_squared_inv (sv_model_->data_length());
  std::vector<arma::vec> vs (sv_model_->data_length());

  const std::vector<double>& m = 
    sv_model_->get_constant_vol_model()->
    get_gammas().get_mixture_means();
  const std::vector<double>& v_square = 
    sv_model_->get_constant_vol_model()->
    get_gammas().get_mixture_variances();

  const std::vector<int>& gammas = 
    sv_model_->get_constant_vol_model()->
    get_gammas().get_gammas();
  const std::vector<double>& y_star = 
    sv_model_->get_constant_vol_model()->get_y_star();

  double tau_square_slow = 
    sv_model_->get_ou_model_slow()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());
  double theta_slow =
    sv_model_->get_ou_model_slow()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());

  double tau_square_fast = 
    sv_model_->get_ou_model_fast()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());
  double theta_fast = 
    sv_model_->get_ou_model_fast()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());

  double alpha = 
    sv_model_->get_ou_model_fast()->get_alpha().
    get_discrete_time_parameter(sv_model_->get_delta_t());
  
  double rho = 
    sv_model_->get_ou_model_fast()->get_rho().get_continuous_time_parameter();

  // FORWARD FILTER
  // current mean level;
  double v_current_fast = 
    sv_model_->get_ou_model_fast()->
    get_alpha().get_discrete_time_parameter(sv_model_->get_delta_t());
  double v_current_slow =
    sv_model_->get_ou_model_slow()->
    get_alpha().get_discrete_time_parameter(sv_model_->get_delta_t());

  arma::vec v_current(2);
  v_current(0) = v_current_slow;
  v_current(1) = v_current_fast;

  // current variance level;
  double tau_current_sq_fast = 
    tau_square_fast / 
    (1 - square(theta_fast));

  double tau_current_sq_slow = 
    tau_square_slow / 
    (1 - square(theta_slow));

  arma::mat tau_current_sq = arma::zeros<arma::mat> (2,2);
  tau_current_sq(0,0) = tau_current_sq_slow;
  tau_current_sq(1,1) = tau_current_sq_fast;

  vs[0] = v_current;
  taus_squared_inv[0] = inv(tau_current_sq);

  // freed at end of loop
  arma::vec m_t = arma::vec (2);
  arma::mat s_square_current = arma::mat (2,2);
  arma::vec u_current  = arma::vec (2);

  arma::mat G_j_minus_one = zeros<mat> (2,2);
  arma::mat G_j = zeros<mat> (2,2);

  arma::mat F = arma::mat (2,1);
  F(0,0) = 0.5;
  F(1,0) = 0.5;
  arma::mat Z = zeros<mat> (2,2);
  Z(0,0) = tau_square_slow;
  Z(1,1) = tau_square_fast * (1-square(rho));
  arma::mat Z_inv = inv(Z);

  for (unsigned i=1; i<sv_model_->data_length(); ++i) {
    		   
    double theta_j_minus_one_slow = 
      sv_model_->get_ou_model_fast()->theta_j_one(i-1,gammas[i-1]);

    double theta_j_minus_one_fast = 
      sv_model_->get_ou_model_fast()->theta_j_two(i-1,gammas[i-1]);
    
    double alpha_j_minus_one = 
      sv_model_->get_ou_model_fast()->alpha_j(i-1,gammas[i-1]);

    m_t(0) = alpha*(1-theta_slow);
    m_t(1) = alpha_j_minus_one;
    
    G_j_minus_one(0, 0) = theta_slow;
    G_j_minus_one(1, 0) = theta_j_minus_one_slow;
    G_j_minus_one(1, 1) = theta_j_minus_one_fast;

    // u_{i+1} = u_j
    u_current = G_j_minus_one * v_current + m_t;
    
    // s^2_{i+1} = s^2_{j}; j goes from 1 to n
    s_square_current = 
      G_j_minus_one * tau_current_sq * G_j_minus_one.t() + Z;

    // inverting s_square_current
    arma::mat s_square_current_inv = inv_sympd(s_square_current);

    // tau_current_sq
    arma::mat tau_current_sq_inv = 
      s_square_current_inv + 
      (F*F.t())/(v_square[gammas[i]]/4.0);

    tau_current_sq = 
      inv_sympd(s_square_current_inv + 
		(F*F.t())/(v_square[gammas[i]]/4.0));

    // v_current 
    v_current = tau_current_sq * 
      (s_square_current_inv*u_current + 
       F*(y_star[i]-m[gammas[i]]/2.0) / (v_square[gammas[i]]/4.0));

    vs[i] = v_current;
    taus_squared_inv[i] = tau_current_sq_inv;
  }

  double theta_j_minus_one_slow = 
    sv_model_->get_ou_model_fast()->theta_j_one(sv_model_->data_length()-1,
						gammas[sv_model_->data_length()-1]);

  double theta_j_minus_one_fast = 
    sv_model_->get_ou_model_fast()->theta_j_two(sv_model_->data_length()-1,
						gammas[sv_model_->data_length()-1]);

  double alpha_j_minus_one = 
    sv_model_->get_ou_model_fast()->alpha_j(sv_model_->data_length()-1,
					    gammas[sv_model_->data_length()-1]);

  m_t(0) = alpha*(1-theta_slow);
  m_t(1) = alpha_j_minus_one;
  
  G_j_minus_one(0, 0) = theta_slow;
  G_j_minus_one(1, 0) = theta_j_minus_one_slow;
  G_j_minus_one(1, 1) = theta_j_minus_one_fast;
  
  // u_{i+1} = u_j
  u_current = G_j_minus_one * v_current + m_t;
  
  // s^2_{i+1} = s^2_{j}; j goes from 1 to n
  s_square_current = 
    G_j_minus_one * tau_current_sq * G_j_minus_one.t() + Z;
  
  // BACKWARD SMOOTHER
  // last sample
  arma::vec h_tpone = rmvnorm(rng_, 2, u_current, s_square_current);
  sv_model_->get_ou_model_slow()
    ->set_sigmas_element(sv_model_->data_length(),
			 exp(h_tpone(0)) / sqrt(sv_model_->get_delta_t()),
			 h_tpone(0));

  sv_model_->get_ou_model_fast()
    ->set_sigmas_element(sv_model_->data_length(),
			 exp(h_tpone(1)) / sqrt(sv_model_->get_delta_t()),
			 h_tpone(1));
  
  for (std::vector<int>::size_type i = sv_model_->data_length()-1;
       i != (std::vector<int>::size_type)-1; --i) {

  double theta_j_slow = 
    sv_model_->get_ou_model_fast()->theta_j_one(i,gammas[i]);


    double theta_j_fast = 
      sv_model_->get_ou_model_fast()->theta_j_two(i,gammas[i]);
    
    double alpha_j = 
      sv_model_->get_ou_model_fast()->alpha_j(i,gammas[i]);

    m_t(0) = alpha*(1-theta_slow);
    m_t(1) = alpha_j;
    
    G_j(0, 0) = theta_slow;
    G_j(1, 0) = theta_j_slow;
    G_j(1, 1) = theta_j_fast;
    
    arma::mat omega_ts_sq = 
      inv_sympd(G_j.t() * Z_inv * G_j + taus_squared_inv[i]);

    arma::vec q_t = omega_ts_sq * 
      (taus_squared_inv[i]*vs[i] + G_j.t()*Z_inv*(h_tpone-m_t));
    
    h_tpone = rmvnorm(rng_, 2, q_t, omega_ts_sq);
        
    sv_model_->get_ou_model_slow()->
      set_sigmas_element(i,
    			 exp(h_tpone(0)) / sqrt(sv_model_->get_delta_t()),
    			 h_tpone(0));

    sv_model_->get_ou_model_fast()->
      set_sigmas_element(i,
    			 exp(h_tpone(1)) / sqrt(sv_model_->get_delta_t()),
    			 h_tpone(1));
	
  }
}

void MultifactorStochasticVolatilityPosteriorSampler::draw_filtered_log_prices()
{
//   double mu =
//     model_->get_mu().get_discrete_time_parameter(model_->get_delta_t());
//   const std::vector<SigmaSingletonParameter>& sigmas_fast = 
//     model_->get_ou_model_fast()->get_sigmas().get_sigmas();
//   const std::vector<SigmaSingletonParameter>& sigmas_slow = 
//     model_->get_ou_model_slow()->get_sigmas().get_sigmas();

//   double xi_square = model_->get_xi_square().get_continuous_time_parameter();

//   double m_not = log(model_->get_data_element(0).get_open());
//   double C_not = square(m_not*10.0);
  
//   std::vector<double> vs (model_->data_length()+1);
//   std::vector<double> taus_sq (model_->data_length()+1);
  
//   double v_current = m_not;
//   double tau_current_sq = C_not;
  
//   vs[0] = v_current;
//   taus_sq[0] = tau_current_sq;

//   double u_current = 0.0;
//   double s_current_sq = 0.0;

//   for (unsigned i=0; i<model_->data_length(); ++i) {
//     u_current = mu + v_current;
//     s_current_sq = 
//       square(sigmas_fast[i].get_continuous_time_parameter()*
// 	     sigmas_slow[i].get_continuous_time_parameter()*
// 	     model_->get_delta_t())
//       + tau_current_sq;
    
//     tau_current_sq = 1.0 / 
//       (1.0/xi_square + 1.0/s_current_sq);
    
//     v_current = 
//       log(model_->get_data_element(i).get_close())/(1.0 + xi_square/s_current_sq) + 
//       u_current/(1.0 + s_current_sq/xi_square);

//     vs[i+1] = v_current;
//     taus_sq[i+1] = tau_current_sq;
//   }

//   std::vector<double> samples (model_->data_length()+1);

//   samples[model_->data_length()] = 
//     vs[model_->data_length()] + 
//     sqrt(taus_sq[model_->data_length()])*gsl_ran_gaussian(rng_, 1.0);

//   // BACKWARD SAMPLER
//   for (std::vector<int>::size_type i = model_->data_length()-1;
//        i != (std::vector<int>::size_type) -1; --i) {
 
//    double sigma2 = 
//      square(sigmas_fast[i+1].get_continuous_time_parameter()*
// 	    sigmas_slow[i+1].get_continuous_time_parameter()*
// 	    model_->get_delta_t());

//     double omega_current_sq = 1.0/
//       (1.0/sigma2 + 1.0/taus_sq[i]);
//     double q_current = 
//       (samples[i+1]-mu)/(1.0 + sigma2/taus_sq[i]) + vs[i]/(taus_sq[i]/sigma2 + 1.0);

//     samples[i] = q_current + sqrt(omega_current_sq)*gsl_ran_gaussian(rng_, 1.0);
//   }

//   model_->set_filtered_log_prices(samples);
//   model_->set_y_star_ds();
}

// ================ SV MULTIFACTOR SAMPLER WITH JUMPS ===============

SVWithJumpsPosteriorSampler::
SVWithJumpsPosteriorSampler(SVModelWithJumps *model,
			    gsl_rng * rng,
			    const gsl_matrix * proposal_covariance_matrix,
			    const gsl_matrix * proposal_covariance_matrix_all)
  : sv_model_(model),
    rng_(rng),
    observational_model_sampler_(ObservationalPosteriorSampler(sv_model_->
							       get_observational_model(),
							       rng)),
    constant_vol_sampler_(ConstantVolatilityWithJumpsPosteriorSampler(sv_model_->get_constant_vol_model(),
								      rng)),
    ou_sampler_slow_(OUPosteriorSampler(sv_model_->get_ou_model_slow(),
					rng,
					proposal_covariance_matrix)),
    ou_sampler_fast_(FastOUPosteriorSampler(sv_model_->get_ou_model_fast(),
					    rng,
					    proposal_covariance_matrix)),
  normal_rw_proposal_(NormalRWProposal(5, proposal_covariance_matrix_all)),
  normal_rw_proposal_obs_model_params_(NormalRWProposal(3,proposal_covariance_matrix))
{}

SVWithJumpsPosteriorSampler::~SVWithJumpsPosteriorSampler()
{}

void SVWithJumpsPosteriorSampler::draw_gammas_gsl()
{
  const std::vector<int>& ds = 
    sv_model_->get_constant_vol_model()->
    get_ds();
  const std::vector<double>& bs_correction = 
    sv_model_->get_constant_vol_model()->
    get_bs();
  const std::vector<double>& as_correction = 
    sv_model_->get_constant_vol_model()->
    get_as();

  const std::vector<double>& h_slow = 
    sv_model_->
    get_ou_model_slow()->
    get_sigmas().get_discrete_time_log_sigmas();

  const std::vector<double>& h_fast = 
    sv_model_->
    get_ou_model_fast()->
    get_sigmas().get_discrete_time_log_sigmas();

  const std::vector<double>& y_star = 
    sv_model_->
    get_constant_vol_model()->
    get_y_star();

  double alpha =
    sv_model_->
    get_ou_model_fast()->
    get_alpha().get_discrete_time_parameter(sv_model_->
					    get_delta_t());

  double theta_fast =
    sv_model_->
    get_ou_model_fast()->
    get_theta().get_discrete_time_parameter(sv_model_->
					    get_delta_t());
  double tau_square_fast = 
    sv_model_->
    get_ou_model_fast()->
    get_tau_square().get_discrete_time_parameter(sv_model_->
						 get_delta_t());
  double rho = 
    sv_model_->
    get_ou_model_fast()->
    get_rho().get_continuous_time_parameter();

  const std::vector<double>& m = 
    sv_model_->
    get_constant_vol_model()->
    get_gammas().get_mixture_means();
  const std::vector<double>& v_square = 
    sv_model_->
    get_constant_vol_model()->
    get_gammas().get_mixture_variances();
  const std::vector<double>& probs = 
    sv_model_->
    get_constant_vol_model()->
    get_gammas().get_mixture_probabilities();

  // std::vector<int> new_gamma (sv_model_->data_length());
  int J = sv_model_->get_constant_vol_model()->get_gammas().get_J();

  double posterior_mixture_component_probability[J];    
  for (int j=0; j<J; ++j) {
    posterior_mixture_component_probability[j] = 1;
  }
  double theta_j_slow = 0;
  double theta_j_fast = 0;
  double alpha_j = 0;
  double element = 0;
  double max_element = -1.0*HUGE_VAL;
  for (unsigned i=0; i<sv_model_->data_length(); ++i) {
    max_element = -1.0*HUGE_VAL;
    for (int j=0; j<J; ++j) {
      theta_j_slow =
	-rho*sqrt(tau_square_fast)*ds[i]*exp(m[j]/2.0)*
	bs_correction[j];

      theta_j_fast = 
	theta_fast - rho*sqrt(tau_square_fast)*ds[i]*exp(m[j]/2.0)*
	bs_correction[j];
      
      alpha_j = 
	alpha*(1-theta_fast)
	+ rho*sqrt(tau_square_fast)*ds[i]*exp(m[j]/2.0)
	* (as_correction[j] + bs_correction[j]
	   * 2.0 * (y_star[i] - m[j]/2.0));

      // theta_j_slow = sv_model_->get_ou_model_fast()->
      // 	theta_j_one(i,j);
      // theta_j_fast = sv_model_->get_ou_model_fast()->
      // 	theta_j_two(i,j);
      // alpha_j =  sv_model_->get_ou_model_fast()->
      // 	alpha_j(i,j);
      
      element = 
    	-1.0*log(sqrt(v_square[j])/2.0) +
    	-1.0/(2.0*v_square[j]/4.0) * square(y_star[i]-
					    0.5*h_slow[i]-
					    0.5*h_fast[i]-
					    m[j]/2.0) + 
    	-1.0/(2.0*tau_square_fast*(1.0-square(rho))) * 
    	square(h_fast[i+1] - 
    	       theta_j_slow*h_slow[i] - 
    	       theta_j_fast*h_fast[i] - 
    	       alpha_j) +
    	log(probs[j]);

      if (element >= max_element) {
	max_element = element;
      }
      posterior_mixture_component_probability[j] = (element);
    }

    for (int j=0; j<J; ++j) {
      posterior_mixture_component_probability[j] = 
    	exp(posterior_mixture_component_probability[j] - 
    	    max_element);
    }

    double *P;
    P = posterior_mixture_component_probability;
    gsl_ran_discrete_t * g = gsl_ran_discrete_preproc(J,P);
    int jj = gsl_ran_discrete (rng_, g);

    // new_gamma[i] = j;
    gsl_ran_discrete_free(g);
    sv_model_->get_constant_vol_model()->set_gamma_element(i,jj);
  }
  // sv_model_->get_constant_vol_model()->set_gammas(new_gamma);
}

void SVWithJumpsPosteriorSampler::draw_alpha_hat()
{
  std::vector<double> alpha_likelihood_mean_var_slow = 
    ou_sampler_slow_.get_ou_model()->alpha_posterior_mean_var();
  // std::cout << "slow mean = " << alpha_likelihood_mean_var_slow[0]- 
  //   0.5*log(sv_model_->get_delta_t())
  // 	    << "\n";

  std::vector<double> alpha_likelihood_mean_var_fast = 
    ou_sampler_fast_.get_ou_model_fast()->alpha_posterior_mean_var();

  // std::cout << "fast mean = " << alpha_likelihood_mean_var_fast[0]- 
  //   0.5*log(sv_model_->get_delta_t())
  // 	    << "\n";

  double prior_mean = sv_model_->get_ou_model_fast()->get_alpha_prior().get_alpha_mean();
  double prior_var = square(sv_model_->get_ou_model_fast()->
			    get_alpha_prior().get_alpha_std_dev());

  // std::cout << "prior mean = " << prior_mean
  // 	    << "\n";

  double posterior_var = 1.0/ (1.0/alpha_likelihood_mean_var_slow[1] + 
			       1.0/alpha_likelihood_mean_var_fast[1] + 
			       1.0/prior_var);

  double posterior_mean = posterior_var * 
    ( alpha_likelihood_mean_var_slow[0]/alpha_likelihood_mean_var_slow[1] + 
      alpha_likelihood_mean_var_fast[0]/alpha_likelihood_mean_var_fast[1] + 
      prior_mean/prior_var);

  double alpha = posterior_mean + 
    sqrt(posterior_var)*gsl_ran_gaussian(rng_, 1.0);
  double alpha_hat = alpha - 0.5*log(sv_model_->get_delta_t());
  
  ou_sampler_slow_.get_ou_model()->set_alpha_hat(alpha_hat);
  ou_sampler_fast_.get_ou_model_fast()->set_alpha_hat(alpha_hat);
}

void SVWithJumpsPosteriorSampler::draw_sv_models_params_integrated_vol()
{
  // CURRENT MODEL PARAMETERS
  double alpha_current = ou_sampler_fast_.get_ou_model_fast()->
    get_alpha().get_discrete_time_parameter(sv_model_->get_delta_t());

  double theta_slow_current = ou_sampler_slow_.get_ou_model()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double theta_hat_slow_current = ou_sampler_slow_.get_ou_model()->
    get_theta().get_continuous_time_parameter();
  double tau_square_slow_current = ou_sampler_slow_.get_ou_model()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_hat_slow_current = ou_sampler_slow_.get_ou_model()->
    get_tau_square().get_continuous_time_parameter();

  double rho_current = ou_sampler_fast_.get_ou_model_fast()->
    get_rho().get_continuous_time_parameter();
  double theta_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double theta_hat_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_theta().get_continuous_time_parameter();
  double tau_square_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_hat_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square().get_continuous_time_parameter();

  // TRANSFORMING TO TILDE SCALE
  double theta_tilde_slow_current = logit(theta_slow_current);
  double tau_square_tilde_slow_current = log(tau_square_slow_current);
  
  double rho_tilde_current = logit((rho_current+1.0)/2.0);
  double theta_tilde_fast_current = logit(theta_fast_current);
  double tau_square_tilde_fast_current = log(tau_square_fast_current);

  // SETTING THE MEAN
  std::vector<double> mean { alpha_current,
      theta_tilde_slow_current,
      tau_square_tilde_slow_current,
      rho_tilde_current,
      theta_tilde_fast_current,
      tau_square_tilde_fast_current };

  // PROPOSING ON THE TILDE SCALE
  std::vector<double> proposal = 
    normal_rw_proposal_.propose_parameters(rng_, mean);

  double alpha_proposal = proposal[0];
  double theta_tilde_slow_proposal = proposal[1];
  double tau_square_tilde_slow_proposal = proposal[2];
  double rho_tilde_proposal = proposal[3];
  double theta_tilde_fast_proposal = proposal[4];
  double tau_square_tilde_fast_proposal = proposal[5];

  // TRANSFORMING TO THE NOMINAL SCALE
  double theta_slow_proposal = exp(theta_tilde_slow_proposal)/
    (exp(theta_tilde_slow_proposal) + 1.0);
  double tau_square_slow_proposal = exp(tau_square_tilde_slow_proposal);
  double rho_proposal = 2.0*exp(rho_tilde_proposal)/
    (exp(rho_tilde_proposal) + 1.0) - 1.0;
  double theta_fast_proposal = exp(theta_tilde_fast_proposal)/
    (exp(theta_tilde_fast_proposal) + 1.0);
  double tau_square_fast_proposal = exp(tau_square_tilde_fast_proposal);

  // CALCULATING LOG LIKELIHOOD FOR PROPOSAL
  double log_likelihood_proposal = 
    // ll 
    sv_model_->log_likelihood_ous_integrated_vol(alpha_proposal,
						 rho_proposal,
						 theta_slow_proposal,
						 theta_fast_proposal,
						 tau_square_slow_proposal,
						 tau_square_fast_proposal)
    // priors
    + ou_sampler_slow_.get_ou_model()->
    get_alpha_prior().log_likelihood(alpha_proposal)
    + ou_sampler_slow_.get_ou_model()->
    get_theta_prior().log_likelihood(theta_slow_proposal)
    + ou_sampler_slow_.get_ou_model()->
    get_tau_square_prior().log_likelihood(tau_square_slow_proposal)
    + ou_sampler_fast_.get_ou_model_fast()->
    get_rho_prior().log_likelihood(rho_proposal)
    + ou_sampler_fast_.get_ou_model_fast()->
    get_theta_prior().log_likelihood(theta_fast_proposal)
    + ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square_prior().log_likelihood(tau_square_fast_proposal)
    // transformation determinant
    + theta_tilde_slow_proposal - 2.0*log(exp(theta_tilde_slow_proposal) + 1.0)
    + tau_square_tilde_slow_proposal
    + log(2.0) + rho_tilde_proposal - 2.0*log(exp(rho_tilde_proposal) + 1.0)
    + theta_tilde_fast_proposal - 2.0*log(exp(theta_tilde_fast_proposal) + 1.0)
    + tau_square_tilde_fast_proposal;

  // CALCULATING LOG LIKELIHOOD FOR CURRENT
  double log_likelihood_current = 
    // ll 
    sv_model_->log_likelihood_ous_integrated_vol(alpha_current,
						 rho_current,
						 theta_slow_current,
						 theta_fast_current,
						 tau_square_slow_current,
						 tau_square_fast_current)
    // priors
    + ou_sampler_slow_.get_ou_model()->
    get_alpha_prior().log_likelihood(alpha_current)
    + ou_sampler_slow_.get_ou_model()->
    get_theta_prior().log_likelihood(theta_slow_current)
    + ou_sampler_slow_.get_ou_model()->
    get_tau_square_prior().log_likelihood(tau_square_slow_current)
    + ou_sampler_fast_.get_ou_model_fast()->
    get_rho_prior().log_likelihood(rho_current)
    + ou_sampler_fast_.get_ou_model_fast()->
    get_theta_prior().log_likelihood(theta_fast_current)
    + ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square_prior().log_likelihood(tau_square_fast_current)
    // transformation determinant
    + theta_tilde_slow_current - 2.0*log(exp(theta_tilde_slow_current) + 1.0)
    + tau_square_tilde_slow_current
    + log(2.0) + rho_tilde_current -2*log(exp(rho_tilde_current) + 1.0)
    + theta_tilde_fast_current - 2.0*log(exp(theta_tilde_fast_current) + 1.0)
    + tau_square_tilde_fast_current;

  // ACCEPT / REJECT
  double log_a_acceptance = log_likelihood_proposal -
    log_likelihood_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    double alpha_hat_proposal = alpha_proposal - 0.5*log(sv_model_->get_delta_t());

    // std::cout << "; move accepted \n";
    // std::cout << "alpha_hat_proposal = " << alpha_hat_proposal << "\n";

    double theta_hat_slow_proposal = -1.0*log(theta_slow_proposal) / 
      sv_model_->get_delta_t();

    double tau_square_hat_slow_proposal = 
      tau_square_slow_proposal / 
    ((1.0 - exp(-2.0*theta_hat_slow_proposal*
		sv_model_->get_delta_t()))/(2.0*theta_hat_slow_proposal));

    double theta_hat_fast_proposal = -1.0*log(theta_fast_proposal) / 
      sv_model_->get_delta_t();

    double tau_square_hat_fast_proposal = 
      tau_square_fast_proposal / 
    ((1.0 - exp(-2.0*theta_hat_fast_proposal*
		sv_model_->get_delta_t()))/(2.0*theta_hat_fast_proposal));

    ou_sampler_slow_.get_ou_model()->set_alpha_hat(alpha_hat_proposal);
    ou_sampler_slow_.get_ou_model()->set_tau_square_hat(tau_square_hat_slow_proposal);
    ou_sampler_slow_.get_ou_model()->set_theta_hat(theta_hat_slow_proposal);

    ou_sampler_fast_.get_ou_model_fast()->set_alpha_hat(alpha_hat_proposal);
    ou_sampler_fast_.get_ou_model_fast()->set_tau_square_hat(tau_square_hat_fast_proposal);
    ou_sampler_fast_.get_ou_model_fast()->set_theta_hat(theta_hat_fast_proposal);
    ou_sampler_fast_.get_ou_model_fast()->set_rho(rho_proposal);
  } else {
    double alpha_hat_current = alpha_current - 0.5*log(sv_model_->get_delta_t());

    // std::cout << "; move NOT accepted \n";
    // std::cout << "alpha_hat_current = " << alpha_hat_current << "\n";

    ou_sampler_slow_.get_ou_model()->set_alpha_hat(alpha_hat_current);
    ou_sampler_slow_.get_ou_model()->set_tau_square_hat(tau_square_hat_slow_current);
    ou_sampler_slow_.get_ou_model()->set_theta_hat(theta_hat_slow_current);

    ou_sampler_fast_.get_ou_model_fast()->set_alpha_hat(alpha_hat_current);
    ou_sampler_fast_.get_ou_model_fast()->
      set_tau_square_hat(tau_square_hat_fast_current);
    ou_sampler_fast_.get_ou_model_fast()->set_theta_hat(theta_hat_fast_current);
    ou_sampler_fast_.get_ou_model_fast()->set_rho(rho_current);
  }  
}

void SVWithJumpsPosteriorSampler::draw_sv_models_minus_rho_params_integrated_vol()
{
  // CURRENT MODEL PARAMETERS
  double alpha_current = ou_sampler_fast_.get_ou_model_fast()->
    get_alpha().get_discrete_time_parameter(sv_model_->get_delta_t());

  double theta_slow_current = ou_sampler_slow_.get_ou_model()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double theta_hat_slow_current = ou_sampler_slow_.get_ou_model()->
    get_theta().get_continuous_time_parameter();
  double tau_square_slow_current = ou_sampler_slow_.get_ou_model()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_hat_slow_current = ou_sampler_slow_.get_ou_model()->
    get_tau_square().get_continuous_time_parameter();

  double rho_current = ou_sampler_fast_.get_ou_model_fast()->
    get_rho().get_continuous_time_parameter();
  double theta_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double theta_hat_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_theta().get_continuous_time_parameter();
  double tau_square_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_hat_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square().get_continuous_time_parameter();

  // TRANSFORMING TO TILDE SCALE
  double theta_tilde_slow_current = logit(theta_slow_current);
  double tau_square_tilde_slow_current = log(tau_square_slow_current);
  
  double rho_tilde_current = logit((rho_current+1.0)/2.0);
  double theta_tilde_fast_current = logit(theta_fast_current);
  double tau_square_tilde_fast_current = log(tau_square_fast_current);

  // SETTING THE MEAN
  std::vector<double> mean { alpha_current,
      theta_tilde_slow_current,
      tau_square_tilde_slow_current,
      theta_tilde_fast_current,
      tau_square_tilde_fast_current };

  // PROPOSING ON THE TILDE SCALE
  std::vector<double> proposal = 
    normal_rw_proposal_.propose_parameters(rng_, mean);

  double alpha_proposal = proposal[0];
  double theta_tilde_slow_proposal = proposal[1];
  double tau_square_tilde_slow_proposal = proposal[2];
  double theta_tilde_fast_proposal = proposal[3];
  double tau_square_tilde_fast_proposal = proposal[4];

  // TRANSFORMING TO THE NOMINAL SCALE
  double theta_slow_proposal = exp(theta_tilde_slow_proposal)/
    (exp(theta_tilde_slow_proposal) + 1.0);
  double tau_square_slow_proposal = exp(tau_square_tilde_slow_proposal);
  double theta_fast_proposal = exp(theta_tilde_fast_proposal)/
    (exp(theta_tilde_fast_proposal) + 1.0);
  double tau_square_fast_proposal = exp(tau_square_tilde_fast_proposal);

  // CALCULATING LOG LIKELIHOOD FOR PROPOSAL
  double log_likelihood_proposal = 
    // ll 
    sv_model_->log_likelihood_ous_integrated_vol(alpha_proposal,
						 rho_current,
						 theta_slow_proposal,
						 theta_fast_proposal,
						 tau_square_slow_proposal,
						 tau_square_fast_proposal)
    // priors
    + ou_sampler_slow_.get_ou_model()->
    get_alpha_prior().log_likelihood(alpha_proposal)
    + ou_sampler_slow_.get_ou_model()->
    get_theta_prior().log_likelihood(theta_slow_proposal)
    + ou_sampler_slow_.get_ou_model()->
    get_tau_square_prior().log_likelihood(tau_square_slow_proposal)
    + ou_sampler_fast_.get_ou_model_fast()->
    get_rho_prior().log_likelihood(rho_current)
    + ou_sampler_fast_.get_ou_model_fast()->
    get_theta_prior().log_likelihood(theta_fast_proposal)
    + ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square_prior().log_likelihood(tau_square_fast_proposal)
    // transformation determinant
    + theta_tilde_slow_proposal - 2.0*log(exp(theta_tilde_slow_proposal) + 1.0)
    + tau_square_tilde_slow_proposal
    + log(2.0) + rho_tilde_current - 2.0*log(exp(rho_tilde_current) + 1.0)
    + theta_tilde_fast_proposal - 2.0*log(exp(theta_tilde_fast_proposal) + 1.0)
    + tau_square_tilde_fast_proposal;

  // CALCULATING LOG LIKELIHOOD FOR CURRENT
  double log_likelihood_current = 
    // ll 
    sv_model_->log_likelihood_ous_integrated_vol(alpha_current,
						 rho_current,
						 theta_slow_current,
						 theta_fast_current,
						 tau_square_slow_current,
						 tau_square_fast_current)
    // priors
    + ou_sampler_slow_.get_ou_model()->
    get_alpha_prior().log_likelihood(alpha_current)
    + ou_sampler_slow_.get_ou_model()->
    get_theta_prior().log_likelihood(theta_slow_current)
    + ou_sampler_slow_.get_ou_model()->
    get_tau_square_prior().log_likelihood(tau_square_slow_current)
    + ou_sampler_fast_.get_ou_model_fast()->
    get_rho_prior().log_likelihood(rho_current)
    + ou_sampler_fast_.get_ou_model_fast()->
    get_theta_prior().log_likelihood(theta_fast_current)
    + ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square_prior().log_likelihood(tau_square_fast_current)
    // transformation determinant
    + theta_tilde_slow_current - 2.0*log(exp(theta_tilde_slow_current) + 1.0)
    + tau_square_tilde_slow_current
    + log(2.0) + rho_tilde_current -2*log(exp(rho_tilde_current) + 1.0)
    + theta_tilde_fast_current - 2.0*log(exp(theta_tilde_fast_current) + 1.0)
    + tau_square_tilde_fast_current;

  // ACCEPT / REJECT
  double log_a_acceptance = log_likelihood_proposal -
    log_likelihood_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    double alpha_hat_proposal = alpha_proposal - 0.5*log(sv_model_->get_delta_t());

    // std::cout << "; move accepted \n";
    // std::cout << "alpha_hat_proposal = " << alpha_hat_proposal << "\n";

    double theta_hat_slow_proposal = -1.0*log(theta_slow_proposal) / 
      sv_model_->get_delta_t();

    double tau_square_hat_slow_proposal = 
      tau_square_slow_proposal / 
    ((1.0 - exp(-2.0*theta_hat_slow_proposal*
		sv_model_->get_delta_t()))/(2.0*theta_hat_slow_proposal));

    double theta_hat_fast_proposal = -1.0*log(theta_fast_proposal) / 
      sv_model_->get_delta_t();
    // std::cout << "theta_hat_fast_proposal = " << theta_hat_fast_proposal << "\n";

    double tau_square_hat_fast_proposal = 
      tau_square_fast_proposal / 
    ((1.0 - exp(-2.0*theta_hat_fast_proposal*
		sv_model_->get_delta_t()))/(2.0*theta_hat_fast_proposal));

    ou_sampler_slow_.get_ou_model()->set_alpha_hat(alpha_hat_proposal);
    ou_sampler_slow_.get_ou_model()->set_tau_square_hat(tau_square_hat_slow_proposal);
    ou_sampler_slow_.get_ou_model()->set_theta_hat(theta_hat_slow_proposal);

    ou_sampler_fast_.get_ou_model_fast()->set_alpha_hat(alpha_hat_proposal);
    ou_sampler_fast_.get_ou_model_fast()->set_tau_square_hat(tau_square_hat_fast_proposal);
    ou_sampler_fast_.get_ou_model_fast()->set_theta_hat(theta_hat_fast_proposal);
    ou_sampler_fast_.get_ou_model_fast()->set_rho(rho_current);
  } else {
    double alpha_hat_current = alpha_current - 0.5*log(sv_model_->get_delta_t());

    // double theta_hat_fast_proposal = -1.0*log(theta_fast_proposal) / 
    //   sv_model_->get_delta_t();
    // std::cout << "theta_hat_fast_proposal = " << theta_hat_fast_proposal << "\n";

    // std::cout << "; move NOT accepted \n";
    // std::cout << "alpha_hat_current = " << alpha_hat_current << "\n";

    ou_sampler_slow_.get_ou_model()->set_alpha_hat(alpha_hat_current);
    ou_sampler_slow_.get_ou_model()->set_tau_square_hat(tau_square_hat_slow_current);
    ou_sampler_slow_.get_ou_model()->set_theta_hat(theta_hat_slow_current);

    ou_sampler_fast_.get_ou_model_fast()->set_alpha_hat(alpha_hat_current);
    ou_sampler_fast_.get_ou_model_fast()->
      set_tau_square_hat(tau_square_hat_fast_current);
    ou_sampler_fast_.get_ou_model_fast()->set_theta_hat(theta_hat_fast_current);
    ou_sampler_fast_.get_ou_model_fast()->set_rho(rho_current);
  }  
}

// NOT SAMPLING RHO BE CAREFUL CHECK AND DOUBLE CHECK AND WHEN YOU DO ERASE
void SVWithJumpsPosteriorSampler::draw_sv_models_params_integrated_prices()
{
  // CURRENT MODEL PARAMETERS
  double alpha_current = ou_sampler_fast_.get_ou_model_fast()->
    get_alpha().get_discrete_time_parameter(sv_model_->get_delta_t());

  double theta_slow_current = ou_sampler_slow_.get_ou_model()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double theta_hat_slow_current = ou_sampler_slow_.get_ou_model()->
    get_theta().get_continuous_time_parameter();
  double tau_square_slow_current = ou_sampler_slow_.get_ou_model()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_hat_slow_current = ou_sampler_slow_.get_ou_model()->
    get_tau_square().get_continuous_time_parameter();

  double rho_current = ou_sampler_fast_.get_ou_model_fast()->
    get_rho().get_continuous_time_parameter();
  double theta_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double theta_hat_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_theta().get_continuous_time_parameter();
  double tau_square_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_hat_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square().get_continuous_time_parameter();

  double xi_square_current = observational_model_sampler_.
    get_observational_model()->get_xi_square().get_continuous_time_parameter();

  double mu_current = constant_vol_sampler_.get_constant_vol_model()->
    get_mu().get_discrete_time_parameter(sv_model_->get_delta_t());

  // TRANSFORMING TO TILDE SCALE
  double theta_tilde_slow_current = logit(theta_slow_current);
  double tau_square_tilde_slow_current = log(tau_square_slow_current);
  
  double rho_tilde_current = logit((rho_current+1.0)/2.0);
  double theta_tilde_fast_current = logit(theta_fast_current);
  double tau_square_tilde_fast_current = log(tau_square_fast_current);
  
  // SETTING THE MEAN
  std::vector<double> mean { alpha_current,
      theta_tilde_slow_current,
      tau_square_tilde_slow_current,
      rho_tilde_current,
      theta_tilde_fast_current,
      tau_square_tilde_fast_current};

  // PROPOSING ON THE TILDE SCALE
  std::vector<double> proposal = 
    normal_rw_proposal_.propose_parameters(rng_, mean);

  double alpha_proposal = proposal[0];
  double theta_tilde_slow_proposal = proposal[1];
  double tau_square_tilde_slow_proposal = proposal[2];

  double theta_tilde_fast_proposal = proposal[4];
  double tau_square_tilde_fast_proposal = proposal[5];

  // TRANSFORMING TO THE NOMINAL SCALE
  double theta_slow_proposal = exp(theta_tilde_slow_proposal)/
    (exp(theta_tilde_slow_proposal) + 1.0);
  double tau_square_slow_proposal = exp(tau_square_tilde_slow_proposal);

  double theta_fast_proposal = exp(theta_tilde_fast_proposal)/
    (exp(theta_tilde_fast_proposal) + 1.0);
  double tau_square_fast_proposal = exp(tau_square_tilde_fast_proposal);

  // CALCULATING LOG LIKELIHOOD FOR PROPOSAL
  double log_likelihood_proposal = 
    // ll 
    sv_model_->
    log_likelihood_ous_integrated_filtered_prices(alpha_proposal,
						  rho_current,
						  theta_slow_proposal,
						  theta_fast_proposal,
						  tau_square_slow_proposal,
						  tau_square_fast_proposal,
						  xi_square_current,
						  mu_current)
    // priors
    + ou_sampler_slow_.get_ou_model()->get_alpha_prior().
    log_likelihood(alpha_proposal)
    + ou_sampler_slow_.get_ou_model()->get_theta_prior().
    log_likelihood(theta_slow_proposal)
    + ou_sampler_slow_.get_ou_model()->
    get_tau_square_prior().log_likelihood(tau_square_slow_proposal)
    + ou_sampler_fast_.get_ou_model_fast()->get_rho_prior().
    log_likelihood(rho_current)
    + ou_sampler_fast_.get_ou_model_fast()->get_theta_prior().
    log_likelihood(theta_fast_proposal)
    + ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square_prior().log_likelihood(tau_square_fast_proposal)
    // transformation determinant
    + theta_tilde_slow_proposal - 2.0*log(exp(theta_tilde_slow_proposal) + 1.0)
    + tau_square_tilde_slow_proposal
    + log(2.0) + rho_tilde_current - 2.0*log(exp(rho_tilde_current) + 1.0)
    + theta_tilde_fast_proposal - 2.0*log(exp(theta_tilde_fast_proposal) + 1.0)
    + tau_square_tilde_fast_proposal;

  // CALCULATING LOG LIKELIHOOD FOR CURRENT
  double log_likelihood_current = 
    // ll 
    sv_model_->
    log_likelihood_ous_integrated_filtered_prices(alpha_current,
						  rho_current,
						  theta_slow_current,
						  theta_fast_current,
						  tau_square_slow_current,
						  tau_square_fast_current,
						  xi_square_current,
						  mu_current)
    // priors
    + ou_sampler_slow_.get_ou_model()->get_alpha_prior().
    log_likelihood(alpha_current)
    + ou_sampler_slow_.get_ou_model()->get_theta_prior().
    log_likelihood(theta_slow_current)
    + ou_sampler_slow_.get_ou_model()->
    get_tau_square_prior().log_likelihood(tau_square_slow_current)
    + ou_sampler_fast_.get_ou_model_fast()->get_rho_prior().
    log_likelihood(rho_current)
    + ou_sampler_fast_.get_ou_model_fast()->get_theta_prior().
    log_likelihood(theta_fast_current)
    + ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square_prior().log_likelihood(tau_square_fast_current)
    // transformation determinant
    + theta_tilde_slow_current - 2.0*log(exp(theta_tilde_slow_current) + 1.0)
    + tau_square_tilde_slow_current
    + log(2.0) + rho_tilde_current -2*log(exp(rho_tilde_current) + 1.0)
    + theta_tilde_fast_current - 2.0*log(exp(theta_tilde_fast_current) + 1.0)
    + tau_square_tilde_fast_current;

  // ACCEPT / REJECT
  double log_a_acceptance = log_likelihood_proposal -
    log_likelihood_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    double alpha_hat_proposal = alpha_proposal - 0.5*log(sv_model_->get_delta_t());

    // std::cout << "; move accepted \n";
    // std::cout << "alpha_hat_proposal = " << alpha_hat_proposal << "\n";

    double theta_hat_slow_proposal = -1.0*log(theta_slow_proposal) / 
      sv_model_->get_delta_t();

    double tau_square_hat_slow_proposal = 
      tau_square_slow_proposal / 
    ((1.0 - exp(-2.0*theta_hat_slow_proposal*
		sv_model_->get_delta_t()))/(2.0*theta_hat_slow_proposal));

    double theta_hat_fast_proposal = -1.0*log(theta_fast_proposal) / 
      sv_model_->get_delta_t();

    double tau_square_hat_fast_proposal = 
      tau_square_fast_proposal / 
    ((1.0 - exp(-2.0*theta_hat_fast_proposal*
		sv_model_->get_delta_t()))/(2.0*theta_hat_fast_proposal));

    ou_sampler_slow_.get_ou_model()->
      set_alpha_hat(alpha_hat_proposal);
    ou_sampler_slow_.get_ou_model()->
      set_tau_square_hat(tau_square_hat_slow_proposal);
    ou_sampler_slow_.get_ou_model()->
      set_theta_hat(theta_hat_slow_proposal);

    ou_sampler_fast_.get_ou_model_fast()->
      set_alpha_hat(alpha_hat_proposal);
    ou_sampler_fast_.get_ou_model_fast()->
      set_tau_square_hat(tau_square_hat_fast_proposal);
    ou_sampler_fast_.get_ou_model_fast()->
      set_theta_hat(theta_hat_fast_proposal);
    ou_sampler_fast_.get_ou_model_fast()->
      set_rho(rho_current);
  } else {
    double alpha_hat_current = alpha_current - 0.5*log(sv_model_->get_delta_t());

    // std::cout << "; move NOT accepted \n";
    // std::cout << "alpha_hat_current = " << alpha_hat_current << "\n";

    ou_sampler_slow_.get_ou_model()->
      set_alpha_hat(alpha_hat_current);
    ou_sampler_slow_.get_ou_model()->
      set_tau_square_hat(tau_square_hat_slow_current);
    ou_sampler_slow_.get_ou_model()->
      set_theta_hat(theta_hat_slow_current);

    ou_sampler_fast_.get_ou_model_fast()->
      set_alpha_hat(alpha_hat_current);
    ou_sampler_fast_.get_ou_model_fast()->
      set_tau_square_hat(tau_square_hat_fast_current);
    ou_sampler_fast_.get_ou_model_fast()->
      set_theta_hat(theta_hat_fast_current);
    ou_sampler_fast_.get_ou_model_fast()->
      set_rho(rho_current);
  }  
}

void SVWithJumpsPosteriorSampler::
draw_rho_integrated_prices(double rho_tilde_sd)
{
  // CURRENT MODEL PARAMETERS
  double alpha_current = ou_sampler_fast_.get_ou_model_fast()->
    get_alpha().get_discrete_time_parameter(sv_model_->get_delta_t());

  double theta_slow_current = ou_sampler_slow_.get_ou_model()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_slow_current = ou_sampler_slow_.get_ou_model()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());

  double rho_current = ou_sampler_fast_.get_ou_model_fast()->
    get_rho().get_continuous_time_parameter();
  double theta_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());

  double xi_square_current = observational_model_sampler_.
    get_observational_model()->get_xi_square().get_continuous_time_parameter();

  double mu_current = constant_vol_sampler_.get_constant_vol_model()->
    get_mu().get_discrete_time_parameter(sv_model_->get_delta_t());

  // TRANSFORMING TO TILDE SCALE
  double rho_tilde_current = logit((rho_current+1.0)/2.0);

  // SETTING THE MEAN
  std::vector<double> mean { rho_tilde_current };

  // PROPOSING ON THE TILDE SCALE
  double proposal = mean[0] + gsl_ran_gaussian(rng_, rho_tilde_sd);
  double rho_tilde_proposal = proposal;

  // TRANSFORMING TO THE NOMINAL SCALE
  double rho_proposal = 2.0*exp(rho_tilde_proposal)/
    (exp(rho_tilde_proposal) + 1.0) - 1.0;

  // CALCULATING LOG LIKELIHOOD FOR PROPOSAL
  double log_likelihood_proposal = 
    // ll 
    sv_model_->
    log_likelihood_ous_integrated_filtered_prices(alpha_current,
						  rho_proposal,
						  theta_slow_current,
						  theta_fast_current,
						  tau_square_slow_current,
						  tau_square_fast_current,
						  xi_square_current,
						  mu_current)
    // priors
    + ou_sampler_fast_.get_ou_model_fast()->get_rho_prior().
    log_likelihood(rho_proposal)
    // transformation determinant
    + log(2.0) + rho_tilde_proposal - 2.0*log(exp(rho_tilde_proposal) + 1.0);

  // CALCULATING LOG LIKELIHOOD FOR CURRENT
  double log_likelihood_current = 
    // ll 
    sv_model_->
    log_likelihood_ous_integrated_filtered_prices(alpha_current,
						  rho_current,
						  theta_slow_current,
						  theta_fast_current,
						  tau_square_slow_current,
						  tau_square_fast_current,
						  xi_square_current,
						  mu_current)
    // priors
    + ou_sampler_fast_.get_ou_model_fast()->get_rho_prior().
    log_likelihood(rho_current)
    // transformation determinant
    + log(2.0) + rho_tilde_current -2*log(exp(rho_tilde_current) + 1.0);

  // ACCEPT / REJECT
  double log_a_acceptance = log_likelihood_proposal -
    log_likelihood_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    // std::cout << "; move accepted \n";
    // std::cout << "alpha_hat_proposal = " << alpha_hat_proposal << "\n";
    ou_sampler_fast_.get_ou_model_fast()->
      set_rho(rho_proposal);
  } else {
    // std::cout << "; move NOT accepted \n";
    // std::cout << "alpha_hat_current = " << alpha_hat_current << "\n";
    ou_sampler_fast_.get_ou_model_fast()->
      set_rho(rho_current);
  }  
}

void SVWithJumpsPosteriorSampler::
draw_rho_integrated_prices_MLE()
{
  // CURRENT MODEL PARAMETERS
  double alpha_current = ou_sampler_fast_.get_ou_model_fast()->
    get_alpha().get_discrete_time_parameter(sv_model_->get_delta_t());

  double theta_slow_current = ou_sampler_slow_.get_ou_model()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_slow_current = ou_sampler_slow_.get_ou_model()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());

  double rho_current = ou_sampler_fast_.get_ou_model_fast()->
    get_rho().get_continuous_time_parameter();
  double theta_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());

  double xi_square_current = observational_model_sampler_.
    get_observational_model()->get_xi_square().get_continuous_time_parameter();

  double mu_current = constant_vol_sampler_.get_constant_vol_model()->
    get_mu().get_discrete_time_parameter(sv_model_->get_delta_t());

  // TRANSFORMING TO TILDE SCALE
  double rho_tilde_current = logit((rho_current+1.0)/2.0);

  // SETTING THE MEAN AND VAR ON THE TILDE SCALE
  std::vector<double> mean_var = sv_model_->rho_tilde_MLE_mean_var();

  // PROPOSING ON THE TILDE SCALE
  double nu = 1.0;
  double rho_tilde_proposal =  mean_var[0] + 2*sqrt(mean_var[1])*gsl_ran_tdist(rng_, nu);

  // TRANSFORMING TO THE NOMINAL SCALE
  double rho_proposal = 2.0*exp(rho_tilde_proposal)/
    (exp(rho_tilde_proposal) + 1.0) - 1.0;
  
  // CALCULATING LOG LIKELIHOOD FOR PROPOSAL
  double log_likelihood_proposal = 
    // ll 
    sv_model_->
    log_likelihood_ous_integrated_filtered_prices(alpha_current,
						  rho_proposal,
						  theta_slow_current,
						  theta_fast_current,
						  tau_square_slow_current,
						  tau_square_fast_current,
						  xi_square_current,
						  mu_current)
    // priors
    + ou_sampler_fast_.get_ou_model_fast()->get_rho_prior().
    log_likelihood(rho_proposal)
    // q(current|prposal)
    + log(gsl_ran_tdist_pdf((rho_tilde_current - mean_var[0])/(2.0*sqrt(mean_var[1])), 
			    nu))
    // transformation determinant
    + log(2.0) + rho_tilde_proposal - 2.0*log(exp(rho_tilde_proposal) + 1.0);

  // CALCULATING LOG LIKELIHOOD FOR CURRENT
  double log_likelihood_current = 
    // ll 
    sv_model_->
    log_likelihood_ous_integrated_filtered_prices(alpha_current,
						  rho_current,
						  theta_slow_current,
						  theta_fast_current,
						  tau_square_slow_current,
						  tau_square_fast_current,
						  xi_square_current,
						  mu_current)
    // priors
    + ou_sampler_fast_.get_ou_model_fast()->get_rho_prior().
    log_likelihood(rho_current)
    // q(prposal|current)
    + log(gsl_ran_tdist_pdf((rho_tilde_proposal - mean_var[0])/(2.0*sqrt(mean_var[1])), 
			    nu))
    // transformation determinant
    + log(2.0) + rho_tilde_current -2*log(exp(rho_tilde_current) + 1.0);

  // ACCEPT / REJECT
  double log_a_acceptance = log_likelihood_proposal -
    log_likelihood_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    // std::cout << "; move accepted \n";
    // std::cout << "alpha_hat_proposal = " << alpha_hat_proposal << "\n";
    ou_sampler_fast_.get_ou_model_fast()->
      set_rho(rho_proposal);
  } else {
    // std::cout << "; move NOT accepted \n";
    // std::cout << "alpha_hat_current = " << alpha_hat_current << "\n";
    ou_sampler_fast_.get_ou_model_fast()->
      set_rho(rho_current);
  }  
}

void SVWithJumpsPosteriorSampler::
draw_rho_xi_mu_integrated_prices()
{
  // CURRENT MODEL PARAMETERS
  double alpha_current = ou_sampler_fast_.get_ou_model_fast()->
    get_alpha().get_discrete_time_parameter(sv_model_->get_delta_t());

  double theta_slow_current = ou_sampler_slow_.get_ou_model()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_slow_current = ou_sampler_slow_.get_ou_model()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());

  double rho_current = ou_sampler_fast_.get_ou_model_fast()->
    get_rho().get_continuous_time_parameter();
  double theta_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());

  double xi_square_current = observational_model_sampler_.
    get_observational_model()->get_xi_square().get_continuous_time_parameter();
  double mu_current = constant_vol_sampler_.get_constant_vol_model()->
    get_mu().get_discrete_time_parameter(sv_model_->get_delta_t());
  double mu_hat_current = constant_vol_sampler_.get_constant_vol_model()->
    get_mu().get_continuous_time_parameter();

  // TRANSFORMING TO TILDE SCALE
  double rho_tilde_current = logit((rho_current+1.0)/2.0);
  double xi_square_tilde_current = log(xi_square_current);

  // SETTING THE MEAN
  std::vector<double> mean { rho_tilde_current,
      xi_square_tilde_current,
      mu_current};

  // PROPOSING ON THE TILDE SCALE
  std::vector<double> proposal = 
    normal_rw_proposal_obs_model_params_.propose_parameters(rng_, mean);

  double rho_tilde_proposal = proposal[0];
  double xi_square_tilde_proposal = proposal[1];
  double mu_proposal = proposal[2];
  double mu_hat_proposal = mu_proposal/sv_model_->get_delta_t();

  // TRANSFORMING TO THE NOMINAL SCALE
  double rho_proposal = 2.0*exp(proposal[0])/
    (exp(proposal[0]) + 1.0) - 1.0;
  double xi_square_proposal = exp(proposal[1]);

  // CALCULATING LOG LIKELIHOOD FOR PROPOSAL
  double log_likelihood_proposal = 
    // ll 
    sv_model_->
    log_likelihood_ous_integrated_filtered_prices(alpha_current,
						  rho_proposal,
						  theta_slow_current,
						  theta_fast_current,
						  tau_square_slow_current,
						  tau_square_fast_current,
						  xi_square_proposal,
						  mu_proposal)
    // priors
    + ou_sampler_fast_.get_ou_model_fast()->get_rho_prior().
    log_likelihood(rho_proposal)
    + observational_model_sampler_.get_observational_model()->get_xi_square_prior().
    log_likelihood(xi_square_proposal)
    + constant_vol_sampler_.get_constant_vol_model()->get_mu_prior().
    log_likelihood(mu_proposal)
    // transformation determinant
    + log(2.0) + rho_tilde_proposal - 2.0*log(exp(rho_tilde_proposal) + 1.0)
    + xi_square_tilde_proposal;

  // CALCULATING LOG LIKELIHOOD FOR CURRENT
  double log_likelihood_current = 
    // ll 
    sv_model_->
    log_likelihood_ous_integrated_filtered_prices(alpha_current,
						  rho_current,
						  theta_slow_current,
						  theta_fast_current,
						  tau_square_slow_current,
						  tau_square_fast_current,
						  xi_square_current,
						  mu_current)
    // priors
    + ou_sampler_fast_.get_ou_model_fast()->get_rho_prior().
    log_likelihood(rho_current)
    + observational_model_sampler_.get_observational_model()->get_xi_square_prior().
    log_likelihood(xi_square_current)
    + constant_vol_sampler_.get_constant_vol_model()->get_mu_prior().
    log_likelihood(mu_current)
    // transformation determinant
    + log(2.0) + rho_tilde_current -2*log(exp(rho_tilde_current) + 1.0)
    + xi_square_tilde_current;

  // ACCEPT / REJECT
  double log_a_acceptance = log_likelihood_proposal -
    log_likelihood_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    // std::cout << "; move accepted \n";
    // std::cout << "rho_proposal = " << rho_proposal << "\n";
    // std::cout << "xi_square_proposal = " << xi_square_proposal << "\n";
    // std::cout << "mu_proposal_hat = " << mu_proposal/
    //   sv_model_->get_delta_t() << "\n";

    ou_sampler_fast_.get_ou_model_fast()->
      set_rho(rho_proposal);
    observational_model_sampler_.get_observational_model()->
      set_xi_square(xi_square_proposal);
    constant_vol_sampler_.get_constant_vol_model()->
      set_mu_hat(mu_hat_proposal);
    constant_vol_sampler_.get_constant_vol_model()->
      set_y_star_ds();
  } // else {
  //   // std::cout << "; move NOT accepted \n";
  //   // std::cout << "alpha_hat_current = " << alpha_hat_current << "\n";
  //   ou_sampler_fast_.get_ou_model_fast()->
  //     set_rho(rho_current);
  //   observational_model_sampler_.get_observational_model()->
  //     set_xi_square(xi_square_current);
  //   constant_vol_sampler_.get_constant_vol_model()->
  //     set_mu_hat(mu_current/sv_model_->get_delta_t());
  // }

}

void SVWithJumpsPosteriorSampler::
draw_rho_xi_mu_integrated_prices_MLE()
{
  // CURRENT MODEL PARAMETERS
  double alpha_current = ou_sampler_fast_.get_ou_model_fast()->
    get_alpha().get_discrete_time_parameter(sv_model_->get_delta_t());

  double theta_slow_current = ou_sampler_slow_.get_ou_model()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_slow_current = ou_sampler_slow_.get_ou_model()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());

  double rho_current = ou_sampler_fast_.get_ou_model_fast()->
    get_rho().get_continuous_time_parameter();
  double theta_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());

  double xi_square_current = observational_model_sampler_.
    get_observational_model()->get_xi_square().get_continuous_time_parameter();
  double mu_current = constant_vol_sampler_.get_constant_vol_model()->
    get_mu().get_discrete_time_parameter(sv_model_->get_delta_t());
  double mu_hat_current = constant_vol_sampler_.get_constant_vol_model()->
    get_mu().get_continuous_time_parameter();

  // TRANSFORMING TO TILDE SCALE
  double rho_tilde_current = logit((rho_current+1.0)/2.0);
  double xi_square_tilde_current = log(xi_square_current);

  // SETTING THE MEAN AND RHO VAR ON TILDE SCALE
  std::vector<double> rho_mean_var = sv_model_->
    rho_tilde_xi_square_tilde_mu_MLE_mean_var();

  std::vector<double> mean { rho_mean_var[0],
      rho_mean_var[1],
      rho_mean_var[2]};

  std::vector<double> current {rho_tilde_current,
      xi_square_tilde_current,
      mu_current};

  gsl_matrix * proposal_covariance_matrix_ptr = gsl_matrix_alloc(3,3);
  gsl_matrix_set_zero(proposal_covariance_matrix_ptr);
  for (int i=0; i<3; ++i) {
    gsl_matrix_set(proposal_covariance_matrix_ptr, i, i,
		   rho_mean_var[i+3]);
  }

  std::cout << "rho_tilde_mean = " << rho_mean_var[0] << "\n";
  std::cout << "xi2_tilde_mean = " << rho_mean_var[1] << "\n";
  std::cout << "mu_mean = " << rho_mean_var[2] << "\n";
  std::cout << "rho_tilde_proposal_cov = " << gsl_matrix_get(proposal_covariance_matrix_ptr, 0, 0) << "\n";
  std::cout << "xi2_tilde_proposal_cov = " << gsl_matrix_get(proposal_covariance_matrix_ptr, 1, 1) << "\n";
  std::cout << "mu_proposal_cov = " << gsl_matrix_get(proposal_covariance_matrix_ptr, 2, 2) << "\n";

  // PROPOSING ON THE TILDE SCALE
  int dof = 1;
  std::vector<double> proposal =
    normal_rw_proposal_obs_model_params_.propose_parameters(rng_,
							    3,
							    mean,
							    proposal_covariance_matrix_ptr,
							    dof);
  std::cout << "rho_tilde_proposal = " << proposal[0] << std::endl;
  std::cout << "xi_tilde_proposal = " << proposal[1] << std::endl;
  std::cout << "mu_proposal = " << proposal[2] << std::endl;
  std::cout << std::endl;
  
  double rho_tilde_proposal = proposal[0];
  double xi_square_tilde_proposal = proposal[1];
  double mu_proposal = proposal[2];
  double mu_hat_proposal = mu_proposal/sv_model_->get_delta_t();

  // TRANSFORMING TO THE NOMINAL SCALE
  double rho_proposal = 2.0*exp(proposal[0])/
    (exp(proposal[0]) + 1.0) - 1.0;
  double xi_square_proposal = exp(proposal[1]);

  std::cout << "rho_proposal = " << rho_proposal << "\n";
  std::cout << "xi2_proposal = " << xi_square_proposal << "\n";
  std::cout << "mu_hat_proposal = " << mu_proposal/sv_model_->get_delta_t() << "\n";

  // CALCULATING LOG LIKELIHOOD FOR PROPOSAL
  double log_likelihood_proposal =
    // ll
    sv_model_->
    log_likelihood_ous_integrated_filtered_prices(rho_proposal,
						  xi_square_proposal,
						  mu_proposal)
    // priors
    + ou_sampler_fast_.get_ou_model_fast()->get_rho_prior().
    log_likelihood(rho_proposal)
    + observational_model_sampler_.get_observational_model()->get_xi_square_prior().
    log_likelihood(xi_square_proposal)
    + constant_vol_sampler_.get_constant_vol_model()->get_mu_prior().
    log_likelihood(mu_proposal)
    // q(current | proposal)
    + dmvt_log(3, current, mean, proposal_covariance_matrix_ptr, dof)
    // transformation determinant
    + log(2.0) + rho_tilde_proposal - 2.0*log(exp(rho_tilde_proposal) + 1.0)
    + xi_square_tilde_proposal;

  // CALCULATING LOG LIKELIHOOD FOR CURRENT
  double log_likelihood_current =
    // ll
    sv_model_->
    log_likelihood_ous_integrated_filtered_prices(rho_current,
						  xi_square_current,
						  mu_current)
    // priors
    + ou_sampler_fast_.get_ou_model_fast()->get_rho_prior().
    log_likelihood(rho_current)
    + observational_model_sampler_.get_observational_model()->get_xi_square_prior().
    log_likelihood(xi_square_current)
    + constant_vol_sampler_.get_constant_vol_model()->get_mu_prior().
    log_likelihood(mu_current)
    // q(proposal | current)
    + dmvt_log(3, proposal, mean, proposal_covariance_matrix_ptr, dof)
    // transformation determinant
    + log(2.0) + rho_tilde_current -2*log(exp(rho_tilde_current) + 1.0)
    + xi_square_tilde_current;

  // ACCEPT / REJECT
  std::cout << "ll_prop = " << log_likelihood_proposal
	    << "; ll_cur = " << log_likelihood_current
	    << "; ll_prop - ll_cur = " << log_likelihood_proposal - log_likelihood_current
	    << "; q(c|p) = " << dmvnorm_log(3, current, mean, proposal_covariance_matrix_ptr)
	    << "; q(p|c) = " << dmvnorm_log(3, proposal, mean, proposal_covariance_matrix_ptr)
	    << "\n";

  double log_a_acceptance = log_likelihood_proposal -
    log_likelihood_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    // std::cout << "; move accepted \n";
    // std::cout << "rho_proposal = " << rho_proposal << "\n";
    // std::cout << "xi_square_proposal = " << xi_square_proposal << "\n";
    // std::cout << "mu_proposal_hat = " << mu_proposal/
    //   sv_model_->get_delta_t() << "\n";
    ou_sampler_fast_.get_ou_model_fast()->
      set_rho(rho_proposal);
    observational_model_sampler_.get_observational_model()->
      set_xi_square(xi_square_proposal);
    constant_vol_sampler_.get_constant_vol_model()->
      set_mu_hat(mu_hat_proposal);
    constant_vol_sampler_.get_constant_vol_model()->
      set_y_star_ds();
  } else {
    // std::cout << "; move NOT accepted \n";
    // std::cout << "alpha_hat_current = " << alpha_hat_current << "\n";
    ou_sampler_fast_.get_ou_model_fast()->
      set_rho(rho_current);
    observational_model_sampler_.get_observational_model()->
      set_xi_square(xi_square_current);
    constant_vol_sampler_.get_constant_vol_model()->
      set_mu_hat(mu_current/sv_model_->get_delta_t());
  }
  gsl_matrix_free(proposal_covariance_matrix_ptr);
}

void SVWithJumpsPosteriorSampler::
draw_xi_mu_integrated_prices()
{
  // CURRENT MODEL PARAMETERS
  double alpha_current = ou_sampler_fast_.get_ou_model_fast()->
    get_alpha().get_discrete_time_parameter(sv_model_->get_delta_t());

  double theta_slow_current = ou_sampler_slow_.get_ou_model()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_slow_current = ou_sampler_slow_.get_ou_model()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());

  double rho_current = ou_sampler_fast_.get_ou_model_fast()->
    get_rho().get_continuous_time_parameter();
  double theta_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());

  double xi_square_current = observational_model_sampler_.
    get_observational_model()->get_xi_square().get_continuous_time_parameter();
  double mu_current = constant_vol_sampler_.get_constant_vol_model()->
    get_mu().get_discrete_time_parameter(sv_model_->get_delta_t());
  double mu_hat_current = constant_vol_sampler_.get_constant_vol_model()->
    get_mu().get_continuous_time_parameter();

  // TRANSFORMING TO TILDE SCALE
  double rho_tilde_current = logit((rho_current+1.0)/2.0);
  double xi_square_tilde_current = log(xi_square_current);

  // SETTING THE MEAN
  std::vector<double> mean { rho_tilde_current,
      xi_square_tilde_current,
      mu_current};

  // PROPOSING ON THE TILDE SCALE
  std::vector<double> proposal = 
    normal_rw_proposal_obs_model_params_.propose_parameters(rng_, mean);

  double rho_tilde_proposal = proposal[0];
  double xi_square_tilde_proposal = proposal[1];
  double mu_proposal = proposal[2];
  double mu_hat_proposal = mu_proposal/sv_model_->get_delta_t();

  // TRANSFORMING TO THE NOMINAL SCALE
  double rho_proposal = 2.0*exp(proposal[0])/
    (exp(proposal[0]) + 1.0) - 1.0;
  double xi_square_proposal = exp(proposal[1]);

  // CALCULATING LOG LIKELIHOOD FOR PROPOSAL
  double log_likelihood_proposal = 
    // ll 
    sv_model_->
    log_likelihood_ous_integrated_filtered_prices(alpha_current,
						  rho_current,
						  theta_slow_current,
						  theta_fast_current,
						  tau_square_slow_current,
						  tau_square_fast_current,
						  xi_square_proposal,
						  mu_proposal)
    // priors
    + ou_sampler_fast_.get_ou_model_fast()->get_rho_prior().
    log_likelihood(rho_current)
    + observational_model_sampler_.get_observational_model()->get_xi_square_prior().
    log_likelihood(xi_square_proposal)
    + constant_vol_sampler_.get_constant_vol_model()->get_mu_prior().
    log_likelihood(mu_proposal)
    // transformation determinant
    + log(2.0) + rho_tilde_current - 2.0*log(exp(rho_tilde_current) + 1.0)
    + xi_square_tilde_proposal;

  // CALCULATING LOG LIKELIHOOD FOR CURRENT
  double log_likelihood_current = 
    // ll 
    sv_model_->
    log_likelihood_ous_integrated_filtered_prices(alpha_current,
						  rho_current,
						  theta_slow_current,
						  theta_fast_current,
						  tau_square_slow_current,
						  tau_square_fast_current,
						  xi_square_current,
						  mu_current)
    // priors
    + ou_sampler_fast_.get_ou_model_fast()->get_rho_prior().
    log_likelihood(rho_current)
    + observational_model_sampler_.get_observational_model()->get_xi_square_prior().
    log_likelihood(xi_square_current)
    + constant_vol_sampler_.get_constant_vol_model()->get_mu_prior().
    log_likelihood(mu_current)
    // transformation determinant
    + log(2.0) + rho_tilde_current -2*log(exp(rho_tilde_current) + 1.0)
    + xi_square_tilde_current;

  // ACCEPT / REJECT
  double log_a_acceptance = log_likelihood_proposal -
    log_likelihood_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    // std::cout << "; move accepted \n";
    // std::cout << "rho_proposal = " << rho_proposal << "\n";
    // std::cout << "xi_square_proposal = " << xi_square_proposal << "\n";
    // std::cout << "mu_proposal_hat = " << mu_proposal/
    //   sv_model_->get_delta_t() << "\n";

    ou_sampler_fast_.get_ou_model_fast()->
      set_rho(rho_current);
    observational_model_sampler_.get_observational_model()->
      set_xi_square(xi_square_proposal);
    constant_vol_sampler_.get_constant_vol_model()->
      set_mu_hat(mu_hat_proposal);
    constant_vol_sampler_.get_constant_vol_model()->
      set_y_star_ds();
  } // else {
  //   // std::cout << "; move NOT accepted \n";
  //   // std::cout << "alpha_hat_current = " << alpha_hat_current << "\n";
  //   ou_sampler_fast_.get_ou_model_fast()->
  //     set_rho(rho_current);
  //   observational_model_sampler_.get_observational_model()->
  //     set_xi_square(xi_square_current);
  //   constant_vol_sampler_.get_constant_vol_model()->
  //     set_mu_hat(mu_current/sv_model_->get_delta_t());
  // }

}

void SVWithJumpsPosteriorSampler::draw_nu_integrated_deltas()
{
  double nu_current = observational_model_sampler_.
    get_observational_model()->get_nu();
  const NuPrior& nu_prior = observational_model_sampler_.
    get_observational_model()->get_nu_prior();
  
  long unsigned int number_nus = 
    nu_prior.get_nu_max() -
    nu_prior.get_nu_min() + 1;

  double nu_proposal = 0;
  if (nu_current == nu_prior.get_nu_min()) {
    // std::cout << "case 1; ";
    nu_proposal = nu_current + 
      gsl_rng_uniform_int(rng_, 2);
  } else if (nu_current == nu_prior.get_nu_max()) {
    // std::cout << "case 2; ";
    nu_proposal = nu_current + 
      (gsl_rng_uniform_int(rng_, 2) - 1);
  } else {
    double rand = gsl_rng_uniform_int(rng_, 3);
    nu_proposal = nu_current + (rand - 1);
    // std::cout << "case 3; rand = " << rand << "; ";
  }

  // std::cout << "nu_current = " << nu_current << "; ";
  // std::cout << "nu_proposal = " << nu_proposal << "\n";

  const std::vector<double>& filtered_log_prices = observational_model_sampler_.
    get_observational_model()->get_filtered_log_prices();

  double xi_square = observational_model_sampler_.
    get_observational_model()->get_xi_square().
    get_continuous_time_parameter();

  double log_P_t = 0;
  double ll_current = 0;
  double ll_proposal = 0;

  for (unsigned i=0; 
       i<=observational_model_sampler_.get_observational_model()->data_length(); 
       ++i) {
    if (i==0) {
      log_P_t = (sv_model_->
		 get_observational_model()->
		 get_data_element(i).get_open());

      ll_current = ll_current +
	log(gsl_ran_tdist_pdf((log_P_t - filtered_log_prices[i])/sqrt(xi_square), 
			      nu_current));

      ll_proposal = ll_proposal +
	log(gsl_ran_tdist_pdf((log_P_t - filtered_log_prices[i])/sqrt(xi_square), 
			      nu_proposal));
    } else {
      log_P_t = (sv_model_->
		 get_observational_model()->
		 get_data_element(i-1).get_close());

      ll_current = ll_current +
	log(gsl_ran_tdist_pdf((log_P_t - filtered_log_prices[i])/sqrt(xi_square), 
			      nu_current));

      ll_proposal = ll_proposal +
	log(gsl_ran_tdist_pdf((log_P_t - filtered_log_prices[i])/sqrt(xi_square), 
			      nu_proposal));
    }
  }

  ll_current = ll_current + 
    nu_prior.log_likelihood(nu_current);
  ll_proposal = ll_proposal + 
    nu_prior.log_likelihood(nu_proposal);

  double log_a_acceptance = ll_proposal -
    ll_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    observational_model_sampler_.
    get_observational_model()->set_nu(nu_proposal);
    // std::cout << "nu accepted: " << nu_proposal << "\n";
  } else {
    // std::cout << "nu NOT accepted: " << nu_proposal << "\n";
    observational_model_sampler_.
    get_observational_model()->set_nu(nu_current);
  }
}

void SVWithJumpsPosteriorSampler::draw_sv_fast_params()
{
  double alpha = ou_sampler_slow_.get_ou_model()->
    get_alpha().get_discrete_time_parameter(sv_model_->get_delta_t());

  // CURRENT MODEL PARAMETERS
  double theta_slow_current = ou_sampler_slow_.get_ou_model()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_slow_current = ou_sampler_slow_.get_ou_model()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());

  double rho_current = ou_sampler_fast_.get_ou_model_fast()->
    get_rho().get_continuous_time_parameter();
  double theta_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_theta().get_discrete_time_parameter(sv_model_->get_delta_t());
  double theta_hat_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_theta().get_continuous_time_parameter();
  double tau_square_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square().get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_hat_fast_current = ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square().get_continuous_time_parameter();

  // TRANSFORMING TO TILDE SCALE
  double theta_tilde_slow_current = logit(theta_slow_current);
  double tau_square_tilde_slow_current = log(tau_square_slow_current);
  
  double rho_tilde_current = logit((rho_current+1.0)/2.0);
  double theta_tilde_fast_current = logit(theta_fast_current);
  double tau_square_tilde_fast_current = log(tau_square_fast_current);

  // SETTING THE MEAN
  std::vector<double> mean 
  { rho_tilde_current,
      theta_tilde_fast_current,
      tau_square_tilde_fast_current };

  // PROPOSING ON THE TILDE SCALE
  NormalRWProposal proposal_obj =
    NormalRWProposal(3,
		     ou_sampler_fast_.get_proposal().get_proposal_covariance_matrix());

  std::vector<double> proposal =
    proposal_obj.propose_parameters(rng_, mean);

  double rho_tilde_proposal = proposal[0];
  double theta_tilde_fast_proposal = proposal[1];
  double tau_square_tilde_fast_proposal = proposal[2];

  // std::cout << rho_tilde_proposal << ","
  // 	    << theta_tilde_fast_proposal << ","
  // 	    << tau_square_tilde_fast_proposal << "\n";

  // TRANSFORMING TO THE NOMINAL SCALE
  double rho_proposal = 2.0*exp(rho_tilde_proposal)/
    (exp(rho_tilde_proposal) + 1.0) - 1.0;
  double theta_fast_proposal = exp(theta_tilde_fast_proposal)/
    (exp(theta_tilde_fast_proposal) + 1.0);
  double tau_square_fast_proposal = exp(tau_square_tilde_fast_proposal);

  // CALCULATING LOG LIKELIHOOD FOR PROPOSAL
  double log_likelihood_proposal =
    // ll
    sv_model_->log_likelihood_ous_integrated_vol(alpha,
						 rho_proposal,
						 theta_slow_current,
						 theta_fast_proposal,
						 tau_square_slow_current,
						 tau_square_fast_proposal)
    // priors
    + ou_sampler_fast_.get_ou_model_fast()->get_alpha_prior().log_likelihood(alpha)
    + ou_sampler_fast_.get_ou_model_fast()->get_rho_prior().log_likelihood(rho_proposal)
    + ou_sampler_fast_.get_ou_model_fast()->get_theta_prior().log_likelihood(theta_fast_proposal)
    + ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square_prior().log_likelihood(tau_square_fast_proposal)
    // transformation determinant
    + theta_tilde_slow_current - 2.0*log(exp(theta_tilde_slow_current) + 1.0)
    + tau_square_tilde_slow_current
    + log(2.0) + rho_tilde_proposal -2*log(exp(rho_tilde_proposal) + 1.0)
    + theta_tilde_fast_proposal - 2.0*log(exp(theta_tilde_fast_proposal) + 1.0)
    + tau_square_tilde_fast_proposal;

  // CALCULATING LOG LIKELIHOOD FOR CURRENT
  double log_likelihood_current = 
    // ll 
    sv_model_->log_likelihood_ous_integrated_vol(alpha,
						 rho_current,
						 theta_slow_current,
						 theta_fast_current,
						 tau_square_slow_current,
						 tau_square_fast_current)
    // priors
    + ou_sampler_fast_.get_ou_model_fast()->get_alpha_prior().log_likelihood(alpha)
    + ou_sampler_fast_.get_ou_model_fast()->get_rho_prior().log_likelihood(rho_current)
    + ou_sampler_fast_.get_ou_model_fast()->get_theta_prior().log_likelihood(theta_fast_current)
    + ou_sampler_fast_.get_ou_model_fast()->
    get_tau_square_prior().log_likelihood(tau_square_fast_current)
    // transformation determinant
    + theta_tilde_slow_current - 2.0*log(exp(theta_tilde_slow_current) + 1.0)
    + tau_square_tilde_slow_current
    + log(2.0) + rho_tilde_current -2*log(exp(rho_tilde_current) + 1.0)
    + theta_tilde_fast_current - 2.0*log(exp(theta_tilde_fast_current) + 1.0)
    + tau_square_tilde_fast_current;

  // std::cout << "log_likelihood_proposal = " << log_likelihood_proposal << "\n";
  // std::cout << "log_likelihood_current = " << log_likelihood_current << "\n";

  // ACCEPT / REJECT
  double log_a_acceptance = log_likelihood_proposal -
    log_likelihood_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    //    std::cout << "; move accepted ";
    double theta_hat_fast_proposal = -1.0*log(theta_fast_proposal) / 
      sv_model_->get_delta_t();

    double tau_square_hat_fast_proposal = 
      tau_square_fast_proposal / 
    ((1.0 - exp(-2.0*theta_hat_fast_proposal*
		sv_model_->get_delta_t()))/(2.0*theta_hat_fast_proposal));

    ou_sampler_fast_.get_ou_model_fast()->set_tau_square_hat(tau_square_hat_fast_proposal);
    ou_sampler_fast_.get_ou_model_fast()->set_theta_hat(theta_hat_fast_proposal);
    ou_sampler_fast_.get_ou_model_fast()->set_rho(rho_proposal);
  } else {
    ou_sampler_fast_.get_ou_model_fast()->set_tau_square_hat(tau_square_hat_fast_current);
    ou_sampler_fast_.get_ou_model_fast()->set_theta_hat(theta_hat_fast_current);
    ou_sampler_fast_.get_ou_model_fast()->set_rho(rho_current);
  }  
}


void SVWithJumpsPosteriorSampler::draw_sigmas() 
{
  double theta_slow = sv_model_->get_ou_model_slow()->get_theta().
    get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_slow = sv_model_->get_ou_model_slow()->get_tau_square().
    get_discrete_time_parameter(sv_model_->get_delta_t());

  double rho = sv_model_->get_ou_model_fast()->get_rho().
    get_continuous_time_parameter();
  double theta_fast = sv_model_->get_ou_model_fast()->get_theta().
    get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_fast = sv_model_->get_ou_model_fast()->get_tau_square().
    get_discrete_time_parameter(sv_model_->get_delta_t());

  std::vector<arma::mat> taus_squared_inv (sv_model_->data_length());
  std::vector<arma::vec> vs (sv_model_->data_length());

  const std::vector<double>& m = 
    sv_model_->get_constant_vol_model()->
    get_gammas().get_mixture_means();
  const std::vector<double>& v_square = 
    sv_model_->get_constant_vol_model()->
    get_gammas().get_mixture_variances();
  const std::vector<int>& ds = 
    sv_model_->get_constant_vol_model()->
    get_ds();
  const std::vector<int>& gammas = 
    sv_model_->get_constant_vol_model()->
    get_gammas().get_gammas();
  const std::vector<double>& y_star = 
    sv_model_->get_constant_vol_model()->
    get_y_star();

  const std::vector<double>& bs_correction = 
    sv_model_->get_constant_vol_model()->
    get_bs();

  const std::vector<double>& as_correction = 
    sv_model_->get_constant_vol_model()->
    get_as();

  // const std::vector<double>& h_fast =
  //   ou_model_fast_->get_sigmas().get_discrete_time_log_sigmas();
  // const std::vector<double>& h_slow =
  //   ou_model_slow_->get_sigmas().get_discrete_time_log_sigmas();

  double alpha = 
    sv_model_->get_ou_model_fast()->get_alpha().
    get_discrete_time_parameter(sv_model_->get_delta_t());

  // FORWARD FILTER
  // current mean level;
  double u_current_fast = 
    sv_model_->get_ou_model_fast()->
    get_alpha().get_discrete_time_parameter(sv_model_->get_delta_t());
  double u_current_slow =
    sv_model_->get_ou_model_slow()->
    get_alpha().get_discrete_time_parameter(sv_model_->get_delta_t());

  arma::vec u_current(2);
  u_current(0) = u_current_slow;
  u_current(1) = u_current_fast;

  // current variance level;
  double s_current_sq_fast = 
    tau_square_fast / 
    (1 - square(theta_fast));

  double s_current_sq_slow = 
    tau_square_slow / 
    (1 - square(theta_slow));

  arma::mat S_current_sq = arma::zeros<arma::mat> (2,2);
  S_current_sq(0,0) = s_current_sq_slow;
  S_current_sq(1,1) = s_current_sq_fast;

  // freed at end of loop
  arma::vec m_t = arma::vec (2);
  arma::mat Tau_square_current = arma::mat (2,2);
  arma::vec v_current  = arma::vec (2);

  arma::mat G_j_minus_one = arma::zeros<arma::mat> (2,2);
  arma::mat G_j = arma::zeros<arma::mat> (2,2);

  arma::mat F = arma::mat (2,1);
  F(0,0) = 0.5;
  F(1,0) = 0.5;
  arma::mat FFt = F*F.t();

  arma::mat Z = arma::zeros<arma::mat> (2,2);
  Z(0,0) = tau_square_slow;
  Z(1,1) = tau_square_fast * (1-square(rho));
  // std::cout << "tau_square_fast = " << tau_square_fast << "\n";
  // std::cout << "Z=" << Z << "\n";
  arma::mat Z_inv = inv(Z);

  double theta_j_minus_one_slow = 0;
  double theta_j_minus_one_fast = 0;
  double alpha_j_minus_one = 0;
  arma::mat S_square_current_inv = arma::zeros<arma::mat> (2,2);
  arma::mat S_square_current = arma::zeros<arma::mat> (2,2);
  S_square_current_inv(0,0) = 1.0/s_current_sq_slow;
  S_square_current_inv(1,1) = 1.0/s_current_sq_fast;

  arma::mat Tau_current_sq_inv = arma::zeros<arma::mat> (2,2);
  arma::mat Tau_current_sq = arma::zeros<arma::mat> (2,2);
  arma::mat omega_ts_sq_inv = arma::mat (2,2);

  arma::mat L (2,2);
  arma::mat U (2,2);
  arma::mat Q (2,2);
  arma::mat R (2,2);
  arma::mat X (2,2);
  arma::vec y (2);
  arma::vec b (2);
  arma::vec epsilon (2);
  arma::mat I (2,2, arma::fill::eye);

  // FORWARD FILTER
  for (unsigned i=0; i<sv_model_->data_length(); ++i) {
    if (i==0) {
      Tau_current_sq_inv = 
	S_square_current_inv + (FFt)/(v_square[gammas[i]]/4.0);

      X(0,0) = Tau_current_sq_inv(1,1);
      X(0,1) = -1.0*Tau_current_sq_inv(0,1);
      X(1,1) = Tau_current_sq_inv(0,0);
      X(1,0) = -1.0*Tau_current_sq_inv(1,0);
      Tau_current_sq = 1.0/(Tau_current_sq_inv(0,0)*Tau_current_sq_inv(1,1) -
			    Tau_current_sq_inv(0,1)*Tau_current_sq_inv(1,0)) * 
	X;

      v_current = Tau_current_sq * 
	(S_square_current_inv*u_current + 
	 F*(y_star[i]-m[gammas[i]]/2.0) / (v_square[gammas[i]]/4.0));

    } else {
      theta_j_minus_one_slow =
	-rho*sqrt(tau_square_fast)*ds[i-1]*exp(m[gammas[i-1]]/2.0)*
	bs_correction[gammas[i-1]];

      theta_j_minus_one_fast = 
	theta_fast - rho*sqrt(tau_square_fast)*ds[i-1]*exp(m[gammas[i-1]]/2.0)*
	bs_correction[gammas[i-1]];
      
      alpha_j_minus_one = 
	alpha*(1-theta_fast)
	+ rho*sqrt(tau_square_fast)*ds[i-1]*exp(m[gammas[i-1]]/2.0)
	* (as_correction[gammas[i-1]] + bs_correction[gammas[i-1]]
	   * 2.0 * (y_star[i-1] - m[gammas[i-1]]/2.0));
      
      m_t(0) = alpha*(1-theta_slow);
      m_t(1) = alpha_j_minus_one;
      
      G_j_minus_one(0, 0) = theta_slow;
      G_j_minus_one(1, 0) = theta_j_minus_one_slow;
      G_j_minus_one(1, 1) = theta_j_minus_one_fast;

      // u_{i+1} = u_j
      u_current = G_j_minus_one * v_current + m_t;
      
      // s^2_{i+1} = s^2_{j}; j goes from 1 to n
      S_square_current = 
	G_j_minus_one * Tau_current_sq * G_j_minus_one.t() + Z;

      // vs, taus
      // ========= inversion by hand ==================
      X(0,0) = S_square_current(1,1);
      X(0,1) = -1.0*S_square_current(0,1);
      X(1,1) = S_square_current(0,0);
      X(1,0) = -1.0*S_square_current(1,0);
      S_square_current_inv = 1.0/(S_square_current(0,0)*S_square_current(1,1) -
				  S_square_current(0,1)*S_square_current(1,0)) * 
	X;
      // ==============================================
      
      // lu(L,U,s_square_current);
      // X = solve(trimatl(L),I);
      // s_square_current_inv = solve(trimatu(U),X);
      
      // tau_current_sq
      Tau_current_sq_inv = 
	S_square_current_inv + 
	(FFt)/(v_square[gammas[i]]/4.0);

      // // std::cout << "inverting tau_current_sq\n";
      // tau_current_sq = 
      //   inv_sympd(s_square_current_inv + 
      // 		(FFt)/(v_square[gammas[i]]/4.0));
      // ========= inversion by hand ==================
      X(0,0) = Tau_current_sq_inv(1,1);
      X(0,1) = -1.0*Tau_current_sq_inv(0,1);
      X(1,1) = Tau_current_sq_inv(0,0);
      X(1,0) = -1.0*Tau_current_sq_inv(1,0);
      Tau_current_sq = 1.0/(Tau_current_sq_inv(0,0)*Tau_current_sq_inv(1,1) -
			    Tau_current_sq_inv(0,1)*Tau_current_sq_inv(1,0)) * 
	X;
      // ==============================================
      
      // v_current 
      v_current = Tau_current_sq * 
	(S_square_current_inv*u_current + 
	 F*(y_star[i]-m[gammas[i]]/2.0) / (v_square[gammas[i]]/4.0));
      // std::cout << "v_current 1 = " << v_current << "\n";
      
      // y = (s_square_current_inv*u_current + F*(y_star[i]-m[gammas[i]]/2.0) / 
      // 	 (v_square[gammas[i]]/4.0));
      // qr(Q,R,tau_current_sq_inv);
      // X = solve(trimatl(L),I);
      // tau_current_sq = solve(trimatu(U),X);
      // v_current = tau_current_sq * y;
      // v_current = solve(trimatu(R), Q.t()*y);
      // // std::cout << "v_current 2 = " << v_current << "\n";
    }

    vs[i] = v_current;
    taus_squared_inv[i] = Tau_current_sq_inv;
  }

  // JUST ONE MORE STEP FORWARD TO THE u_{t+1} and S_{t+1}^2
  theta_j_minus_one_slow =
    -rho*sqrt(tau_square_fast)*ds[sv_model_->data_length()-1]
    *exp(m[gammas[sv_model_->data_length()-1]]/2.0)*
    bs_correction[gammas[sv_model_->data_length()-1]];
  
  theta_j_minus_one_fast = 
    theta_fast - rho*sqrt(tau_square_fast)*ds[sv_model_->data_length()-1]
    *exp(m[gammas[sv_model_->data_length()-1]]/2.0)*
    bs_correction[gammas[sv_model_->data_length()-1]];
      
  alpha_j_minus_one = 
    alpha*(1-theta_fast)
    + rho*sqrt(tau_square_fast)*ds[sv_model_->data_length()-1]
    * exp(m[gammas[sv_model_->data_length()-1]]/2.0)
    * (as_correction[gammas[sv_model_->data_length()-1]] 
       + bs_correction[gammas[sv_model_->data_length()-1]]
       * 2.0 * (y_star[sv_model_->data_length()-1] 
		- m[gammas[sv_model_->data_length()-1]]/2.0));
  
  m_t(0) = alpha*(1-theta_slow);
  m_t(1) = alpha_j_minus_one;
  
  G_j_minus_one(0, 0) = theta_slow;
  G_j_minus_one(1, 0) = theta_j_minus_one_slow;
  G_j_minus_one(1, 1) = theta_j_minus_one_fast;
  
  // u_{i+1} = u_j
  u_current = G_j_minus_one * v_current + m_t;
  
  // s^2_{i+1} = s^2_{j}; j goes from 1 to n
  S_square_current = 
    G_j_minus_one * Tau_current_sq * G_j_minus_one.t() + Z;

  // FIRST POSTERIOR SAMPLE, ie LAST h_{T+1}
  arma::vec h_tpone = rmvnorm(rng_, 2, u_current, S_square_current);

  sv_model_->get_ou_model_slow()
    ->set_sigmas_element(sv_model_->data_length(),
  			 exp(h_tpone(0)) / sqrt(sv_model_->get_delta_t()),
  			 h_tpone(0));

  sv_model_->get_ou_model_fast()
    ->set_sigmas_element(sv_model_->data_length(),
  			 exp(h_tpone(1)) / sqrt(sv_model_->get_delta_t()),
  			 h_tpone(1));
  
  double theta_j_slow = 0;
  double theta_j_fast = 0;
  double alpha_j = 0;
  arma::mat omega_ts_sq = arma::zeros<arma::mat> (2,2);
  arma::vec q_t = arma::zeros<arma::vec> (2);

  for (std::vector<int>::size_type i = sv_model_->data_length()-1;
       i != (std::vector<int>::size_type)-1; --i) {
    
    theta_j_slow =
      -rho*sqrt(tau_square_fast)*ds[i]*exp(m[gammas[i]]/2.0)*
      bs_correction[gammas[i]];
    
    theta_j_fast = 
      theta_fast - rho*sqrt(tau_square_fast)*ds[i]*exp(m[gammas[i]]/2.0)*
      bs_correction[gammas[i]];
    
    alpha_j = 
	alpha*(1-theta_fast)
      + rho*sqrt(tau_square_fast)*ds[i]*exp(m[gammas[i]]/2.0)
      * (as_correction[gammas[i]] + bs_correction[gammas[i]]
	 * 2.0 * (y_star[i] - m[gammas[i]]/2.0));
    
    m_t(0) = alpha*(1-theta_slow);
    m_t(1) = alpha_j;
    
    G_j(0, 0) = theta_slow;
    G_j(1, 0) = theta_j_slow;
    G_j(1, 1) = theta_j_fast;
    
    // //    std::cout << "inverting omega_ts_sq\n";
    // omega_ts_sq_inv = (G_j.t() * Z_inv * G_j + taus_squared_inv[i]);
    // b = (taus_squared_inv[i]*vs[i] + G_j.t()*Z_inv*(h_tpone-m_t));
    // epsilon(0) = gsl_ran_gaussian(rng_, 1.0);
    // epsilon(1) = gsl_ran_gaussian(rng_, 1.0);
    // lu(L,U,omega_ts_sq_inv);
    // y = solve(trimatl(L),b);
    // h_tpone = solve(trimatu(U), y + epsilon);
    
    // ========= inversion by hand ==================    
    omega_ts_sq_inv = (G_j.t() * Z_inv * G_j + taus_squared_inv[i]);
    X(0,0) = omega_ts_sq_inv(1,1);
    X(0,1) = -1.0*omega_ts_sq_inv(0,1);
    X(1,1) = omega_ts_sq_inv(0,0);
    X(1,0) = -1.0*omega_ts_sq_inv(1,0);
    omega_ts_sq = 1.0/(omega_ts_sq_inv(0,0)*omega_ts_sq_inv(1,1) -
		       omega_ts_sq_inv(0,1)*omega_ts_sq_inv(1,0)) * 
      X;
    // ==============================================
    // omega_ts_sq = inv_sympd(G_j.t() * Z_inv * G_j + taus_squared_inv[i]);

    q_t = omega_ts_sq * 
      (taus_squared_inv[i]*vs[i] + G_j.t()*Z_inv*(h_tpone-m_t));
    h_tpone = rmvnorm(rng_, 2, q_t, omega_ts_sq);

    sv_model_->get_ou_model_slow()->
      set_sigmas_element(i,
    			 exp(h_tpone(0)) / sqrt(sv_model_->get_delta_t()),
    			 h_tpone(0));

    sv_model_->get_ou_model_fast()->
      set_sigmas_element(i,
    			 exp(h_tpone(1)) / sqrt(sv_model_->get_delta_t()),
    			 h_tpone(1));
  }
}

void SVWithJumpsPosteriorSampler::draw_filtered_log_prices()
{
  double mu = sv_model_->get_constant_vol_model()->get_mu().
    get_discrete_time_parameter(sv_model_->get_delta_t());
  
  double alpha = sv_model_->get_ou_model_slow()->get_alpha().
    get_discrete_time_parameter(sv_model_->get_delta_t());

  double theta_fast = sv_model_->get_ou_model_fast()->get_theta().
    get_discrete_time_parameter(sv_model_->get_delta_t());

  double rho = sv_model_->get_ou_model_fast()->get_rho().
    get_discrete_time_parameter(sv_model_->get_delta_t());
  
  double tau_square_fast = sv_model_->get_ou_model_fast()->get_tau_square().
    get_discrete_time_parameter(sv_model_->get_delta_t());

  const std::vector<SigmaSingletonParameter>& sigmas_slow = 
    sv_model_->get_ou_model_slow()->
    get_sigmas().get_sigmas();

  const std::vector<SigmaSingletonParameter>& sigmas_fast = 
    sv_model_->get_ou_model_fast()->
    get_sigmas().get_sigmas();

  const std::vector<double>& jump_sizes =
    sv_model_->get_constant_vol_model()->
    get_jump_sizes();

  const std::vector<double>& deltas = 
    sv_model_->get_observational_model()->
    get_deltas();

  double xi_square = sv_model_->get_observational_model()->
    get_xi_square().get_continuous_time_parameter();

  double m_not = (sv_model_->get_observational_model()->
		     get_data_element(0).get_open());

  double C_not = xi_square;
  
  std::vector<double> vs (sv_model_->data_length()+1);
  std::vector<double> taus_sq (sv_model_->data_length()+1);
  
  double v_current = m_not;
  double tau_current_sq = C_not;
  
  double u_current = 0.0;
  double s_current_sq = 0.0;
  double log_P_t = 0;

  double eta_t_2 = 0; 
  double sigma_t_slow = 0;
  double sigma_t_fast = 0;

  for (unsigned i=0; i<=sv_model_->data_length(); ++i) {
    if (i == 0) {
      vs[i] = v_current;
      taus_sq[i] = tau_current_sq;
    } else {
      // the time corresponds to i*delta_t

      sigma_t_slow = 
	(sigmas_slow[i-1].
	 get_discrete_time_parameter(sv_model_->
				     get_delta_t()));
      sigma_t_fast =
	(sigmas_fast[i-1].
	 get_discrete_time_parameter(sv_model_->
				     get_delta_t()));

      double sigma_tp1_fast = (sigmas_fast[i].
			       get_discrete_time_parameter(sv_model_->
							   get_delta_t()));
      
      eta_t_2 =
	((log(sigma_tp1_fast) - alpha) - theta_fast*(log(sigma_t_fast) - alpha))
	/ sqrt(tau_square_fast);

      // ===============================
      u_current = 
      	v_current + mu + jump_sizes[i-1] +
      	sqrt(sigma_t_slow*sigma_t_fast)*rho*eta_t_2;
      
      s_current_sq = 
      	tau_current_sq + 
      	sigma_t_fast*sigma_t_slow*(1-square(rho));
      // =========================
      
      tau_current_sq = 1.0 / 
      	(deltas[i-1]/xi_square + 1.0/s_current_sq);
      
      log_P_t = (sv_model_->
		 get_observational_model()->
		 get_data_element(i-1).get_close());

      v_current = tau_current_sq *
        ((log_P_t)*deltas[i-1]/xi_square + 
         u_current/s_current_sq);
      
      vs[i] = v_current;
      taus_sq[i] = tau_current_sq;
    }
  }
  std::cout << "done with forward filter" << std::endl;
  std::vector<double> samples (sv_model_->data_length()+1);

  std::cout << samples[sv_model_->data_length()] << "\n";
  // BACKWARD SAMPLER
  double sigma2_tp1 = 0;
  double sigma_tp1_slow = 0;
  double sigma_tp1_fast = 0;

  for (std::vector<int>::size_type i = sv_model_->data_length();
       i != (std::vector<int>::size_type) - 1; --i) {
    if (i == sv_model_->data_length()) {
        samples[i] = 
	  vs[i] + sqrt(taus_sq[i])*gsl_ran_gaussian(rng_, 1.0);
    } else {

      sigma_tp1_slow = sigmas_slow[i].
	get_discrete_time_parameter(sv_model_->
				    get_delta_t());
      
      sigma_tp1_fast = sigmas_fast[i].
	get_discrete_time_parameter(sv_model_->
				    get_delta_t());

      double sigma_tp2_fast = sigmas_fast[i+1].
	get_discrete_time_parameter(sv_model_->
				    get_delta_t());
      
      sigma2_tp1 = sigma_tp1_slow*sigma_tp1_fast;
      
      eta_t_2 = 
	((log(sigma_tp2_fast) - alpha) - theta_fast*(log(sigma_tp1_fast) - alpha))
	/ sqrt(tau_square_fast);
      
      // ======================================== //
      double omega_current_sq = 1.0/
        (1.0/(sigma2_tp1*(1-square(rho))) + 1.0/taus_sq[i]);

      double q_current = omega_current_sq *
      	((samples[i+1]-mu-jump_sizes[i]-sigma2_tp1*rho*eta_t_2)/
      	 (sigma2_tp1*(1-square(rho))) + 
      	 vs[i]/taus_sq[i]);
      
      samples[i] = q_current + sqrt(omega_current_sq)*gsl_ran_gaussian(rng_,1.0);
      // ======================================== //
    }
  }

  sv_model_->get_constant_vol_model()->set_filtered_log_prices(samples);

  sv_model_->get_constant_vol_model()->set_y_star_ds();
}

void SVWithJumpsPosteriorSampler::draw_xi_square(double log_xi_square_prop_sd)
{
  double alpha = sv_model_->get_ou_model_fast()->get_alpha().
    get_discrete_time_parameter(sv_model_->get_delta_t());
  double rho = sv_model_->get_ou_model_fast()->get_rho().
    get_discrete_time_parameter(sv_model_->get_delta_t());
  double theta_fast = sv_model_->get_ou_model_fast()->get_theta().
    get_discrete_time_parameter(sv_model_->get_delta_t());
  double tau_square_fast = sv_model_->get_ou_model_fast()->get_tau_square().
    get_discrete_time_parameter(sv_model_->get_delta_t());
  const std::vector<SigmaSingletonParameter>& sigmas_slow = 
    sv_model_->get_ou_model_slow()->
    get_sigmas().get_sigmas();
  const std::vector<SigmaSingletonParameter>& sigmas_fast = 
    sv_model_->get_ou_model_fast()->
    get_sigmas().get_sigmas();
  const std::vector<double>& h_fast =
    sv_model_->get_ou_model_fast()->get_sigmas().get_discrete_time_log_sigmas();
  const std::vector<double>& jump_sizes =
    sv_model_->get_constant_vol_model()->get_jump_sizes();
  const std::vector<double>& deltas = 
    sv_model_->get_observational_model()->get_deltas();

  // CURRENT MODEL PARAMETERS
  double xi_square_current = sv_model_->
    get_observational_model()->get_xi_square().get_continuous_time_parameter();

  // TRANSFORMING TO TILDE SCALE
  double xi_square_tilde_current = log(xi_square_current);

  // SETTING THE MEAN
  double mean = xi_square_tilde_current;

  // PROPOSING ON THE TILDE SCALE
  double xi_square_tilde_proposal = mean + gsl_ran_gaussian(rng_, log_xi_square_prop_sd);

  // TRANSFORMING TO THE NOMINAL SCALE
  double xi_square_proposal = exp(xi_square_tilde_proposal);

  // CALCULATING LOG LIKELIHOOD FOR PROPOSAL
  double log_likelihood_proposal = 
    // ll 
    observational_model_sampler_.get_observational_model()->
    log_likelihood_integrated_filtered_prices(xi_square_proposal,
					      alpha,
					      theta_fast,
					      tau_square_fast,
					      rho,
					      sigmas_slow,
					      sigmas_fast,
					      h_fast,
					      jump_sizes,
					      deltas)
    // priors
    + observational_model_sampler_.
    get_observational_model()->get_xi_square_prior().log_likelihood(xi_square_proposal)
    // transformation determinant
    + xi_square_tilde_proposal;

  // CALCULATING LOG LIKELIHOOD FOR CURRENT
  double log_likelihood_current = 
    // ll 
    observational_model_sampler_.get_observational_model()->
    log_likelihood_integrated_filtered_prices(xi_square_current,
					      alpha,
					      theta_fast,
					      tau_square_fast,
					      rho,
					      sigmas_slow,
					      sigmas_fast,
					      h_fast,
					      jump_sizes,
					      deltas)
    // priors
    + observational_model_sampler_.
    get_observational_model()->get_xi_square_prior().log_likelihood(xi_square_current)
    // transformation determinant
    + xi_square_tilde_current;

  // ACCEPT / REJECT
  double log_a_acceptance = log_likelihood_proposal -
    log_likelihood_current;

  if (log(gsl_ran_flat(rng_,0,1)) <= log_a_acceptance) {
    // move accepted;
    // std::cout << "xi^2 move accpeted \n";
    observational_model_sampler_.get_observational_model()->
      set_xi_square(xi_square_proposal);

  } else {
    // move rejected;
    // std::cout << "xi^2 move rejected \n";
    observational_model_sampler_.get_observational_model()->
      set_xi_square(xi_square_current);
  }  
  
}
