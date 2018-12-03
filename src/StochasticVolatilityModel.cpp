#define ARMA_DONT_USE_WRAPPER
#include "src/armadillo-7.600.2/include/armadillo"
#include <cmath>
#define MATHLIB_STANDALONE
#include <Rmath.h>
#include <vector>
#include "MultivariateNormal.hpp"
#include "StochasticVolatilityModel.hpp"

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
}

// ==================== BASE MODEL ==========================
BaseModel::BaseModel()
  : delta_t_(0),
    const_delta_t_(false)
{}

BaseModel::BaseModel(double delta_t)
  : delta_t_(delta_t),
    const_delta_t_(true)
{}

BaseModel::~BaseModel()
{}

double BaseModel::get_delta_t() const
{
  return delta_t_;
}

bool BaseModel::const_delta_t() const
{
  return const_delta_t_;
}

// =============== OBSERVATIONAL MODEL =====================
ObservationalModel::ObservationalModel(const OpenCloseData& data,
				       double delta_t)
  : BaseModel(delta_t),
    ObservationalParams(),
    ObservationalPriors(delta_t),
    OpenCloseData(data),
    deltas_(std::vector<double> (data_length(), 1.0)),
    delta_prior_(DeltaPrior()),
    nu_(delta_prior_.get_nu())
{
  set_xi_square(get_xi_square_prior().get_xi_square_mean());
}

const std::vector<double>& ObservationalModel::get_filtered_log_prices() const
{
  return constant_vol_model_->get_filtered_log_prices();
}

const std::vector<double>& ObservationalModel::get_deltas() const
{
  return deltas_;
}

double ObservationalModel::get_nu() const
{
  return nu_;
}

void ObservationalModel::set_nu(double nu)
{
  nu_ = nu;
}

void ObservationalModel::set_delta_element(unsigned i, double delta)
{
  deltas_[i] = delta;
}

unsigned ObservationalModel::data_length() const
{
  return OpenCloseData::data_length();
}

double ObservationalModel::log_likelihood() const
{
  double log_likelihood = 0.0;
  double xi_squared = get_xi_square().get_continuous_time_parameter();

  for (unsigned i=0; i<data_length(); ++i) {
    double Y_i = get_data_element(i).get_close();
    double log_S_i = constant_vol_model_->get_filtered_log_prices()[i+1];

    log_likelihood = log_likelihood +
      dnorm(Y_i, log_S_i, sqrt(xi_squared/deltas_[i]), 1);
  }

  return log_likelihood;
}

double ObservationalModel::
log_likelihood_integrated_filtered_prices(
	  double xi_square,
	  double alpha,
	  double theta_fast,
	  double tau_square_fast,
	  double rho,
	  const std::vector<SigmaSingletonParameter>& sigmas_slow,
	  const std::vector<SigmaSingletonParameter>& sigmas_fast,
	  const std::vector<double>& h_fast,
	  const std::vector<double>& jump_sizes,
	  const std::vector<double>& deltas)  const
{
  double mu = constant_vol_model_->get_mu().
    get_discrete_time_parameter(get_delta_t());

  double u_current = get_data_element(0).get_open();
  double s_current_sq = xi_square;
  double log_P_t = get_data_element(0).get_open();

  double v_current = 0;
  double tau_current_sq = 0;

  double eta_t_2 = 0;
  double sigma_t_slow = 0;
  double sigma_t_fast = 0;

  double log_likelihood = 0;

  for (unsigned i=0; i<=data_length(); ++i) {
    if (i == 0) {
      log_likelihood = log_likelihood +
	dnorm(log_P_t, u_current, sqrt(xi_square/deltas[i] + s_current_sq), 1);

      v_current = u_current/(s_current_sq*deltas[i]/xi_square + 1.0) +
	log_P_t/(xi_square/(deltas[i]*s_current_sq) + 1.0);

      tau_current_sq = 1.0/(deltas[i]/xi_square + 1.0/s_current_sq);
    } else {
      // the time corresponds to i*delta_t
      sigma_t_slow =
	(sigmas_slow[i-1].
	 get_discrete_time_parameter(get_delta_t()));

      sigma_t_fast =
	(sigmas_fast[i-1].
	 get_discrete_time_parameter(get_delta_t()));

      double sigma_tp1_fast = (sigmas_fast[i].
			       get_discrete_time_parameter(get_delta_t()));

      eta_t_2 =
	((log(sigma_tp1_fast) - alpha) - theta_fast*(log(sigma_t_fast) - alpha))
	/ sqrt(tau_square_fast);

      // =========================================================
      u_current =
      	v_current + mu +
      	sqrt(sigma_t_slow*sigma_t_fast)*rho*eta_t_2;

      s_current_sq =
      	tau_current_sq +
      	sigma_t_fast*sigma_t_slow*(1-square(rho));
      // =========================================================

      // =========================================================
      log_P_t = (get_data_element(i-1).get_close());

      log_likelihood = log_likelihood +
	dnorm(log_P_t, u_current, sqrt(xi_square/deltas[i-1] + s_current_sq), 1);
      // =========================================================

      tau_current_sq = 1.0 /
      	(deltas[i-1]/xi_square + 1.0/s_current_sq);

      v_current = tau_current_sq *
        ((log_P_t)*deltas[i-1]/xi_square +
         u_current/s_current_sq);

    }
  }
  return log_likelihood;
}

// ================ CONST VOL MODEL =========================
ConstantVolatilityModel::
ConstantVolatilityModel(const ObservationalModel * observational_model,
			double delta_t)
  : BaseModel(delta_t),
    ConstantVolatilityParams(observational_model->data_length(),
			     delta_t),
    ConstantVolatilityPriors(delta_t),
    observational_model_(observational_model),
    y_star_(std::vector<double>(observational_model_->data_length())),
    ds_(std::vector<int>(observational_model_->data_length())),
    filtered_log_prices_(std::vector<double>(observational_model_->data_length()+1)),
    as_correction_(std::vector<double> {1.01418, 1.02248, 1.03403, 1.05207,
	  1.08153, 1.13114, 1.21754, 1.37454,
	  1.68327, 2.50097}),
    bs_correction_(std::vector<double> {0.50710, 0.51124, 0.51701, 0.52604,
	  0.54076, 0.56557, 0.60877, 0.68728,
	  0.84163, 1.25049})
{
  set_mu_hat(get_mu_prior().get_mu_hat_mean());
  for (unsigned i=0; i<data_length(); ++i) {
    if (i==0) {
      filtered_log_prices_[i] =
	(observational_model_->get_data_element(i).get_open());

      filtered_log_prices_[i+1] =
	(observational_model_->get_data_element(i).get_close());

    } else {
      filtered_log_prices_[i+1] =
	(observational_model_->get_data_element(i).get_close());
    }
  }
  set_y_star_ds();
}

const std::vector<double>& ConstantVolatilityModel::get_y_star() const
{
  return y_star_;
}

const std::vector<int>& ConstantVolatilityModel::get_ds() const
{
  return ds_;
}

const std::vector<double>& ConstantVolatilityModel::get_as() const
{
  return as_correction_;
}

const std::vector<double>& ConstantVolatilityModel::get_bs() const
{
  return bs_correction_;
}

unsigned ConstantVolatilityModel::data_length() const
{
  return y_star_.size();
}

void ConstantVolatilityModel::set_y_star_ds()
{
  std::cout << "In ConstantVolatilityModel::set_y_star_ds()" << "\n";
  double mu = get_mu().get_discrete_time_parameter(get_delta_t());
  for (unsigned i=0; i<data_length(); ++i) {
    double diff =
      filtered_log_prices_[i+1] -
      filtered_log_prices_[i] -
      mu;
    y_star_[i] =
      log( std::abs(diff) );
    if (y_star_[i] == -1.0*HUGE_VAL) {
      y_star_[i] = -20.0;
    }
    if (diff >= 0.0) {
      ds_[i] = 1;
    } else {
      ds_[i] = -1;
    }
  }
}

void ConstantVolatilityModel::
set_filtered_log_prices(const std::vector<double>& flps)
{
  if (flps.size()-1 == data_length()) {
    filtered_log_prices_ = flps;
    set_y_star_ds();
  } else{
    std::cout << "WARNING: NOT CORRECT LENGTH" << std::endl;
  }
}

double ConstantVolatilityModel::log_likelihood() const
{
  double log_likelihood = 0.0;
  const std::vector<int>& gammas = get_gammas().get_gammas();
  const std::vector<double>& ms = get_gammas().get_mixture_means();
  const std::vector<double>& vs_squared = get_gammas().get_mixture_variances();

  for (unsigned i=0; i<data_length(); ++i) {
    double y_i_star = get_y_star()[i];
    double h = ou_model_->get_sigmas().get_discrete_time_log_sigmas()[i];
    double m = ms[gammas[i]];
    double v_square = vs_squared[gammas[i]];
    log_likelihood = log_likelihood +
      dnorm(y_i_star, (h + m/2.0), sqrt(v_square)/2.0, 1);
  }

  return log_likelihood;
}

// =================== CONSTANT MULTIFACTOR VOL MODEL =================
ConstantMultifactorVolatilityModel::
ConstantMultifactorVolatilityModel(const ObservationalModel* observational_model,
				   double delta_t)
  : ConstantVolatilityModel(observational_model,
			    delta_t)
{}

double ConstantMultifactorVolatilityModel::log_likelihood() const
{
  double log_likelihood = 0.0;
  const std::vector<int>& gammas = get_gammas().get_gammas();
  const std::vector<double>& ms = get_gammas().get_mixture_means();
  const std::vector<double>& vs_squared = get_gammas().get_mixture_variances();

  for (unsigned i=0; i<data_length(); ++i) {
    double y_i_star = get_y_star()[i];

    double h_slow = get_ou_model_slow()->
      get_sigmas().get_discrete_time_log_sigmas()[i];
    double h_fast = get_ou_model_fast()->
      get_sigmas().get_discrete_time_log_sigmas()[i];

    double m = ms[gammas[i]];
    double v_square = vs_squared[gammas[i]];

    log_likelihood = log_likelihood +
      dnorm(y_i_star, (0.5*h_slow + 0.5*h_fast + m/2.0), sqrt(v_square)/2.0, 1);
  }

  return log_likelihood;
}

// =================== CONSTANT MULTIFACTOR VOL MODEL WITH JUMPS  ===============
ConstantMultifactorVolatilityModelWithJumps::
ConstantMultifactorVolatilityModelWithJumps(const ObservationalModel* observational_model,
					    double delta_t)
  : ConstantMultifactorVolatilityModel(observational_model,
				       delta_t),
    number_jumps_(0),
    jump_indicators_(std::vector<bool> (data_length(), false)),
    jump_sizes_(std::vector<double> (data_length(), 0.0)),
    jump_size_prior_(MuPrior(0.0,1e-8,delta_t)),
    jump_size_variance_prior_(SigmaSquarePrior()),
    jump_rate_prior_(LambdaPrior()),
    jump_size_mean_(MuParameter(jump_size_prior_.get_mu_hat_mean(),
				delta_t)),
   jump_size_variance_(SigmaSquareParam(jump_size_variance_prior_.
					 get_sigma_square_mean())),
		 jump_rate_(LambdaParam(jump_rate_prior_.get_lambda_mean()))
{
  set_y_star_ds();
}

void ConstantMultifactorVolatilityModelWithJumps::set_y_star_ds()
{
  const std::vector<double>& filtered_log_prices = get_filtered_log_prices();
  std::vector<double> y_star_new = std::vector<double> (data_length());
  std::vector<int> ds_new = std::vector<int> (data_length());
  double mu = get_mu().get_discrete_time_parameter(get_delta_t());

  for (unsigned i=1; i<data_length()+1; ++i) {
    double diff =
      filtered_log_prices[i] -
      filtered_log_prices[i-1] -
      mu -
      jump_sizes_[i-1];
    y_star_new[i-1] =
      log( std::abs(diff) );
    if ( std::abs(diff) < 1e-16 ) {
      const std::vector<double>& deltas = get_observational_model()->get_deltas();

      std::cout << "-HUGE_VAL encountered:" << filtered_log_prices[i+1] << " "
		<< filtered_log_prices[i] << " " << mu << " " << jump_sizes_[i]
		<< " " << deltas[i]
		<< " " <<  (filtered_log_prices[i+1]
			    - filtered_log_prices[i]
			    - mu
			    - jump_sizes_[i])
		<< "\n";
      // y_star_new[i] = -30.0;
    }
    if (std::signbit(diff)) {
      ds_new[i-1] = -1;
    } else {
      ds_new[i-1] = 1;
    }
  }

  set_y_star(y_star_new);
  set_ds(ds_new);
}

// ==================== OU MODEL ============================
OUModel::OUModel(const ConstantVolatilityModel* const_vol_model)
  : BaseModel(),
    const_vol_model_(const_vol_model),
    theta_tau_square_parameter_(ThetaTauSquareParameter()),
    alpha_parameter_(AlphaParameter()),
    rho_parameter_(RhoParameter()),
    sigmas_parameter_(SigmaParameter()),
    theta_prior_(ThetaPrior()),
    tau_square_prior_(TauSquarePrior(theta_prior_)),
    alpha_prior_(AlphaPrior()),
    rho_prior_(RhoPrior())
{
  alpha_parameter_.set_continuous_time_parameter(alpha_prior_.get_alpha_hat_mean());
}

OUModel::OUModel(const ConstantVolatilityModel* const_vol_model,
		 double delta_t)
  : BaseModel(delta_t),
    const_vol_model_(const_vol_model),
    theta_tau_square_parameter_(ThetaTauSquareParameter(delta_t)),
    alpha_parameter_(AlphaParameter(delta_t)),
    rho_parameter_(RhoParameter()),
    sigmas_parameter_(SigmaParameter(delta_t)),
    theta_prior_(ThetaPrior(delta_t)),
    tau_square_prior_(TauSquarePrior(theta_prior_, delta_t)),
    alpha_prior_(AlphaPrior(delta_t)),
    rho_prior_(RhoPrior())
{
  alpha_parameter_.set_continuous_time_parameter(alpha_prior_.get_alpha_hat_mean());
}

const ConstantVolatilityModel * OUModel::get_const_vol_model() const
{
  return const_vol_model_;
}

const ThetaParameter& OUModel::get_theta() const
{
  return theta_tau_square_parameter_.get_theta_parameter();
}

const TauSquareParameter& OUModel::get_tau_square() const
{
  return theta_tau_square_parameter_.get_tau_square_parameter();
}

const AlphaParameter& OUModel::get_alpha() const
{
  return alpha_parameter_;
}

const RhoParameter& OUModel::get_rho() const
{
  return rho_parameter_;
}

const SigmaParameter& OUModel::get_sigmas() const
{
  return sigmas_parameter_;
}

unsigned OUModel::data_length() const
{
  return sigmas_parameter_.get_sigmas().size()-1;
}

ThetaPrior& OUModel::get_theta_prior()
{
  return theta_prior_;
}

const TauSquarePrior& OUModel::get_tau_square_prior() const
{
  return tau_square_prior_;
}

const AlphaPrior& OUModel::get_alpha_prior() const
{
  return alpha_prior_;
}

const RhoPrior& OUModel::get_rho_prior() const
{
  return rho_prior_;
}

void OUModel::set_const_vol_model(const ConstantVolatilityModel* const_vol_model)
{
  const_vol_model_ = const_vol_model;
}

void OUModel::set_sigmas(const SigmaParameter& sigmas)
{
  sigmas_parameter_ = sigmas;
}

void OUModel::set_sigmas_element(unsigned i,
				 double sigma_hat,
				 double log_sigma)
{
  sigmas_parameter_.
    set_discrete_time_log_sigma_element(i, log_sigma);

  sigmas_parameter_.
    set_sigma_hat_element(i, sigma_hat);
}

void OUModel::set_tau_square_hat(double tau_square_hat)
{
  theta_tau_square_parameter_.set_tau_square_hat(tau_square_hat);
}

void OUModel::set_theta_hat(double theta_hat)
{
  theta_tau_square_parameter_.set_theta_hat(theta_hat);
}

void OUModel::set_alpha_hat(double alpha_hat)
{
  alpha_parameter_.set_continuous_time_parameter(alpha_hat);
}

void OUModel::set_rho(double rho)
{
  rho_parameter_.set_continuous_time_parameter(rho);
}

double OUModel::theta_j(unsigned i_data_index,
			unsigned j_mixture_index) const
{
  double theta =
    theta_tau_square_parameter_.get_theta_parameter().
    get_discrete_time_parameter(get_delta_t());
  double rho =
    rho_parameter_.get_discrete_time_parameter(get_delta_t());
  double tau_squared =
    theta_tau_square_parameter_.get_tau_square_parameter().
    get_discrete_time_parameter(get_delta_t());

  const std::vector<double>& mixture_means =
    const_vol_model_->get_gammas().get_mixture_means();
  const std::vector<int>& ds =
    const_vol_model_->get_ds();
  const std::vector<double>& bs_correction =
    const_vol_model_->get_bs();

  return theta -
    ds[i_data_index]*rho*sqrt(tau_squared) *
    bs_correction[j_mixture_index] *
    2.0 *
    exp(mixture_means[j_mixture_index]/2.0);
}

double OUModel::alpha_j(unsigned i_data_index,
			unsigned j_mixture_index) const
{
  double theta =
    theta_tau_square_parameter_.get_theta_parameter().
    get_discrete_time_parameter(get_delta_t());
  double rho =
    rho_parameter_.get_continuous_time_parameter();
  double tau_squared =
    theta_tau_square_parameter_.get_tau_square_parameter().
    get_discrete_time_parameter(get_delta_t());
  double alpha =
    alpha_parameter_.get_discrete_time_parameter(get_delta_t());

  const std::vector<double>& mixture_means =
    const_vol_model_->get_gammas().get_mixture_means();
  const std::vector<int>& ds =
    const_vol_model_->get_ds();
  const std::vector<double>& as_correction =
    const_vol_model_->get_as();
  const std::vector<double>& bs_correction =
    const_vol_model_->get_bs();
  const std::vector<double>& y_star =
    const_vol_model_->get_y_star();

  return alpha*(1-theta) +
    rho*ds[i_data_index] * sqrt(tau_squared) *
    exp(mixture_means[j_mixture_index]/2) *
    (as_correction[j_mixture_index] +
     bs_correction[j_mixture_index] * 2.0 *
     (y_star[i_data_index] - mixture_means[j_mixture_index]/2.0));
}

double OUModel::log_likelihood() const
{
  double log_likelihood = 0.0;
  double tau_squared =
    theta_tau_square_parameter_.get_tau_square_parameter().
    get_discrete_time_parameter(get_delta_t());
  double rho = rho_parameter_.get_continuous_time_parameter();
  double alpha = alpha_parameter_.get_discrete_time_parameter(get_delta_t());
  double theta = theta_tau_square_parameter_.get_theta_parameter().
    get_discrete_time_parameter(get_delta_t());

  const std::vector<int>& gammas =
    const_vol_model_->get_gammas().get_gammas();
  const std::vector<double>& hs =
    sigmas_parameter_.get_discrete_time_log_sigmas();

  double h = 0;
  double h_plus_one = 0;
  double theta_i = 0;
  double alpha_i = 0;
  for (unsigned i=0; i<data_length(); ++i) {
    h = hs[i];
    h_plus_one = hs[i+1];

    theta_i = theta_j(i,gammas[i]);
    alpha_i = alpha_j(i,gammas[i]);

    log_likelihood = log_likelihood +
      dnorm(h_plus_one, theta_i*h + alpha_i, sqrt(tau_squared*(1.0-square(rho))), 1);
  }

  log_likelihood = log_likelihood +
    dnorm(hs[0],
	  alpha,
	  sqrt(tau_squared/(1-square(theta))),
	  1);
  return log_likelihood;
}

std::vector<double> OUModel::alpha_posterior_mean_var() const
{
  const std::vector<double>& h =
    get_sigmas().get_discrete_time_log_sigmas();

  double alpha =
    get_alpha().get_discrete_time_parameter(get_delta_t());
   double theta =
    get_theta().get_discrete_time_parameter(get_delta_t());
  double rho =
    get_rho().get_continuous_time_parameter();
  double tau_square =
    get_tau_square().get_discrete_time_parameter(get_delta_t());
  const std::vector<int>& gammas =
    get_const_vol_model()->
    get_gammas().get_gammas();

  // const std::vector<double>& y_star =
  //   get_const_vol_model()->get_y_star();

  // const std::vector<double>& filtered_log_prices =
  //   get_const_vol_model()->get_filtered_log_prices();

  double likelihood_mean_over_variance = 0;
  double likelihood_var_inv = 0;
  double theta_t = 0;
  double A_t = 0;
  for (unsigned i=0; i<data_length(); ++i) {
    theta_t = theta_j(i,gammas[i]);
    A_t = alpha_j(i,gammas[i]) - alpha*(1-theta);

    // std::cout << "theta_" << i << "=" << theta_t
    // 	      << "; A_" << i << "=" << A_t
    // 	      << "; alpha_" << i << "=" << alpha_j(i,gammas[i])
    // 	      << "; mu=" << get_const_vol_model()->
    //   get_mu().get_discrete_time_parameter(get_delta_t())
    // 	      << "; y_star_" << i << "=" << y_star[i]
    // 	      << "; filtered_log_prices[" << i << "]=" << filtered_log_prices[i];
    // if (i>0) {
    //   std::cout << "; filtered_log_prices[" << i+1 << "]=" << filtered_log_prices[i+1];
    //   double y_star_ish = log(std::abs(filtered_log_prices[i+1] -
    // 				       filtered_log_prices[i] -
    // 				       get_const_vol_model()->get_mu().get_discrete_time_parameter(get_delta_t())));
    //   std::cout << "; y_star-ish=" << y_star_ish;
    //   if (y_star_ish == -1*HUGE_VAL) {
    // 	std::cout << "; -HUGE_VAL";
    //   }
    // }
    // std::cout << "\n";

    likelihood_mean_over_variance = likelihood_mean_over_variance +
      (h[i+1] - A_t - h[i]*theta_t)*
      (1-theta) /
      (tau_square * (1-square(rho)));

    likelihood_var_inv = likelihood_var_inv +
      square(1-theta) / (tau_square * (1-square(rho)));
  }
  double likelihood_var = 1.0/likelihood_var_inv;
  double likelihood_mean = likelihood_var * likelihood_mean_over_variance;

  // std::cout << "likelihood_var SLOW=" << likelihood_var << "\n";
  // std::cout << "likelihood_mean SLOW=" << likelihood_mean << "\n";

  double h_0_var = tau_square / (1-square(theta));
  double h_0_mean = h[0];

  double posterior_var =
    1.0 / (1.0/likelihood_var + 1.0/h_0_var);
  double posterior_mean =
    posterior_var * (likelihood_mean/likelihood_var +
		     h_0_mean/h_0_var);

  std::vector<double> out = std::vector<double> {posterior_mean, posterior_var};
  return out;
}

std::vector<double> OUModel::rho_posterior_mean_var() const
{
  const std::vector<double>& h =
    get_sigmas().get_discrete_time_log_sigmas();
  const std::vector<int>& ds =
    get_const_vol_model()->get_ds();
  double alpha =
    get_alpha().get_discrete_time_parameter(get_delta_t());
  double theta =
    get_theta().get_discrete_time_parameter(get_delta_t());
  double tau_square =
    get_tau_square().get_discrete_time_parameter(get_delta_t());
  const std::vector<int>& gammas =
    get_const_vol_model()->get_gammas().get_gammas();
  const std::vector<double>& y_star =
    get_const_vol_model()->get_y_star();

  const std::vector<double>& bs = get_const_vol_model()->get_bs();
  const std::vector<double>& as = get_const_vol_model()->get_as();

  const std::vector<double>& ms = get_const_vol_model()->get_gammas().get_mixture_means();

  double proposal_mean_over_variance = 0.0;
  double proposal_variance_inverse = 0.0;
  double R_j = 0;
  double R_j_h = 0;
  for (unsigned i=0; i<data_length(); ++i) {

    R_j =  sqrt(tau_square) * ds[i] * exp(ms[gammas[i]]/2.0) *
      (as[gammas[i]] + bs[gammas[i]] * 2.0 * (y_star[i] - ms[gammas[i]]/2.0));

    R_j_h = sqrt(tau_square) * ds[i] * exp(ms[gammas[i]]/2.0) *
      bs[gammas[i]] * 2.0;

    proposal_variance_inverse = proposal_variance_inverse +
      1.0 / (tau_square / square(-R_j + h[i]*R_j_h));

    proposal_mean_over_variance = proposal_mean_over_variance +
      ((-h[i+1] + alpha * (1-theta) + h[i]*theta) * (-R_j + h[i]*R_j_h)) /
      tau_square;
  }
  double proposal_variance = 1.0/proposal_variance_inverse;
  double proposal_mean = proposal_mean_over_variance * proposal_variance;

  std::vector<double> out {proposal_mean, 1.0*proposal_variance};
  return out;
}

std::vector<double> OUModel::tau_square_posterior_shape_rate() const
{
  double tau_square_alpha =
    get_tau_square_prior().get_tau_square_shape();
  double tau_square_beta =
    get_tau_square_prior().get_tau_square_scale();

  double rho =
    get_rho().get_continuous_time_parameter();
  double alpha =
    get_alpha().get_discrete_time_parameter(get_delta_t());
  double theta =
    get_theta().get_discrete_time_parameter(get_delta_t());
  const std::vector<double>& h =
    get_sigmas().get_discrete_time_log_sigmas();

  double proposal_alpha = tau_square_alpha + (data_length())/2.0;
  double proposal_beta  = tau_square_beta;

  for (unsigned i=0; i<data_length(); ++i) {
    proposal_beta = proposal_beta +
      0.5/(1-square(rho))*square(h[i+1] - h[i]*theta - alpha*(1-theta));
  }
  proposal_beta = proposal_beta +
    (1-square(theta))/2.0 * square(h[0]-alpha);

  std::vector<double> out = {proposal_alpha, proposal_beta};
  return out;
}

// ==================== FAST OU MODEL =======================
FastOUModel::FastOUModel(const ConstantVolatilityModel* const_vol_model,
			 const OUModel* ou_model_slow)
  : OUModel(const_vol_model),
    ou_model_slow_(ou_model_slow)
{}

FastOUModel::FastOUModel(const ConstantVolatilityModel* const_vol_model,
			 const OUModel* ou_model_slow,
			 double delta_t)
  : OUModel(const_vol_model,
	    delta_t),
    ou_model_slow_(ou_model_slow)
{}

FastOUModel::~FastOUModel()
{}

double FastOUModel::theta_j_one(unsigned i_data_index,
				unsigned j_mixture_index) const
{
  double rho =
    get_rho().get_continuous_time_parameter();
  double tau_squared =
    get_tau_square().get_discrete_time_parameter(get_delta_t());

  const std::vector<double>& mixture_means =
    get_const_vol_model()->get_gammas().get_mixture_means();
  const std::vector<int>& ds =
    get_const_vol_model()->get_ds();
  const std::vector<double>& bs_correction =
    get_const_vol_model()->get_bs();

  return -ds[i_data_index]*rho*sqrt(tau_squared)
    * bs_correction[j_mixture_index]
    * exp(mixture_means[j_mixture_index]/2.0);
}

double FastOUModel::theta_j_two(unsigned i_data_index,
				unsigned j_mixture_index) const
{
  double theta =
    get_theta().get_discrete_time_parameter(get_delta_t());
  double rho =
    get_rho().get_continuous_time_parameter();
  double tau_squared =
    get_tau_square().get_discrete_time_parameter(get_delta_t());

  const std::vector<double>& mixture_means =
    get_const_vol_model()->get_gammas().get_mixture_means();
  const std::vector<int>& ds =
    get_const_vol_model()->get_ds();
  const std::vector<double>& bs_correction =
    get_const_vol_model()->get_bs();

  return theta -
    ds[i_data_index]*rho*sqrt(tau_squared) *
    bs_correction[j_mixture_index] *
    exp(mixture_means[j_mixture_index]/2.0);
}

double FastOUModel::alpha_j(unsigned i_data_index,
			    unsigned j_mixture_index) const
{
  double theta =
    get_theta().get_discrete_time_parameter(get_delta_t());
  double rho =
    get_rho().get_continuous_time_parameter();
  double tau_squared =
    get_tau_square().get_discrete_time_parameter(get_delta_t());
  double alpha = 
    get_alpha().get_discrete_time_parameter(get_delta_t());
  const std::vector<double>& v_squared =
    get_const_vol_model()->get_gammas().get_mixture_variances();
  const std::vector<double>& y_star =
    get_const_vol_model()->get_y_star();

  const std::vector<double>& mixture_means =
    get_const_vol_model()->get_gammas().get_mixture_means();
  const std::vector<int>& ds =
    get_const_vol_model()->get_ds();
  const std::vector<double>& as_correction =
    get_const_vol_model()->get_as();
  const std::vector<double>& bs_correction =
    get_const_vol_model()->get_bs();

  double out = 
    // alpha*(1-theta) + 
    // sqrt(tau_squared)*(ds[i_data_index]*rho*exp(mixture_means[j_mixture_index]/2)*
    // 		       as_correction[j_mixture_index] + 
    // 		       ds[i_data_index]*rho*bs_correction[j_mixture_index]*
    // 		       sqrt(v_squared[j_mixture_index])*
    // 		       exp(mixture_means[j_mixture_index]/2)*
    // 		       (y_star[i_data_index]-mixture_means[j_mixture_index]/2)/
    // 		       (sqrt(v_squared[j_mixture_index])/2));
    alpha*(1-theta)
	+ rho*sqrt(tau_squared)*ds[i_data_index]*exp(mixture_means[j_mixture_index]/2.0)
	* (as_correction[j_mixture_index] + bs_correction[j_mixture_index]
	   * 2.0 * (y_star[i_data_index] - mixture_means[j_mixture_index]/2.0));
  return out;
}

double FastOUModel::log_likelihood() const
{
  double log_likelihood = 0.0;
  double tau_squared =
    get_tau_square().get_discrete_time_parameter(get_delta_t());
  double rho = get_rho().get_continuous_time_parameter();
  double alpha = get_alpha().get_discrete_time_parameter(get_delta_t());
  double theta = get_theta().get_discrete_time_parameter(get_delta_t());
  const std::vector<int>& gammas = get_const_vol_model()->get_gammas().get_gammas();

  double h = 0;
  double h_slow = 0;
  double h_plus_one = 0;
  double theta_i_one = 0;
  double theta_i_two = 0;
  double alpha_i = 0;

  for (unsigned i=0; i<data_length(); ++i) {
    h = get_sigmas().get_discrete_time_log_sigmas()[i];
    h_slow = ou_model_slow_->get_sigmas().get_discrete_time_log_sigmas()[i];
    h_plus_one = get_sigmas().get_discrete_time_log_sigmas()[i+1];

    theta_i_one = theta_j_one(i,gammas[i]);
    theta_i_two = theta_j_two(i,gammas[i]);

    alpha_i = alpha_j(i,gammas[i]);

    log_likelihood = log_likelihood +
      dnorm(h_plus_one,
	    theta_i_one*h_slow + theta_i_two*h + alpha_i,
	    sqrt(tau_squared*(1.0-square(rho))), 1);
  }

  log_likelihood = log_likelihood +
    dnorm(get_sigmas().get_discrete_time_log_sigmas()[0],
  	  alpha,
  	  sqrt(tau_squared/(1-square(theta))),
  	  1);
  return log_likelihood;
}

double FastOUModel::log_likelihood(double rho,
				   double theta,
				   double tau_squared)
{
  double log_likelihood = 0.0;

  double alpha = get_alpha().get_discrete_time_parameter(get_delta_t());

  const std::vector<int>& gammas = get_const_vol_model()->get_gammas().get_gammas();
  const std::vector<double>& mixture_means =
    get_const_vol_model()->get_gammas().get_mixture_means();
  const std::vector<int>& ds =
    get_const_vol_model()->get_ds();
  const std::vector<double>& as_correction =
    get_const_vol_model()->get_as();
  const std::vector<double>& bs_correction =
    get_const_vol_model()->get_bs();
  const std::vector<double>& y_star =
    get_const_vol_model()->get_y_star();

  double h = 0;
  double h_slow = 0;
  double h_plus_one = 0;
  double theta_i_one = 0;
  double theta_i_two = 0;
  double alpha_i = 0;

  for (unsigned i=0; i<data_length(); ++i) {
    h = get_sigmas().get_discrete_time_log_sigmas()[i];
    h_slow = ou_model_slow_->get_sigmas().get_discrete_time_log_sigmas()[i];
    h_plus_one = get_sigmas().get_discrete_time_log_sigmas()[i+1];

    theta_i_one = -1.0*
      ds[i]*rho*sqrt(tau_squared) *
      bs_correction[gammas[i]] *
      exp(mixture_means[gammas[i]]/2.0);
    // theta_i_one = theta_j_one(i, gammas[i]);

    theta_i_two = theta - ds[i]*rho*sqrt(tau_squared) *
      bs_correction[gammas[i]] *
      exp(mixture_means[gammas[i]]/2.0);
    // theta_i_two = theta_j_two(i,gammas[i]);

    alpha_i = alpha*(1-theta) +
      rho*ds[i] * sqrt(tau_squared) *
      exp(mixture_means[gammas[i]]/2) *
      (as_correction[gammas[i]] +
       bs_correction[gammas[i]] * 2.0 *
       (y_star[i] - mixture_means[gammas[i]]/2.0));
    //    alpha_i = alpha_j(i, gammas[i]);

    log_likelihood = log_likelihood +
      dnorm(h_plus_one,
	    theta_i_one*h_slow + theta_i_two*h + alpha_i,
	    sqrt(tau_squared*(1.0-square(rho))), 1);
  }

  log_likelihood = log_likelihood +
    dnorm(get_sigmas().get_discrete_time_log_sigmas()[0],
	  alpha,
	  sqrt(tau_squared/(1-square(theta))),
	  1);

  return log_likelihood;
}

double FastOUModel::log_likelihood_tilde(double rho_tilde,
					 double theta_tilde,
					 double tau_squared_tilde)
{
  double rho = 2*(exp(rho_tilde) / (exp(rho_tilde)+1)) - 1;
  double theta = (exp(theta_tilde) / (exp(theta_tilde)+1));
  double tau_square = exp(tau_squared_tilde);

  double ll = (log_likelihood(rho,
			      theta,
			      tau_square)
	       + log(2) + rho_tilde - 2*log(exp(rho_tilde)+1)
	       + theta_tilde - 2*log(exp(theta_tilde)+1)
	       + tau_squared_tilde);
  return ll;
}

double FastOUModel::log_likelihood_rho(double rho)
{
  double log_likelihood = 0.0;
  double tau_squared =
    get_tau_square().get_discrete_time_parameter(get_delta_t());
  double alpha = get_alpha().get_discrete_time_parameter(get_delta_t());
  double theta = get_theta().get_discrete_time_parameter(get_delta_t());
  const std::vector<int>& gammas = get_const_vol_model()->get_gammas().get_gammas();
  const std::vector<double>& mixture_means =
    get_const_vol_model()->get_gammas().get_mixture_means();
  const std::vector<int>& ds =
    get_const_vol_model()->get_ds();
  const std::vector<double>& as_correction =
    get_const_vol_model()->get_as();
  const std::vector<double>& bs_correction =
    get_const_vol_model()->get_bs();
  const std::vector<double>& y_star =
    get_const_vol_model()->get_y_star();

  double h = 0;
  double h_slow = 0;
  double h_plus_one = 0;
  double theta_i_one = 0;
  double theta_i_two = 0;
  double alpha_i = 0;

  for (unsigned i=0; i<data_length(); ++i) {
    h = get_sigmas().get_discrete_time_log_sigmas()[i];
    h_slow = ou_model_slow_->get_sigmas().get_discrete_time_log_sigmas()[i];
    h_plus_one = get_sigmas().get_discrete_time_log_sigmas()[i+1];

    theta_i_one = -1.0*
      ds[i]*rho*sqrt(tau_squared) *
      bs_correction[gammas[i]] *
      exp(mixture_means[gammas[i]]/2.0);
    // theta_i_one = theta_j_one(i, gammas[i]);

    theta_i_two = theta - ds[i]*rho*sqrt(tau_squared) *
      bs_correction[gammas[i]] *
      exp(mixture_means[gammas[i]]/2.0);
    // theta_i_two = theta_j_two(i,gammas[i]);

    alpha_i = alpha*(1-theta) +
      rho*ds[i] * sqrt(tau_squared) *
      exp(mixture_means[gammas[i]]/2) *
      (as_correction[gammas[i]] +
       bs_correction[gammas[i]] * 2.0 *
       (y_star[i] - mixture_means[gammas[i]]/2.0));
    //    alpha_i = alpha_j(i, gammas[i]);

    log_likelihood = log_likelihood +
      dnorm(h_plus_one,
	    theta_i_one*h_slow + theta_i_two*h + alpha_i,
	    sqrt(tau_squared*(1.0-square(rho))), 1);
  }

  log_likelihood = log_likelihood +
    dnorm(get_sigmas().get_discrete_time_log_sigmas()[0],
	  alpha,
	  sqrt(tau_squared/(1-square(theta))),
	  1);

  return log_likelihood;
}

double FastOUModel::log_likelihood_tau_square(double tau_squared)
{
  double log_likelihood = 0.0;
  double rho =
    get_rho().get_discrete_time_parameter(get_delta_t());
  double alpha = get_alpha().get_discrete_time_parameter(get_delta_t());
  double theta = get_theta().get_discrete_time_parameter(get_delta_t());
  const std::vector<int>& gammas = get_const_vol_model()->get_gammas().get_gammas();
  const std::vector<double>& mixture_means =
    get_const_vol_model()->get_gammas().get_mixture_means();
  const std::vector<int>& ds =
    get_const_vol_model()->get_ds();
  const std::vector<double>& as_correction =
    get_const_vol_model()->get_as();
  const std::vector<double>& bs_correction =
    get_const_vol_model()->get_bs();
  const std::vector<double>& y_star =
    get_const_vol_model()->get_y_star();

  double h = 0;
  double h_slow = 0;
  double h_plus_one = 0;
  double theta_i_one = 0;
  double theta_i_two = 0;
  double alpha_i = 0;

  for (unsigned i=0; i<data_length(); ++i) {
    h = get_sigmas().get_discrete_time_log_sigmas()[i];
    h_slow = ou_model_slow_->get_sigmas().get_discrete_time_log_sigmas()[i];
    h_plus_one = get_sigmas().get_discrete_time_log_sigmas()[i+1];

    theta_i_one = -1.0*
      ds[i]*rho*sqrt(tau_squared) *
      bs_correction[gammas[i]] *
      exp(mixture_means[gammas[i]]/2.0);
    // theta_i_one = theta_j_one(i, gammas[i]);

    theta_i_two = theta - ds[i]*rho*sqrt(tau_squared) *
      bs_correction[gammas[i]] *
      exp(mixture_means[gammas[i]]/2.0);
    // theta_i_two = theta_j_two(i,gammas[i]);

    alpha_i = alpha*(1-theta) +
      rho*ds[i] * sqrt(tau_squared) *
      exp(mixture_means[gammas[i]]/2) *
      (as_correction[gammas[i]] +
       bs_correction[gammas[i]] * 2.0 *
       (y_star[i] - mixture_means[gammas[i]]/2.0));
    //    alpha_i = alpha_j(i, gammas[i]);

    log_likelihood = log_likelihood +
      dnorm(h_plus_one,
	    theta_i_one*h_slow + theta_i_two*h + alpha_i,
	    sqrt(tau_squared*(1.0-square(rho))), 1);
  }

  log_likelihood = log_likelihood +
    dnorm(get_sigmas().get_discrete_time_log_sigmas()[0],
	  alpha,
	  sqrt(tau_squared/(1-square(theta))),
	  1);

  return log_likelihood;
}

// numeric deriv nominal scale
double FastOUModel::rho_deriv_numeric_nominal_scale(double rho,
						    double theta,
						    double tau_squared,
						    double drho)
{
  double ll_rho_plus = log_likelihood(rho + drho,
				      theta,
				      tau_squared);

  double ll_rho_minus = log_likelihood(rho - drho,
				       theta,
				       tau_squared);

  double out = (ll_rho_plus - ll_rho_minus)/(2.0*drho);
  return out;
}

double FastOUModel::rho_deriv_analytic_nominal_scale(double rho,
						     double theta,
						     double tau_squared)
{
  const std::vector<double>& h_fast =
    get_sigmas().get_discrete_time_log_sigmas();
  const std::vector<double>& h_slow =
    get_ou_model_slow()->get_sigmas().get_discrete_time_log_sigmas();

  const std::vector<int>& ds =
    get_const_vol_model()->get_ds();
  double alpha =
    get_alpha().get_discrete_time_parameter(get_delta_t());
  const std::vector<int>& gammas =
    get_const_vol_model()->get_gammas().get_gammas();
  const std::vector<double>& y_star =
    get_const_vol_model()->get_y_star();

  const std::vector<double>& bs = get_const_vol_model()->get_bs();
  const std::vector<double>& as = get_const_vol_model()->get_as();

  const std::vector<double>& ms =
    get_const_vol_model()->get_gammas().get_mixture_means();

  double sum_1 = 0;
  double sum_2 = 0;

  double R_j = 0;
  double R_j_h_fast = 0;
  double R_j_h_slow = 0;
  double A_j = 0;
  for (unsigned i=0; i<data_length(); ++i) {
    R_j =  sqrt(tau_squared) * ds[i] * exp(ms[gammas[i]]/2.0) *
      (as[gammas[i]] + bs[gammas[i]] * 2.0 * (y_star[i] - ms[gammas[i]]/2.0));

    R_j_h_fast = sqrt(tau_squared) * ds[i] * exp(ms[gammas[i]]/2.0) *
      bs[gammas[i]];

    R_j_h_slow = R_j_h_fast;

    A_j = h_fast[i]*R_j_h_fast + h_slow[i]*R_j_h_slow - R_j;

    sum_1 = sum_1 +
      -square(A_j)/(2.0*tau_squared*square(1-square(rho)))*2.0*rho*
      square(rho - (alpha*(1-theta) + h_fast[i]*theta - h_fast[i+1])/A_j);
    sum_2 = sum_2 +
      -square(A_j)/(2.0*tau_squared*(1-square(rho)))*2.0*
      (rho - (alpha*(1-theta) + h_fast[i]*theta - h_fast[i+1])/A_j);
  }
  long double out = data_length()*rho/(1.0-square(rho))
    + sum_1
    + sum_2;

  return out;
}

double FastOUModel::rho_double_deriv_numeric_nominal_scale(double rho,
							   double theta,
							   double tau_squared,
							   double drho)
{
  double ll_rho_plus = log_likelihood(rho + drho,
				      theta,
				      tau_squared);

  double ll_rho_minus = log_likelihood(rho - drho,
				       theta,
				       tau_squared);

  double ll_rho = log_likelihood(rho,
				 theta,
				 tau_squared);

  double out = (ll_rho_plus - 2.0*ll_rho + ll_rho_minus)/(square(drho));
  return out;
}

double FastOUModel::rho_theta_deriv_numeric_nominal_scale(double rho,
							  double theta,
							  double tau_squared,
							  double drho,
							  double dtheta)
{
  double ll_rho_theta_plus_plus = log_likelihood(rho + drho,
						 theta + dtheta,
						 tau_squared);
  double ll_rho_theta_plus_minus = log_likelihood(rho + drho,
						  theta - dtheta,
						  tau_squared);
  double ll_rho_theta_minus_plus = log_likelihood(rho - drho,
						  theta + dtheta,
						  tau_squared);
  double ll_rho_theta_minus_minus = log_likelihood(rho - drho,
						   theta - dtheta,
						   tau_squared);

  double out =
    (ll_rho_theta_plus_plus -
     ll_rho_theta_plus_minus -
     ll_rho_theta_minus_plus +
     ll_rho_theta_minus_minus) /
    (4*drho*dtheta);
  return out;
}

double FastOUModel::rho_tau_square_deriv_numeric_nominal_scale(double rho,
							       double theta,
							       double tau_squared,
							       double drho,
							       double dtau_sq)
{
  double ll_rho_tau_plus_plus = log_likelihood(rho + drho,
					       theta,
					       tau_squared + dtau_sq);
  double ll_rho_tau_plus_minus = log_likelihood(rho + drho,
						theta,
						tau_squared - dtau_sq);
  double ll_rho_tau_minus_plus = log_likelihood(rho - drho,
						theta,
						tau_squared + dtau_sq);
  double ll_rho_tau_minus_minus = log_likelihood(rho - drho,
						   theta,
						   tau_squared - dtau_sq);

  double out =
    (ll_rho_tau_plus_plus -
     ll_rho_tau_plus_minus -
     ll_rho_tau_minus_plus +
     ll_rho_tau_minus_minus) /
    (4*drho*dtau_sq);
  return out;
}

double FastOUModel::theta_deriv_numeric_nominal_scale(double rho,
						      double theta,
						      double tau_squared,
						      double dtheta)
{
  double ll_theta_plus = log_likelihood(rho,
					theta + dtheta,
					tau_squared);

  double ll_theta_minus = log_likelihood(rho,
					 theta - dtheta,
					 tau_squared);

  double out = (ll_theta_plus - ll_theta_minus)/(2.0*dtheta);
  return out;
}

double FastOUModel::theta_double_deriv_numeric_nominal_scale(double rho,
							     double theta,
							     double tau_squared,
							     double dtheta)
{
  double ll_theta_plus = log_likelihood(rho,
					theta + dtheta,
					tau_squared);

  double ll_theta_minus = log_likelihood(rho,
					 theta - dtheta,
					 tau_squared);

  double ll_theta = log_likelihood(rho,
				   theta,
				   tau_squared);

  double out = (ll_theta_plus - 2.0*ll_theta + ll_theta_minus)/(square(dtheta));
  return out;
}

double FastOUModel::theta_tau_square_deriv_numeric_nominal_scale(double rho,
							       double theta,
							       double tau_squared,
							       double dtheta,
							       double dtau_sq)
{
  double ll_theta_tau_plus_plus = log_likelihood(rho,
						 theta + dtheta,
						 tau_squared + dtau_sq);
  double ll_theta_tau_plus_minus = log_likelihood(rho,
						theta + dtheta,
						tau_squared - dtau_sq);
  double ll_theta_tau_minus_plus = log_likelihood(rho,
						  theta - dtheta,
						  tau_squared + dtau_sq);
  double ll_theta_tau_minus_minus = log_likelihood(rho,
						   theta - dtheta,
						   tau_squared - dtau_sq);

  double out =
    (ll_theta_tau_plus_plus -
     ll_theta_tau_plus_minus -
     ll_theta_tau_minus_plus +
     ll_theta_tau_minus_minus) /
    (4*dtheta*dtau_sq);
  return out;
}

double FastOUModel::tau_square_deriv_numeric_nominal_scale(double rho,
							   double theta,
							   double tau_squared,
							   double dtau_sq)
{
  double ll_tau_sq_plus = log_likelihood(rho,
					 theta,
					 tau_squared + dtau_sq);

  double ll_tau_sq_minus = log_likelihood(rho,
					  theta,
					  tau_squared - dtau_sq);

  double out = (ll_tau_sq_plus - ll_tau_sq_minus)/(2.0*dtau_sq);
  return out;
}

double FastOUModel::tau_square_double_deriv_numeric_nominal_scale(double rho,
								  double theta,
								  double tau_squared,
								  double dtau_sq)
{
  double ll_tau_sq_plus = log_likelihood(rho,
					 theta,
					 tau_squared + dtau_sq);

  double ll_tau_sq_minus = log_likelihood(rho,
					  theta,
					  tau_squared - dtau_sq);

  double ll_tau_sq = log_likelihood(rho,
				    theta,
				    tau_squared);

  double out = (ll_tau_sq_plus - 2.0*ll_tau_sq + ll_tau_sq_minus)/
    (square(dtau_sq));
  return out;
}

// numeric deriv tilde scale
double FastOUModel::rho_deriv_numeric_tilde_scale(double rho,
						  double theta,
						  double tau_squared,
						  double drho)
{
  double ll_plus = log_likelihood_tilde(rho + drho,
					theta,
					tau_squared);

  double ll_minus = log_likelihood_tilde(rho - drho,
					 theta,
					 tau_squared);

  double out = (ll_plus - ll_minus)/(2.0*drho);
  return out;
}

double FastOUModel::rho_deriv_analytic_tilde_scale(double rho_tilde,
						   double theta_tilde,
						   double tau_squared_tilde)
{
  double rho = 2*exp(rho_tilde)/(exp(rho_tilde)+1.0) - 1.0;
  double theta = exp(theta_tilde)/(exp(theta_tilde)+1.0);
  double tau_square = exp(tau_squared_tilde);

  double dldrho = rho_deriv_analytic_nominal_scale(rho,theta,tau_square);
  double drho_drho_tilde = 2.0*exp(rho_tilde)/square(exp(rho_tilde)+1.0);

  double out = dldrho*drho_drho_tilde + (1.0-exp(rho_tilde))/(exp(rho_tilde)+1.0);
  return out;
}

double FastOUModel::rho_deriv_analytic_tilde_scale(double rho_tilde)
{

  double theta = get_theta().get_discrete_time_parameter(get_delta_t());
  double tau_square = get_tau_square().get_discrete_time_parameter(get_delta_t());

  double rho = 2*exp(rho_tilde)/(exp(rho_tilde)+1.0) - 1.0;

  double dldrho = rho_deriv_analytic_nominal_scale(rho,theta,tau_square);
  double drho_drho_tilde = 2.0*exp(rho_tilde)/square(exp(rho_tilde)+1.0);

  double out = dldrho*drho_drho_tilde + (1.0-exp(rho_tilde))/(exp(rho_tilde)+1.0);
  return out;
}

double FastOUModel::rho_double_deriv_numeric_tilde_scale(double rho,
							 double theta,
							 double tau_squared,
							 double drho)
{
  double ll_plus = log_likelihood_tilde(rho + drho,
					theta,
					tau_squared);

  double ll_minus = log_likelihood_tilde(rho - drho,
					 theta,
					 tau_squared);

  double ll = log_likelihood_tilde(rho,
				   theta,
				   tau_squared);

  double out = (ll_plus - 2.0*ll + ll_minus)/(square(drho));
  return out;
}

double FastOUModel::rho_theta_deriv_numeric_tilde_scale(double rho,
							double theta,
							double tau_squared,
							double drho,
							double dtheta)
{
  double ll_rho_theta_plus_plus = log_likelihood_tilde(rho + drho,
						       theta + dtheta,
						       tau_squared);
  double ll_rho_theta_plus_minus = log_likelihood_tilde(rho + drho,
							theta - dtheta,
							tau_squared);
  double ll_rho_theta_minus_plus = log_likelihood_tilde(rho - drho,
							theta + dtheta,
							tau_squared);
  double ll_rho_theta_minus_minus = log_likelihood_tilde(rho - drho,
							 theta - dtheta,
							 tau_squared);

  double out =
    (ll_rho_theta_plus_plus -
     ll_rho_theta_plus_minus -
     ll_rho_theta_minus_plus +
     ll_rho_theta_minus_minus) /
    (4*drho*dtheta);
  return out;
}

double FastOUModel::rho_tau_square_deriv_numeric_tilde_scale(double rho,
							     double theta,
							     double tau_squared,
							     double drho,
							     double dtau_sq)
{
  double ll_rho_tau_plus_plus = log_likelihood_tilde(rho + drho,
						     theta,
						     tau_squared + dtau_sq);
  double ll_rho_tau_plus_minus = log_likelihood_tilde(rho + drho,
						      theta,
						      tau_squared - dtau_sq);
  double ll_rho_tau_minus_plus = log_likelihood_tilde(rho - drho,
						      theta,
						      tau_squared + dtau_sq);
  double ll_rho_tau_minus_minus = log_likelihood_tilde(rho - drho,
						       theta,
						       tau_squared - dtau_sq);

  double out =
    (ll_rho_tau_plus_plus -
     ll_rho_tau_plus_minus -
     ll_rho_tau_minus_plus +
     ll_rho_tau_minus_minus) /
    (4*drho*dtau_sq);
  return out;
}

double FastOUModel::theta_deriv_numeric_tilde_scale(double rho,
						    double theta,
						    double tau_squared,
						    double dtheta)
{
  double ll_theta_plus = log_likelihood_tilde(rho,
					theta + dtheta,
					tau_squared);

  double ll_theta_minus = log_likelihood_tilde(rho,
					       theta - dtheta,
					       tau_squared);

  double out = (ll_theta_plus - ll_theta_minus)/(2.0*dtheta);
  return out;
}

double FastOUModel::theta_double_deriv_numeric_tilde_scale(double rho,
							   double theta,
							   double tau_squared,
							   double dtheta)
{
  double ll_theta_plus = log_likelihood_tilde(rho,
					      theta + dtheta,
					      tau_squared);

  double ll_theta_minus = log_likelihood_tilde(rho,
					       theta - dtheta,
					       tau_squared);

  double ll_theta = log_likelihood_tilde(rho,
					 theta,
					 tau_squared);

  double out = (ll_theta_plus - 2.0*ll_theta + ll_theta_minus)/(square(dtheta));
  return out;
}

double FastOUModel::theta_tau_square_deriv_numeric_tilde_scale(double rho,
							       double theta,
							       double tau_squared,
							       double dtheta,
							       double dtau_sq)
{
  double ll_theta_tau_plus_plus = log_likelihood_tilde(rho,
						 theta + dtheta,
						 tau_squared + dtau_sq);
  double ll_theta_tau_plus_minus = log_likelihood_tilde(rho,
						theta + dtheta,
						tau_squared - dtau_sq);
  double ll_theta_tau_minus_plus = log_likelihood_tilde(rho,
						  theta - dtheta,
						  tau_squared + dtau_sq);
  double ll_theta_tau_minus_minus = log_likelihood_tilde(rho,
						   theta - dtheta,
						   tau_squared - dtau_sq);

  double out =
    (ll_theta_tau_plus_plus -
     ll_theta_tau_plus_minus -
     ll_theta_tau_minus_plus +
     ll_theta_tau_minus_minus) /
    (4*dtheta*dtau_sq);
  return out;
}

double FastOUModel::tau_square_deriv_numeric_tilde_scale(double rho,
							 double theta,
							 double tau_squared,
							 double dtau_sq)
{
  double ll_tau_sq_plus = log_likelihood_tilde(rho,
					 theta,
					 tau_squared + dtau_sq);

  double ll_tau_sq_minus = log_likelihood_tilde(rho,
					  theta,
					  tau_squared - dtau_sq);

  double out = (ll_tau_sq_plus - ll_tau_sq_minus)/(2.0*dtau_sq);
  return out;
}

double FastOUModel::tau_square_double_deriv_numeric_tilde_scale(double rho,
								double theta,
								double tau_squared,
								double dtau_sq)
{
  double ll_tau_sq_plus = log_likelihood_tilde(rho,
					       theta,
					       tau_squared + dtau_sq);

  double ll_tau_sq_minus = log_likelihood_tilde(rho,
						theta,
						tau_squared - dtau_sq);

  double ll_tau_sq = log_likelihood_tilde(rho,
					  theta,
					  tau_squared);

  double out = (ll_tau_sq_plus - 2.0*ll_tau_sq + ll_tau_sq_minus)/
    (square(dtau_sq));
  return out;
}

// end of numeric derivs

std::vector<double> FastOUModel::alpha_posterior_mean_var() const
{
  const std::vector<double>& h_fast =
    get_sigmas().get_discrete_time_log_sigmas();
  const std::vector<double>& h_slow =
    get_ou_model_slow()->
    get_sigmas().get_discrete_time_log_sigmas();

  double alpha =
    get_alpha().get_discrete_time_parameter(get_delta_t());
  double theta =
    get_theta().get_discrete_time_parameter(get_delta_t());
  double rho =
    get_rho().get_continuous_time_parameter();
  double tau_square =
    get_tau_square().get_discrete_time_parameter(get_delta_t());
  const std::vector<int>& gammas =
    get_const_vol_model()->
    get_gammas().get_gammas();

  double likelihood_var_inverse = 0.0;
  double likelihood_mean_over_variance = 0.0;
  double theta_t_fast = 0;
  double theta_t_slow = 0;
  double A_t = 0;

  for (unsigned i=0; i<data_length(); ++i) {
    theta_t_fast = theta_j_two(i,gammas[i]);
    theta_t_slow = theta_j_one(i,gammas[i]);
    A_t = alpha_j(i,gammas[i]) - alpha*(1-theta);

    likelihood_mean_over_variance = likelihood_mean_over_variance +
      (h_fast[i+1] - A_t - h_slow[i]*theta_t_slow - h_fast[i]*theta_t_fast)*
      (1-theta) /
      (tau_square * (1-square(rho)));

    likelihood_var_inverse = likelihood_var_inverse +
      square(1-theta) / (tau_square * (1-square(rho)));
  }
  double likelihood_var = 1.0/likelihood_var_inverse;
  double likelihood_mean = likelihood_var * likelihood_mean_over_variance;

  double h_0_var = tau_square / (1-square(theta));
  double h_0_mean = h_fast[0];

  double posterior_var =
    1.0 / (1.0/likelihood_var + 1.0/h_0_var);
  double posterior_mean =
    posterior_var * (likelihood_mean/likelihood_var +
		     h_0_mean/h_0_var);

  std::vector<double> out = std::vector<double> {posterior_mean, posterior_var};
  return out;
}

std::vector<double> FastOUModel::tau_square_posterior_shape_rate() const
{
  double tau_square_alpha =
    get_tau_square_prior().get_tau_square_shape();
  double tau_square_beta =
    get_tau_square_prior().get_tau_square_scale();

  double rho =
    get_rho().get_continuous_time_parameter();
  double alpha =
    get_alpha().get_discrete_time_parameter(get_delta_t());
  double theta_fast =
    get_theta().get_discrete_time_parameter(get_delta_t());

  const std::vector<double>& h_fast =
    get_sigmas().get_discrete_time_log_sigmas();

  double proposal_alpha = tau_square_alpha + (data_length())/2.0;
  double proposal_beta  = tau_square_beta;

  for (unsigned i=0; i<data_length(); ++i) {
    proposal_beta = proposal_beta +
      0.5/(1.0-square(rho))*square(h_fast[i+1] - alpha*(1-theta_fast) - h_fast[i]*theta_fast);

  }
  proposal_beta = proposal_beta +
    (1-square(theta_fast))/2.0 * square(h_fast[0]-alpha);

  std::vector<double> out = {proposal_alpha, proposal_beta};
  return out;
}

std::vector<double> FastOUModel::tau_square_MLE_shape_rate()
{
  double tau_square_current = get_tau_square().
    get_discrete_time_parameter(get_delta_t());
  double tau_square_tilde_current = log(tau_square_current);

  std::vector<double> tau_square_tilde_mle {tau_square_tilde_current};

  nlopt::opt opt(nlopt::LN_NELDERMEAD, 1);
  std::vector<double> lb {-1.0*HUGE_VAL};
  std::vector<double> ub {HUGE_VAL};
  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);

  double minf;
  opt.set_min_objective(FastOUModel::wrapper_tau_square, this);
  opt.optimize(tau_square_tilde_mle, minf);

  double tau_square_mle = exp(tau_square_tilde_mle[0]);

  double dtau_square = std::abs(tau_square_mle/1000.0);
  double ll = log_likelihood_tau_square(tau_square_mle);
  double ll_plus_dtau_sq = log_likelihood_tau_square(tau_square_mle +
						  dtau_square);
  double ll_minus_dtau_sq = log_likelihood_tau_square(tau_square_mle -
						   dtau_square);
  double tau_square_var = -1.0*square(dtau_square)/(ll_plus_dtau_sq -
						    2.0*ll + ll_minus_dtau_sq);
  double alpha = tau_square_mle / tau_square_var + 2;
  double beta = tau_square_mle * (alpha-1);

  std::vector<double> out {alpha, beta};
  return out;
}

std::vector<double> FastOUModel::tau_square_MLE_mean_variance_tilde_scale()
{
  double tau_square_current = get_tau_square().
    get_discrete_time_parameter(get_delta_t());
  double tau_square_tilde_current = log(tau_square_current);

  double rho_current = get_rho().get_continuous_time_parameter();
  double rho_tilde_current = log( (rho_current + 1.0)/2.0 /
				  (1 - (rho_current + 1.0)/2.0 ) );

  double theta_current = get_theta().get_discrete_time_parameter(get_delta_t());
  double theta_tilde_current = log(theta_current / (1-theta_current));

  std::vector<double> tau_square_tilde_mle_vec {tau_square_tilde_current};

  nlopt::opt opt(nlopt::LN_NELDERMEAD, 1);
  std::vector<double> lb {-1.0*HUGE_VAL};
  std::vector<double> ub {HUGE_VAL};
  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);

  double minf;
  opt.set_min_objective(FastOUModel::wrapper_tau_square, this);
  opt.optimize(tau_square_tilde_mle_vec, minf);

  double tau_square_tilde_mle = tau_square_tilde_mle_vec[0];
  double dtau_square_tilde = 1e-5;

  double ll = log_likelihood_tilde(rho_tilde_current,
				   theta_tilde_current,
				   tau_square_tilde_mle);

  double ll_plus_dtau_sq_tilde =
    log_likelihood_tilde(rho_tilde_current,
			 theta_tilde_current,
			 tau_square_tilde_mle+dtau_square_tilde);

  double ll_minus_dtau_sq_tilde =
    log_likelihood_tilde(rho_tilde_current,
			 theta_tilde_current,
			 tau_square_tilde_mle-dtau_square_tilde);

  double tau_square_tilde_var =
    -1.0*square(dtau_square_tilde) /
    (ll_plus_dtau_sq_tilde -2.0*ll + ll_minus_dtau_sq_tilde);


  std::vector<double> out {tau_square_tilde_mle, tau_square_tilde_var};
  return out;
}

std::vector<double> FastOUModel::rho_posterior_mean_var() const
{
  const std::vector<double>& h_fast =
    get_sigmas().get_discrete_time_log_sigmas();
  const std::vector<double>& h_slow =
    get_ou_model_slow()->get_sigmas().get_discrete_time_log_sigmas();

  const std::vector<int>& ds =
    get_const_vol_model()->get_ds();
  double alpha =
    get_alpha().get_discrete_time_parameter(get_delta_t());
  double theta =
    get_theta().get_discrete_time_parameter(get_delta_t());
  double tau_squared =
    get_tau_square().get_discrete_time_parameter(get_delta_t());

  const std::vector<int>& gammas =
    get_const_vol_model()->get_gammas().get_gammas();
  const std::vector<double>& y_star =
    get_const_vol_model()->get_y_star();

  const std::vector<double>& bs = get_const_vol_model()->get_bs();
  const std::vector<double>& as = get_const_vol_model()->get_as();

  const std::vector<double>& ms =
    get_const_vol_model()->get_gammas().get_mixture_means();

  double R_j = 0;
  double R_j_h_fast = 0;
  double R_j_h_slow = 0;
  double A_j = 0;

  double mean_over_variance = 0;
  double variance_inverse = 0;

  for (unsigned i=0; i<data_length(); ++i) {
    R_j =  sqrt(tau_squared) * ds[i] * exp(ms[gammas[i]]/2.0) *
      (as[gammas[i]] + bs[gammas[i]] * 2.0 * (y_star[i] - ms[gammas[i]]/2.0));
    R_j_h_fast = sqrt(tau_squared) * ds[i] * exp(ms[gammas[i]]/2.0) *
      bs[gammas[i]];
    R_j_h_slow = R_j_h_fast;
    A_j = h_fast[i]*R_j_h_fast + h_slow[i]*R_j_h_slow - R_j;

    mean_over_variance = mean_over_variance +
      (alpha*(1-theta) + h_fast[i]*theta - h_fast[i+1])*A_j/
      (tau_squared);
    variance_inverse = variance_inverse +
      square(A_j)/(tau_squared);
  }
  double variance = 1/variance_inverse;
  double mean = mean_over_variance * variance;

  std::vector<double> out {mean, variance};
  return out;
}

double FastOUModel::rho_cube_poly(double rho) const
{
  return cube(rho)*A_ + square(rho)*B_ + rho*C_ + D_;
}

double FastOUModel::rho_cube_poly_prime(double rho) const
{
  return square(rho)*A_ + B_ + C_;
}

std::vector<double> FastOUModel::rho_MLE_mean_var()
{
  double rho_current = get_rho().get_continuous_time_parameter();
  double rho_tilde_current = log( ((rho_current+1.0)/2.0) /
				  (1 - (rho_current+1.0)/2.0) );

  std::vector<double> rho_tilde_mle {rho_tilde_current};

  nlopt::opt opt(nlopt::LN_NELDERMEAD, 1);
  std::vector<double> lb {-1.0*HUGE_VAL};
  std::vector<double> ub {HUGE_VAL};
  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);

  double minf;
  opt.set_min_objective(FastOUModel::wrapper_rho, this);
  opt.optimize(rho_tilde_mle, minf);

  double rho_mle = 2*exp(rho_tilde_mle[0]) / (exp(rho_tilde_mle[0])+1) - 1;

  double dr = 0.00001;
  set_rho(rho_mle);
  double ll = log_likelihood();
  set_rho(rho_mle+dr);
  double ll_plus_drho = log_likelihood();
  set_rho(rho_mle-dr);
  double ll_minus_drho = log_likelihood();
  double rho_var = -1.0*square(dr)/(ll_plus_drho - 2*ll + ll_minus_drho);

  set_rho(rho_current);
  std::vector<double> out {rho_mle, rho_var};
  return out;

}

std::vector<double> FastOUModel::rho_MLE_mean_var_tilde()
{
  double rho_current = get_rho().get_continuous_time_parameter();
  double rho_tilde_current = log( ((rho_current+1.0)/2.0) /
				  (1 - (rho_current+1.0)/2.0) );

  double theta_current = get_theta().get_discrete_time_parameter(get_delta_t());
  double theta_tilde_current = log(theta_current / (1.0-theta_current));

  double tau_square_current = get_tau_square().get_discrete_time_parameter(get_delta_t());
  double tau_square_tilde_current = log(tau_square_current);

  std::vector<double> rho_tilde_mle {rho_tilde_current};

  nlopt::opt opt(nlopt::LN_NELDERMEAD, 1);
  // nlopt::opt opt(nlopt::LD_MMA, 1);
  std::vector<double> lb {-1.0*HUGE_VAL};
  std::vector<double> ub {HUGE_VAL};
  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);
  opt.set_xtol_rel(1e-3);
  double minf;
  opt.set_min_objective(FastOUModel::wrapper_rho, this);
  opt.optimize(rho_tilde_mle, minf);

  long double dr = 1.0;
  double ll = log_likelihood_tilde(rho_tilde_mle[0],
				   theta_tilde_current,
				   tau_square_tilde_current);
  double ll_plus_drho = log_likelihood_tilde(rho_tilde_mle[0] + dr,
					     theta_tilde_current,
					     tau_square_tilde_current);
  double ll_minus_drho = log_likelihood_tilde(rho_tilde_mle[0] - dr,
					      theta_tilde_current,
					      tau_square_tilde_current);

  double rho_var = -1.0*square(dr)/(ll_plus_drho - 2.0*ll + ll_minus_drho);

  set_rho(rho_current);
  std::vector<double> out {rho_tilde_mle[0], rho_var};
  return out;

}

std::vector<double> FastOUModel::rho_theta_tau_square_tilde_MLE()
{
  double rho_current = get_rho().get_continuous_time_parameter();
  double rho_tilde_current = log( ((rho_current+1.0)/2.0) /
				  (1 - (rho_current+1.0)/2.0) );

  double theta_current = get_theta().get_discrete_time_parameter(get_delta_t());
  double theta_tilde_current = log(theta_current / (1.0-theta_current));

  double tau_square_current = get_tau_square().get_discrete_time_parameter(get_delta_t());
  double tau_square_tilde_current = log(tau_square_current);

  std::vector<double> tilde_mle {rho_tilde_current,
      theta_tilde_current,
      tau_square_tilde_current};

  nlopt::opt opt(nlopt::LN_NELDERMEAD, 3);
  std::vector<double> lb {-1.0*HUGE_VAL, -1.0*HUGE_VAL, -1.0*HUGE_VAL};
  std::vector<double> ub {HUGE_VAL, HUGE_VAL, HUGE_VAL};
  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);

  double minf;
  opt.set_min_objective(FastOUModel::wrapper_all, this);
  opt.optimize(tilde_mle, minf);

  return tilde_mle;
}

double FastOUModel::
wrapper_rho(const std::vector<double> &x,
	    std::vector<double> &grad,
	    void * data)
{
  FastOUModel * params =
    reinterpret_cast<FastOUModel*>(data);
  return  params->MLE_min_rho(x,grad);
}

double FastOUModel::
wrapper_tau_square(const std::vector<double> &x,
		   std::vector<double> &grad,
		   void * data)
{
  FastOUModel * params =
    reinterpret_cast<FastOUModel*>(data);
  return  params->MLE_min_tau_square(x,grad);
}

double FastOUModel::
wrapper_all(const std::vector<double> &x,
	    std::vector<double> &grad,
	    void * data)
{
  FastOUModel * params =
    reinterpret_cast<FastOUModel*>(data);
  return  params->MLE_min_all(x,grad);
}

double FastOUModel::MLE_min_rho(const std::vector<double> &x,
				std::vector<double> &grad)
{
  double rho_tilde = x[0];
  double rho = 2*(exp(rho_tilde) / (exp(rho_tilde)+1)) - 1;

  if (!grad.empty()) {
    grad[0] = rho_deriv_analytic_tilde_scale(rho_tilde);
  }

  double ll = -1.0* (log_likelihood_rho(rho)
		     + log(2) + rho_tilde
 		     - 2*log(exp(rho_tilde)+1));

  // std::cout << "rho = " << rho << "; ll = " << ll << "\n";
  return ll;
}

double FastOUModel::MLE_min_tau_square(const std::vector<double> &x,
				       std::vector<double> &grad)
{
  if (!grad.empty()) {
    // grad is always empty!
    // should throw an exception if not empty
  }
  double tau_square_tilde = x[0];
  double tau_square = exp(tau_square_tilde);

  double ll = -1.0* (log_likelihood_tau_square(tau_square)
		     + tau_square_tilde);

  // std::cout << "rho = " << rho << "; ll = " << ll << "\n";
  return ll;
}

double FastOUModel::MLE_min_all(const std::vector<double> &x,
				std::vector<double> &grad)
{
  if (!grad.empty()) {
    // grad is always empty!
    // should throw an exception if not empty
  }
  double rho_tilde = x[0];
  double theta_tilde = x[1];
  double tau_square_tilde = x[2];

  double ll = -1.0* (log_likelihood_tilde(rho_tilde,
					  theta_tilde,
					  tau_square_tilde));

  // std::cout << "rho = " << rho << "; ll = " << ll << "\n";
  return ll;
}

// ==================== SV MODEL =============================
StochasticVolatilityModel::
StochasticVolatilityModel(const OpenCloseData& data,
			  double delta_t)
  : BaseModel(delta_t),
    observational_model_(new ObservationalModel(data,
						delta_t)),
    const_vol_model_(new ConstantVolatilityModel(observational_model_,
						 delta_t)),
    ou_model_(new OUModel(const_vol_model_,
			  delta_t))
{
  observational_model_->set_const_vol_model(const_vol_model_);
  const_vol_model_->set_ou_model(ou_model_);
}

StochasticVolatilityModel::~StochasticVolatilityModel()
{
  delete ou_model_;
  delete const_vol_model_;
  delete observational_model_;
}

unsigned StochasticVolatilityModel::data_length() const
{
  return observational_model_->data_length();
}

double StochasticVolatilityModel::log_likelihood() const
{
  double log_likelihood =
    observational_model_->log_likelihood()+
    const_vol_model_->log_likelihood()+
    ou_model_->log_likelihood();

  return log_likelihood;
}

void StochasticVolatilityModel::
generate_data(double time,
	      gsl_rng * rng)
{
  long int N = floor(time / get_delta_t());
  long int dt = get_delta_t();
  std::cout << dt << std::endl;
  std::cout << N << std::endl;

  double tau_square = get_ou_model()->get_tau_square().
    get_discrete_time_parameter(get_delta_t());

  double alpha = get_ou_model()->get_alpha().
    get_discrete_time_parameter(get_delta_t());
  double alpha_hat = get_ou_model()->get_alpha().
    get_continuous_time_parameter();
  double rho = get_ou_model()->get_rho().
    get_discrete_time_parameter(get_delta_t());
  std::cout << "RHO IS " << rho << std::endl;
  double mu = get_const_vol_model()->get_mu().
    get_discrete_time_parameter(get_delta_t());

  double theta = get_ou_model()->get_theta().
    get_discrete_time_parameter(get_delta_t());

  double current_log_sigma_hat = alpha_hat;
  double current_price = 100.0;

  arma::vec mean = arma::zeros<arma::vec> (2);
  arma::mat cov = arma::ones<arma::mat> (2,2);
  cov(0,1) = rho;
  cov(1,0) = rho;

  // RECORD THE RESULTS
  std::ofstream simulation ("/home/gdinolov/Research/SV-with-leverage/simulated-data/simulation-1/no-noise/simulated-prices-and-returns-no-noise-RAW.csv");

  // header
  simulation << "price, log.sigma.hat\n";
  simulation << current_price << "," << current_log_sigma_hat << "\n";

  arma::vec epsilon_nu = arma::zeros<arma::mat> (2,2);
  double epsilon = 0;
  double nu = 0;
  double sigma_hat = 0;
  double log_sigma = 0;

  for (long int i=1; i<N+1; ++i) {
    epsilon_nu = rmvnorm(rng,2,mean,cov);
    epsilon = epsilon_nu(0);
    nu = epsilon_nu(1);

    sigma_hat = exp(current_log_sigma_hat);

    current_price = exp(log(current_price) + mu +
			sqrt(get_delta_t())*sigma_hat*epsilon);

    log_sigma = current_log_sigma_hat + 0.5*log(get_delta_t());

    log_sigma =
      alpha + theta*(log_sigma - alpha) + sqrt(tau_square)*nu;

    current_log_sigma_hat = log_sigma - 0.5*log(get_delta_t());

    simulation << current_price << "," << current_log_sigma_hat << "\n";

    // std::cout << current_log_price << " "
    // 	      << current_log_sigma_hat_slow << " "
    // 	      << current_log_sigma_hat_fast << std::endl;
    if ( i % 1000000 == 0 ) {
      std::cout << i << std::endl;
    }
  }
  simulation.close();
}

// ==================== MULTIFACTOR SV MODEL =============================
MultifactorStochasticVolatilityModel::
MultifactorStochasticVolatilityModel(const OpenCloseData& data,
				     double delta_t,
				     double theta_hat_fast_mean,
				     double theta_hat_fast_std_dev,
				     double theta_hat_slow_mean,
				     double theta_hat_slow_std_dev)
  : BaseModel(delta_t),
    observational_model_(new ObservationalModel(data,
						delta_t)),
    const_multifactor_vol_model_(new ConstantMultifactorVolatilityModel(observational_model_,
									delta_t)),
    ou_model_slow_(new OUModel(const_multifactor_vol_model_,
			       delta_t)),
    ou_model_fast_(new FastOUModel(const_multifactor_vol_model_,
				   ou_model_slow_,
				   delta_t))
{
  observational_model_->set_const_vol_model(const_multifactor_vol_model_);
  const_multifactor_vol_model_->set_ou_model_slow(ou_model_slow_);
  const_multifactor_vol_model_->set_ou_model_fast(ou_model_fast_);

  // adjusting the OU models
  ou_model_fast_->set_theta_hat_mean(theta_hat_fast_mean);
  ou_model_fast_->set_theta_hat(theta_hat_fast_mean);
  ou_model_fast_->set_theta_hat_std_dev(theta_hat_fast_std_dev);

  ou_model_slow_->set_theta_hat_mean(theta_hat_slow_mean);
  ou_model_slow_->set_theta_hat(theta_hat_slow_mean);
  ou_model_slow_->set_theta_hat_std_dev(theta_hat_slow_std_dev);
}

MultifactorStochasticVolatilityModel::~MultifactorStochasticVolatilityModel()
{
  delete ou_model_fast_;
  delete ou_model_slow_;
  delete const_multifactor_vol_model_;
  delete observational_model_;
}

unsigned MultifactorStochasticVolatilityModel::data_length() const
{
  return observational_model_->data_length();
}

double MultifactorStochasticVolatilityModel::log_likelihood() const
{
  double log_likelihood =
    observational_model_->log_likelihood() +
    const_multifactor_vol_model_->log_likelihood() +
    ou_model_slow_->log_likelihood() +
    ou_model_fast_->log_likelihood();
  return log_likelihood;
}

void MultifactorStochasticVolatilityModel::
generate_data(double time,
	      gsl_rng * rng)
{
  long int N = floor(time / get_delta_t());
  long int dt = get_delta_t();
  std::cout << dt << std::endl;
  std::cout << N << std::endl;

  double tau_square_slow = get_ou_model_slow()->get_tau_square().
    get_discrete_time_parameter(get_delta_t());

  // MAKING FAST INNOVATION 6 TIMES BIGGER
  double tau_square_hat_fast = get_ou_model_fast()->get_tau_square().
    get_continuous_time_parameter();
  get_ou_model_fast()->set_tau_square_hat(36.0*tau_square_hat_fast);
  // ==================================
  double tau_square_fast = get_ou_model_fast()->get_tau_square().
    get_discrete_time_parameter(get_delta_t());
  double alpha = get_ou_model_fast()->get_alpha().
    get_discrete_time_parameter(get_delta_t());
  double alpha_hat = get_ou_model_fast()->get_alpha().
    get_continuous_time_parameter();
  double rho = get_ou_model_fast()->get_rho().
    get_discrete_time_parameter(get_delta_t());
  double mu = get_constant_vol_model()->get_mu().
    get_discrete_time_parameter(get_delta_t());

  double theta_slow = get_ou_model_slow()->get_theta().
    get_discrete_time_parameter(get_delta_t());
  double theta_fast = get_ou_model_fast()->get_theta().
    get_discrete_time_parameter(get_delta_t());

  double current_log_sigma_hat_slow = alpha_hat;
  double current_log_sigma_hat_fast = alpha_hat;
  double current_price = 100.0;

  arma::vec mean = arma::zeros<arma::vec> (2);
  arma::mat cov = arma::ones<arma::mat> (2,2);
  cov(0,1) = rho;
  cov(1,0) = rho;

  std::cout << "cov=" << cov << "\n";

  // RECORD THE RESULTS
  std::ofstream simulation ("/home/gdinolov/Research/SV-with-leverage/simulated-data/simulation-1/no-noise/simulated-prices-and-returns-no-noise-RAW.csv");

  // header
  simulation << "price, log.sigma.hat.slow, log.sigma.hat.fast\n";
  simulation << current_price << "," << current_log_sigma_hat_slow
	  << "," << current_log_sigma_hat_fast << "\n";

  for (long int i=1; i<N+1; ++i) {
    double nu_slow = gsl_ran_gaussian(rng, 1.0);
    arma::vec epsilon_nu_fast = rmvnorm(rng,2,mean,cov);
    double epsilon = epsilon_nu_fast(0);
    double nu_fast = epsilon_nu_fast(1);

    double sigma_hat_slow =
      exp(current_log_sigma_hat_slow);
    double sigma_hat_fast =
      exp(current_log_sigma_hat_fast);

    current_price = exp(log(current_price) + mu +
			sqrt(get_delta_t())*
			sqrt(sigma_hat_fast)*
			sqrt(sigma_hat_slow)*
			epsilon);

    double log_sigma_slow = current_log_sigma_hat_slow + 0.5*log(get_delta_t());
    double log_sigma_fast = current_log_sigma_hat_fast + 0.5*log(get_delta_t());

    log_sigma_slow =
      alpha + theta_slow*(log_sigma_slow - alpha) + sqrt(tau_square_slow)*nu_slow;

    log_sigma_fast =
      alpha + theta_fast*(log_sigma_fast - alpha) + sqrt(tau_square_fast)*nu_fast;

    current_log_sigma_hat_slow = log_sigma_slow - 0.5*log(get_delta_t());
    current_log_sigma_hat_fast = log_sigma_fast - 0.5*log(get_delta_t());

    simulation << current_price << "," << current_log_sigma_hat_slow
	       << "," << current_log_sigma_hat_fast << "\n";

    // std::cout << current_log_price << " "
    // 	      << current_log_sigma_hat_slow << " "
    // 	      << current_log_sigma_hat_fast << std::endl;
    if ( i % 1000000 == 0 ) {
      std::cout << i << std::endl;
    }
  }
  simulation.close();
}


// ==================== MULTIFACTOR SV MODEL WITH JUMPS  ===================
SVModelWithJumps::
SVModelWithJumps(const OpenCloseData& data,
		 double delta_t,
		 double theta_hat_fast_mean,
		 double theta_hat_fast_std_dev,
		 double theta_hat_slow_mean,
		 double theta_hat_slow_std_dev)
  : BaseModel(delta_t),
    observational_model_(new ObservationalModel(data,
						delta_t)),
    const_multifactor_vol_model_with_jumps_(new ConstantMultifactorVolatilityModelWithJumps(observational_model_,
									delta_t)),
    ou_model_slow_(new OUModel(const_multifactor_vol_model_with_jumps_,
			       delta_t)),
    ou_model_fast_(new FastOUModel(const_multifactor_vol_model_with_jumps_,
				   ou_model_slow_,
				   delta_t))
{
  observational_model_->set_const_vol_model(const_multifactor_vol_model_with_jumps_);
  const_multifactor_vol_model_with_jumps_->set_ou_model_slow(ou_model_slow_);
  const_multifactor_vol_model_with_jumps_->set_ou_model_fast(ou_model_fast_);

  // adjusting the OU models
  ou_model_fast_->set_theta_hat_mean(theta_hat_fast_mean);
  ou_model_fast_->set_theta_hat(theta_hat_fast_mean);
  ou_model_fast_->set_theta_hat_std_dev(theta_hat_fast_std_dev);

  ou_model_slow_->set_theta_hat_mean(theta_hat_slow_mean);
  ou_model_slow_->set_theta_hat(theta_hat_slow_mean);
  ou_model_slow_->set_theta_hat_std_dev(theta_hat_slow_std_dev);
}

SVModelWithJumps::
SVModelWithJumps(const OpenCloseData& data,
		 double delta_t,
		 double theta_hat_fast_mean,
		 double theta_hat_fast_std_dev,
		 double theta_hat_slow_mean,
		 double theta_hat_slow_std_dev,
		 double tau_square_hat_fast_mean,
		 double tau_square_hat_fast_sd,
		 double tau_square_hat_slow_mean,
		 double tau_square_hat_slow_sd)
  : BaseModel(delta_t),
    observational_model_(new ObservationalModel(data,
						delta_t)),
    const_multifactor_vol_model_with_jumps_(
      new ConstantMultifactorVolatilityModelWithJumps(observational_model_,delta_t)),
    ou_model_slow_(new OUModel(const_multifactor_vol_model_with_jumps_,
			       delta_t)),
    ou_model_fast_(new FastOUModel(const_multifactor_vol_model_with_jumps_,
				   ou_model_slow_,
				   delta_t))
{
  observational_model_->set_const_vol_model(const_multifactor_vol_model_with_jumps_);
  const_multifactor_vol_model_with_jumps_->set_ou_model_slow(ou_model_slow_);
  const_multifactor_vol_model_with_jumps_->set_ou_model_fast(ou_model_fast_);

  // adjusting the OU models
  // fast theta
  ou_model_fast_->set_theta_hat_mean(theta_hat_fast_mean);
  ou_model_fast_->set_theta_hat(theta_hat_fast_mean);
  ou_model_fast_->set_theta_hat_std_dev(theta_hat_fast_std_dev);
  // slow theta
  ou_model_slow_->set_theta_hat_mean(theta_hat_slow_mean);
  ou_model_slow_->set_theta_hat(theta_hat_slow_mean);
  ou_model_slow_->set_theta_hat_std_dev(theta_hat_slow_std_dev);

  // fast tau^2
  ou_model_fast_->set_tau_square_hat_mean(tau_square_hat_fast_mean);
  ou_model_fast_->set_tau_square_hat(tau_square_hat_fast_mean);
  ou_model_fast_->set_tau_square_hat_std_dev(tau_square_hat_fast_sd);

  ou_model_slow_->set_tau_square_hat_mean(tau_square_hat_slow_mean);
  ou_model_slow_->set_tau_square_hat(tau_square_hat_slow_mean);
  ou_model_slow_->set_tau_square_hat_std_dev(tau_square_hat_slow_sd);
}

SVModelWithJumps::
~SVModelWithJumps()
{
  delete ou_model_fast_;
  delete ou_model_slow_;
  delete const_multifactor_vol_model_with_jumps_;
  delete observational_model_;
}

unsigned SVModelWithJumps
::data_length() const
{
  return observational_model_->data_length();
}

double SVModelWithJumps
::log_likelihood() const
{
  double log_likelihood =
    observational_model_->log_likelihood() +
    const_multifactor_vol_model_with_jumps_->log_likelihood() +
    ou_model_slow_->log_likelihood() +
    ou_model_fast_->log_likelihood();
  return log_likelihood;
}

double SVModelWithJumps::
log_likelihood_ous_integrated_vol(double alpha,
				  double rho,
				  double theta_slow,
				  double theta_fast,
				  double tau_square_slow,
				  double tau_square_fast) const
{
  // The dynamical model here will be encoded as
  // Y_t = F' X_t + M_t + v_{gamma_t}/2 * z_t^*
  //
  // X_{t+\Delta} = G_t X_t + mu_t + C_t * W_{t,2}
  //
  // W_{t,2} \sim N(0,I)
  //
  // F' = (0.5, 0.5)
  //
  // M_t = m_{\gamma_t}
  //
  // G_t = (\theta_1(\Delta)    ,          0           )
  //       (\theta_{t,1}(\Delta), \theta_{t,2}(\Delta) )
  //
  // C_t = (\tau_1(\Delta),                  0             )
  //       (      0           \tau_2(\Delta)\sqrt{1-\rho^2})
  //
  //     = C
  //
  // \mu_j' = (\alpha(\Delta), \alpha_{t,2}(\Delta)

  // posterior mean and covarance for log-vols

  const std::vector<double>& m =
    const_multifactor_vol_model_with_jumps_->
    get_gammas().get_mixture_means();
  const std::vector<double>& v_square =
    const_multifactor_vol_model_with_jumps_->
    get_gammas().get_mixture_variances();
  const std::vector<int>& gammas =
    const_multifactor_vol_model_with_jumps_->
    get_gammas().get_gammas();
  const std::vector<double>& y_star =
    const_multifactor_vol_model_with_jumps_->
    get_y_star();

  // FORWARD FILTER
  // GOAL: Integrate out all X_j := (h_{j,1}, h_{j,2}) to evauate
  // p(y_1, \ldots, y_n(\Delta)), as well as derive the posterior
  // mean, covariance for p(X_j | y_1, \ldots, y_n(\Delta))

  // The notation will be
  //
  // Posterior for X_j: p(X_{j} | y_1, \ldots, y_j) = N(X_j | N_j, T^2_j)
  // One-step ahead preditive for X_j: p(X_{j} | y_1, \ldots, y_{j-1}) = N(X_j | U_j, S^2_j)
  //
  // At the beginning of each iteration j, we assume we have
  // p(X_{j-1} | y_1, \ldots, y_{j-1}) = N(X_{j-1} | N_{j-1}, T^2_{j=1})
  //
  // ==> p(X_j | y_1, \ldots, y_{j-1}) = \int p(X_j | X_{j-1}) p(X_{j-1} | y_1, \ldots, y_{j-1}) dX_{j-1}
  //
  //                                   = N(X_j | G_{j-1} N_{j-1} + mu_{j-1}, G_{j-1}*T^2_{j-1}*G'_{j-1} + C_t C'_t)
  //                                   = N(X_j | G_{j-1} N_{j-1} + mu_{j-1}, G_{j-1}*T^2_{j-1}*G'_{j-1} + C C'), b/c C_t=C is time independent
  //                                   = N(X_j | U_j, S_j^2)
  //
  // ==> p(y_j | y_1, \ldots, y_{j-1}) = \int p(y_j | X_j) p(X_j | y_1, \ldots, y_{j-1}) dX_j
  //
  //                                   = \int N(y_j |  F' X_t + M_j, v^2_{gamma_j}/4) N(X_j | U_j, S^2_j) dX_j
  //                                   = N(y_j | F'U_j + M_j, v^2_{gamma_j}/4 + F' S^2_j F)
  //                                   = N(y_j | f_j, q_j)
  //
  // (X_j) \sim N( (U_j),  ( S_j^2   | S_j^2 F ) )
  // (y_j)       ( (f_j)   ( F'S_j^2 | q_j     ) )
  //
  // p(X_j | y_j) = N( X_j | N_j   = U_j + S_j^2 F * (q_j)^{-1} (y_j - f_j)
  //                         T_j^2 = S_j^2 - S_j^2 F * (q_j)^{-1} F' S_j^2
  //
  // To double check the COV matrix:
  // p(y_j | X_j) = N(y_j | F' X_j + M_j, v^2_{gamma_j}/4)
  // v^2_{gamma_j}/4 = v^2_{gamma_j}/4 + F' S^2_j F - F' S_j^2 * S_j^{-2} * S_j^2 F

  // current mean level;

  // At the start of the FF, we need the prior for the first vol
  // elements.
  // We se p(X_1) = N(X_1 | U_1, S_1^2) = N( (h_{1,1}) | (alpha(\Delta)), ( \tau^2_1(\Delta)/(1-\theta_1^2(\Delta)), 0                                      ) )
  //                                       ( (h_{1,2}) | (alpha(\Delta))  ( 0                                      , \tau^2_2(\Delta)/(1-\theta_2^2(\Delta))) )
  //
  //

  // current posterior mean level
  double n_current_fast =
    ou_model_fast_->
    get_alpha().get_discrete_time_parameter(get_delta_t());
  double n_current_slow =
    ou_model_slow_->
    get_alpha().get_discrete_time_parameter(get_delta_t());

  arma::vec N_current(2);
  N_current(0) = n_current_slow;
  N_current(1) = n_current_fast;

  // current posterior variance level with D_0;
  arma::mat T_current_sq = arma::zeros<arma::mat> (2,2);

  T_current_sq(0,0) =
    tau_square_slow / (1 - square(theta_slow));
  T_current_sq(1,1) =
    tau_square_fast / (1 - square(theta_fast));

  // freed at end of loop
  std::vector<arma::mat> Taus_squared (data_length()+1);
  std::vector<arma::vec> Ns (data_length()+1);
  
  arma::vec M_j = arma::vec (2);
  arma::mat G_j_minus_one = arma::zeros<arma::mat> (2,2);
  arma::mat G_j = arma::zeros<arma::mat> (2,2);
  arma::vec h_t = arma::zeros<arma::vec> (2);
  arma::vec h_tp1 = arma::zeros<arma::vec> (2);

  arma::mat F = arma::mat (2,1);
  F(0,0) = 0.5;
  F(1,0) = 0.5;
  arma::mat FFt = F*F.t();

  arma::mat CCt = arma::zeros<arma::mat> (2,2);
  CCt(0,0) = tau_square_slow;
  CCt(1,1) = tau_square_fast * (1-rho*rho);
  // std::cout << "tau_square_fast = " << tau_square_fast << "\n";
  // std::cout << "Z=" << Z << "\n";

  arma::mat S_square_current_inv = arma::zeros<arma::mat> (2,2);
  arma::vec U_current = arma::zeros<arma::mat> (2);
  arma::mat S_square_current = arma::zeros<arma::mat> (2,2);

  arma::mat f_j = arma::zeros<arma::mat> (1,1);
  arma::mat q_j = arma::zeros<arma::mat> (1,1);
  arma::vec mu_j_minus_one = arma::vec (2);
  arma::vec mu_j = arma::vec (2);

  arma::mat Tau_current_sq_inv = arma::zeros<arma::mat> (2,2);
  arma::mat Tau_current_sq = arma::zeros<arma::mat> (2,2);
  arma::mat omega_ts_sq_inv = arma::mat (2,2);


  arma::mat Zt (2,2);
  arma::mat Z (2,2);
  arma::mat Epsilon (2,2);

  double log_likelihood = 0;

  // FORWARD FILTER
  for (unsigned i=0; i<data_length(); ++i) {
    U_current = arma::zeros<arma::vec> (2);
    S_square_current = arma::zeros<arma::mat> (2,2);
    G_j_minus_one = arma::zeros<arma::mat> (2,2);
    mu_j_minus_one = arma::zeros<arma::vec> (2);
    CCt(0,0) = tau_square_slow;
    CCt(1,1) = tau_square_fast;
    
    if (i==0) {
      // Step 1: One-step ahead predictive for X_j:
      U_current(0) = theta_slow*alpha + alpha*(1-theta_slow);
      U_current(1) = theta_fast*alpha + alpha*(1-theta_fast);
      
      S_square_current(0,0) = square(theta_slow)*1.0 + tau_square_slow;
      S_square_current(1,1) = square(theta_fast)*1.0 + tau_square_fast;
    } else {
      G_j_minus_one(0,0) = theta_slow;
      G_j_minus_one(1,0) = 
	ou_model_fast_->theta_j_one(i-1,
				    gammas[i-1]);
      G_j_minus_one(1,1) = 
	ou_model_fast_->theta_j_two(i-1,
				    gammas[i-1]);
      mu_j_minus_one(0) = alpha*(1-theta_slow);
      mu_j_minus_one(1) = 
	ou_model_fast_->alpha_j(i-1,
				gammas[i-1]);
      
      U_current = G_j_minus_one*Ns[i-1] + mu_j_minus_one;
      S_square_current = G_j_minus_one*Taus_squared[i-1]*G_j_minus_one.t() + CCt;
    }

    // multivariate approach for the posterior
    // Step 2: One-step ahead predictive for y_j
    // p(y_j | y_1, \ldots y_{j-1}) = \int N(y_j | F'X_j + 0.5m[gammas[j]], v[gammas[j]]^2/4)
    //                                     N(h_j | U_j, S_j^2) dX_j
    // p(y_j | ... ) = N(y_j | F'U_j + 0.5m[gammas[j]], F' S_j^2 F + v[gammas[j]]^2/4)
    //              := N(y_j | f_j, q_j)

    arma::vec F_i = F.t() * U_current + 0.5*m[gammas[i]];
    arma::mat Q_i = v_square[gammas[i]]/4.0 + F.t() * S_square_current * F;
    double f_i = F_i(0);
    double q_i = Q_i(0,0);

    log_likelihood = log_likelihood +
      log(gsl_ran_gaussian_pdf(y_star[i]-f_i,sqrt(q_i)));

    // Step 3: Joint density of (h_j,y_j | y_1, ... y_{j-1})
    // (X_j) \sim N( (U_j), (S_j^2  | Sigma' ) )
    // (y_j)       ( (f_j), (Sigma  | q_j    ) )
    //
    // => Var[y_j | h_j] = v[gammas[j]]^2/4 = q_j - Simga inv(S_j^2) Sigma'
    //                   =                  = (F' S_j^2 F + v[gammas[j]]^2/4) - Sigma inv(S_j^2) Sigma'
    // => Sigma = F' S_j^2
    // 
    // => E[h_j | y_j] = U_j   + Sigma' inv(q_j) (y_j - f_j) = U_j   + S_j^2 F * inv(q_j) * (y_j-f_j)
    //  Var[h_j | y_j] = S_j^2 - Sigma' inv(q_j) Sigma       = S_j^2 - S_j^2 F * inv(q_j) * F' * S_j^2
    //                                                       = S_j^2 (1 -  F * inv(q_j) * F' * S_j^2)
    
    arma::mat posterior_cov = S_square_current - S_square_current * (FFt/q_i) * S_square_current;
    arma::vec posterior_mean = U_current + S_square_current * F * (1.0/q_i) * (y_star[i]-f_i);

    // double posterior_var = s_square_current*v_square[gammas[i]]/
    //   (s_square_current + v_square[gammas[i]]);
    // double posterior_mean =
    //   s_square_current/(s_square_current+v_square[gammas[i]]) *
    //   (2*y_star[i] - h_fast[i] - m[gammas[i]]) +
    //   v_square[gammas[i]]/(s_square_current+v_square[gammas[i]]) *
    //   u_current;
      
    T_current_sq = posterior_cov;
    Taus_squared[i] = T_current_sq;
    
    N_current = posterior_mean;
    Ns[i] = N_current;
  }
  
  // JUST ONE MORE STEP FORWARD TO THE u_{t+1} and S_{t+1}^2
  G_j_minus_one = arma::zeros<arma::mat> (2,2);
  G_j_minus_one(0,0) = theta_slow;
  G_j_minus_one(1,0) = 
    ou_model_fast_->theta_j_one(data_length()-1,
				gammas[data_length()-1]);
  G_j_minus_one(1,1) = 
    ou_model_fast_->theta_j_two(data_length()-1,
				gammas[data_length()-1]);
  mu_j_minus_one(0) = alpha*(1-theta_slow);
  mu_j_minus_one(1) = 
    ou_model_fast_->alpha_j(data_length()-1,
			    gammas[data_length()-1]);
  U_current = G_j_minus_one * N_current + mu_j_minus_one;
  S_square_current  = G_j_minus_one * T_current_sq * G_j_minus_one.t() + CCt;
  //
  T_current_sq = S_square_current;
  N_current = U_current;

  Taus_squared[data_length()] = T_current_sq;
  Ns[data_length()] = N_current;
  
  return log_likelihood;
}

double SVModelWithJumps::
log_likelihood_ous_integrated_filtered_prices(double alpha,
					      double rho,
					      double theta_slow,
					      double theta_fast,
					      double tau_square_slow,
					      double tau_square_fast,
					      double xi_square,
					      double mu) const
{
  // We seek p(Y_1, \ldots, Y_n(\Delta) | -- ) assuming p(log(S_0)) = N(log(S_0) | eta, kappa^2)
  // p(Y_n(\Delta), \ldots, Y_1 | -- ) =
  //         p(Y_n(\Delta) | Y_{n(\Delta)-1}, \ldots, Y_1, --)
  //         p(Y_{n(\Delta)-1} | Y_{n(\Delta)-2}, \ldots, Y_1)
  //         ...
  //         p(Y_1)
  //
  //
  // For each p(Y_j | Y_{j-1},\ldots,Y_1), assume we have
  // p(log(S_{j-1}) | Y_{j-1},\ldots,Y_1) = N(log(S_{j-1}) | v_{j-1}, tau_{j-1}^2).
  //
  // Now we compute p(log(S_{j}) | Y_{j-1},\ldots,Y_1) = N(log(S_j) | u_j, s_j^2) from the dynamical model.
  //
  // Then
  //
  // p(Y_j | Y_{j-1},\ldots,Y_1) = \int p(Y_j |log(S_j)) p(log(S_j)|Y_{j-1},\ldots,Y_1) dlog(S_j),
  // which contributes to the likelihood.
  //
  // Finally,
  // p(log(S_j) | Y_j, \ldots, Y_1) \propto p(Y_j | log(S_j)) p(log(S_j) | Y_{j-1}, \ldots Y_1) =
  // N(log(S_j) | v_j, tau_j^2)

  const std::vector<SigmaSingletonParameter>& sigmas_slow =
    ou_model_slow_->
    get_sigmas().get_sigmas();
  const std::vector<SigmaSingletonParameter>& sigmas_fast =
    ou_model_fast_->
    get_sigmas().get_sigmas();
  const std::vector<double>& jump_sizes =
    const_multifactor_vol_model_with_jumps_->
    get_jump_sizes();
  const std::vector<double>& deltas =
    observational_model_->
    get_deltas();

  // p(log(S_0)) = N(log(S_0) | eta, kappa^2)
  double eta = observational_model_->get_data_element(0).get_open();
  double kappa_sq = xi_square*100;

  // posterior
  // p(log(S_j) | Y_1, \ldots, Y_{j-1}, Y_j) = N(log(S_j) | v_j, \tau_j^2)
  //
  // one-step ahead predictive
  // p(log(S_j) | Y_1, \ldots, Y_{j-1}) = N(log(S_j) | u_j, s_j^2)

  double epsilon_t_2 = 0;
  double u_current = 0;
  double s_current_sq = 0;
  double log_P_t = 0;

  double v_current = 0;
  double tau_current_sq = 0;

  double sigma_t_slow = 0;
  double sigma_t_fast = 0;
  double sigma_tp1_fast = 0;

  double log_likelihood = 0;

  for (unsigned i=0; i<data_length(); ++i) {
    if (i == 0) {
      // the time corresponds to (i+1)*delta_t b/c the first element in
      // the log(S) vector corresponds to log(S_0)
      sigma_t_slow =
	(sigmas_slow[i].
	 get_discrete_time_parameter(get_delta_t()));
      sigma_t_fast =
	(sigmas_fast[i].
	 get_discrete_time_parameter(get_delta_t()));
      sigma_tp1_fast =
	(sigmas_fast[i+1].
	 get_discrete_time_parameter(get_delta_t()));

      epsilon_t_2 = (log(sigma_tp1_fast)
		     - alpha
		     - theta_fast*(log(sigma_t_fast)-alpha))/sqrt(tau_square_fast);

      // One step-ahead predictive p(log(S_1)) =
      //\int p(log(S_1) | log(S_0))p(log(S_0))dlog(S_0)
      // since we don't have Y_0
      // ===============================
      u_current = mu +
      eta +
      jump_sizes[i] +
      sqrt(sigma_t_slow*sigma_t_fast)*rho*epsilon_t_2;
      s_current_sq = kappa_sq + sigma_t_slow*sigma_t_fast*(1-rho*rho);
      // ===============================

      // p(Y_1)= \int p(Y_1 | log(S_1)) p(los(S_1)) dlog(S_1) = \int N(Y_1|log(S_1),xi^2)N(log(S_1)|u_1,s_1^2)
      // = N(Y_1 | u_current, s_current_sq + xi_square)
      // ===============================
      log_P_t = observational_model_->get_data_element(i).get_close();
      log_likelihood = log_likelihood
	+ dnorm(log_P_t, u_current, sqrt(xi_square/deltas[i] + s_current_sq), 1);
      // ===============================

      // posterior p(log(S_1) | Y_1, log(S_0)) \propto p(Y_1 | log(S_1), log(S_0))p(log(S_1) | log(S_0))
      //                      \propto N(Y_1 | log(S_1), \xi_square) N(log(S_1) | u_current, s_current_sq)
      //                      \propto N(log(S_1) | v_current, tau_current_sq)
      // ===============================
      v_current = u_current/(s_current_sq*deltas[i]/xi_square + 1.0) +
	log_P_t/(xi_square/(deltas[i]*s_current_sq) + 1.0);

      tau_current_sq = 1.0/(deltas[i]/xi_square + 1.0/s_current_sq);
      // ===============================


    } else {
      // the time corresponds to (i+1)*delta_t
      sigma_t_slow =
	(sigmas_slow[i].
	 get_discrete_time_parameter(get_delta_t()));
      sigma_t_fast =
	(sigmas_fast[i].
	 get_discrete_time_parameter(get_delta_t()));
      sigma_tp1_fast =
	(sigmas_fast[i+1].
	 get_discrete_time_parameter(get_delta_t()));

      epsilon_t_2 = (log(sigma_tp1_fast)
		     - alpha
		     - theta_fast*(log(sigma_t_fast)-alpha))/sqrt(tau_square_fast);
      // ===============================
      u_current = mu
      + v_current
      + jump_sizes[i]
      + sqrt(sigma_t_slow*sigma_t_fast)*rho*epsilon_t_2;
      s_current_sq = tau_current_sq + sigma_t_slow*sigma_t_fast*(1-rho*rho);
      // ===============================


      // ===============================
      log_P_t = observational_model_->get_data_element(i).get_close();
      log_likelihood = log_likelihood
	+ dnorm(log_P_t, u_current, sqrt(xi_square/deltas[i] + s_current_sq), 1);
      // ===============================

      // posterior p(log(S_1) | Y_1, log(S_0)) \propto p(Y_1 | log(S_1), log(S_0))p(log(S_1) | log(S_0))
      //                      \propto N(Y_1 | log(S_1), \xi_square) N(log(S_1) | u_current, s_current_sq)
      // ===============================
      v_current = u_current/(s_current_sq*deltas[i]/xi_square + 1.0) +
	log_P_t/(xi_square/(deltas[i]*s_current_sq) + 1.0);

      tau_current_sq = 1.0/(deltas[i]/xi_square + 1.0/s_current_sq);
      // ===============================

    }
  }

  return log_likelihood;
}


double SVModelWithJumps::
log_likelihood_ous_integrated_filtered_prices(double rho)
{
  double alpha = get_ou_model_fast()->get_alpha().
    get_discrete_time_parameter(get_delta_t());
  double theta_slow = get_ou_model_slow()->get_theta().
    get_discrete_time_parameter(get_delta_t());
  double theta_fast = get_ou_model_fast()->get_theta().
    get_discrete_time_parameter(get_delta_t());
  double tau_square_slow = get_ou_model_slow()->get_tau_square().
    get_discrete_time_parameter(get_delta_t());
  double tau_square_fast = get_ou_model_fast()->get_tau_square().
    get_discrete_time_parameter(get_delta_t());
  double xi_square = get_observational_model()->get_xi_square().
    get_continuous_time_parameter();
  double mu = get_constant_vol_model()->get_mu().
    get_discrete_time_parameter(get_delta_t());

  double out = log_likelihood_ous_integrated_filtered_prices(alpha,
  							     rho,
  							     theta_slow,
  							     theta_fast,
  							     tau_square_slow,
  							     tau_square_fast,
  							     xi_square,
  							     mu);

  return out;
}

double SVModelWithJumps::
log_likelihood_ous_integrated_filtered_prices(double rho,
					      double xi_square,
					      double mu)
{
  double alpha = get_ou_model_fast()->get_alpha().
    get_discrete_time_parameter(get_delta_t());
  double theta_slow = get_ou_model_slow()->get_theta().
    get_discrete_time_parameter(get_delta_t());
  double theta_fast = get_ou_model_fast()->get_theta().
    get_discrete_time_parameter(get_delta_t());
  double tau_square_slow = get_ou_model_slow()->get_tau_square().
    get_discrete_time_parameter(get_delta_t());
  double tau_square_fast = get_ou_model_fast()->get_tau_square().
    get_discrete_time_parameter(get_delta_t());


  double out = log_likelihood_ous_integrated_filtered_prices(alpha,
							     rho,
							     theta_slow,
							     theta_fast,
							     tau_square_slow,
							     tau_square_fast,
							     xi_square,
							     mu);
  return out;
}

std::vector<double> SVModelWithJumps::rho_tilde_MLE_mean_var()
{
  double rho_current = get_ou_model_fast()->get_rho().get_continuous_time_parameter();
  double rho_tilde_current = log( ((rho_current+1.0)/2.0) /
				  (1 - (rho_current+1.0)/2.0) );

  std::vector<double> rho_tilde_mle {rho_tilde_current};

  nlopt::opt opt(nlopt::LN_NELDERMEAD, 1);
  opt.set_ftol_abs(1e-4);
  std::vector<double> lb {-1.0*HUGE_VAL};
  std::vector<double> ub {HUGE_VAL};
  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);

  double minf;
  opt.set_min_objective(SVModelWithJumps::wrapper_rho, this);
  opt.optimize(rho_tilde_mle, minf);

  double dr = 0.01;

  double ll =
    log_likelihood_ous_integrated_filtered_prices(get_ou_model_fast()->get_rho().
						  tilde_to_nominal(rho_tilde_mle[0]))
    + log(2.0) + rho_tilde_mle[0] - 2.0*log(exp(rho_tilde_mle[0])+1);

  double ll_plus_drho =
    log_likelihood_ous_integrated_filtered_prices(get_ou_model_fast()->get_rho().
						  tilde_to_nominal(rho_tilde_mle[0]+dr))
    + log(2.0) + (rho_tilde_mle[0]+dr) - 2.0*log(exp(rho_tilde_mle[0]+dr)+1);

  double ll_minus_drho =
    log_likelihood_ous_integrated_filtered_prices(get_ou_model_fast()->get_rho().
						  tilde_to_nominal(rho_tilde_mle[0]-dr))
    + log(2.0) + (rho_tilde_mle[0]-dr) - 2.0*log(exp(rho_tilde_mle[0]-dr)+1);

  // std::cout << "(numerator, denom, deriv) = ("
  // 	    << ll_plus_drho - 2*ll + ll_minus_drho << ", "
  // 	    << dr << ", "
  // 	    << (ll_plus_drho - 2*ll + ll_minus_drho)/square(dr) << ")\n";

  double rho_var = -1.0*square(dr)/(ll_plus_drho - 2*ll + ll_minus_drho);

  std::vector<double> out {rho_tilde_mle[0], rho_var};
  // std::cout << "rho_tilde_mle = " << rho_tilde_mle[0] << "; rho_var = " << rho_var << "\n";
  return out;
}

std::vector<double> SVModelWithJumps::rho_tilde_xi_square_tilde_mu_MLE_mean_var()
{
  double rho_current = get_ou_model_fast()->get_rho().get_continuous_time_parameter();
  double rho_tilde_current = log( ((rho_current+1.0)/2.0) /
				  (1 - (rho_current+1.0)/2.0) );

  double xi_square_current = get_observational_model()->
    get_xi_square().get_continuous_time_parameter();
  double xi_square_tilde_current = log(xi_square_current);

  double mu_current = get_constant_vol_model()->
    get_mu().get_discrete_time_parameter(get_delta_t());

  std::vector<double> rho_tilde_mle {rho_tilde_current,
      xi_square_tilde_current,
      mu_current};

  nlopt::opt opt(nlopt::LN_NELDERMEAD, 3);
  opt.set_ftol_abs(1e-4);
  std::vector<double> lb {-1.0*HUGE_VAL, -1.0*HUGE_VAL, -1.0*HUGE_VAL};
  std::vector<double> ub {HUGE_VAL, HUGE_VAL, HUGE_VAL};
  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);

  double minf;
  opt.set_min_objective(SVModelWithJumps::wrapper_rho_xi_square_mu, this);
  opt.optimize(rho_tilde_mle, minf);


  // double dr = std::abs(rho_tilde_current/100);
  //  double dxi = std::abs(xi_square_tilde_current/100);
  double dmu = std::abs( mu_current/100 );

  double dr = 0.01;
  double dxi = 0.1;
  // double dmu = 0.01;

  std::cout << "dr = " << dr << "; dxi = " << dxi << "; dmu = " << dmu << "\n";

  double ll =
    log_likelihood_ous_integrated_filtered_prices(get_ou_model_fast()->get_rho().
						  tilde_to_nominal(rho_tilde_mle[0]),
						  exp(rho_tilde_mle[1]),
						  rho_tilde_mle[2])
    + log(2.0) + rho_tilde_mle[0] - 2.0*log(exp(rho_tilde_mle[0])+1)
    + (rho_tilde_mle[1]);

  std::cout << "minf = " << minf << "; ll = " << ll
	    << "; ll_current = "
	    << log_likelihood_ous_integrated_filtered_prices(rho_current,
							     xi_square_current,
							     mu_current)
	    << "\n";

  double ll_plus_drho =
    log_likelihood_ous_integrated_filtered_prices(get_ou_model_fast()->get_rho().
						  tilde_to_nominal(rho_tilde_mle[0]+dr),
						  exp(rho_tilde_mle[1]),
						  rho_tilde_mle[2])
    + log(2.0) + rho_tilde_mle[0] + dr - 2.0*log(exp(rho_tilde_mle[0]+dr)+1)
    + (rho_tilde_mle[1]);

  double ll_minus_drho =
    log_likelihood_ous_integrated_filtered_prices(get_ou_model_fast()->get_rho().
						  tilde_to_nominal(rho_tilde_mle[0]-dr),
						  exp(rho_tilde_mle[1]),
						  rho_tilde_mle[2])
    + log(2.0) + rho_tilde_mle[0] - dr - 2.0*log(exp(rho_tilde_mle[0]-dr)+1)
    + (rho_tilde_mle[1]);

  double ll_plus_dxi =
    log_likelihood_ous_integrated_filtered_prices(get_ou_model_fast()->get_rho().
						  tilde_to_nominal(rho_tilde_mle[0]),
						  exp(rho_tilde_mle[1]+dxi),
						  rho_tilde_mle[2])
    + log(2.0) + rho_tilde_mle[0] - 2.0*log(exp(rho_tilde_mle[0])+1)
    + (rho_tilde_mle[1] + dxi);

  double ll_minus_dxi =
    log_likelihood_ous_integrated_filtered_prices(get_ou_model_fast()->get_rho().
						  tilde_to_nominal(rho_tilde_mle[0]),
						  exp(rho_tilde_mle[1] - dxi),
						  rho_tilde_mle[2])
    + log(2.0) + rho_tilde_mle[0] - 2.0*log(exp(rho_tilde_mle[0])+1)
    + (rho_tilde_mle[1] - dxi);

  double ll_plus_dmu =
    log_likelihood_ous_integrated_filtered_prices(get_ou_model_fast()->get_rho().
						  tilde_to_nominal(rho_tilde_mle[0]),
						  exp(rho_tilde_mle[1]),
						  rho_tilde_mle[2]+dmu)
    + log(2.0) + rho_tilde_mle[0] - 2.0*log(exp(rho_tilde_mle[0])+1)
    + (rho_tilde_mle[1]);

  double ll_minus_dmu =
    log_likelihood_ous_integrated_filtered_prices(get_ou_model_fast()->get_rho().
						  tilde_to_nominal(rho_tilde_mle[0]),
						  exp(rho_tilde_mle[1]),
						  rho_tilde_mle[2] - dmu)
    + log(2.0) + rho_tilde_mle[0] - 2.0*log(exp(rho_tilde_mle[0])+1)
    + (rho_tilde_mle[1]);

  // std::cout << "(numerator, denom, deriv) = ("
  // 	    << ll_plus_drho - 2*ll + ll_minus_drho << ", "
  // 	    << dr << ", "
  // 	    << (ll_plus_drho - 2*ll + ll_minus_drho)/square(dr) << ")\n";
  // std::cout << "(numerator, denom, deriv) = ("
  // 	    << ll_plus_dxi << " " <<  ll << " " <<  ll_minus_dxi << ", "
  // 	    << dxi << ", "
  // 	    << (ll_plus_dxi - 2*ll + ll_minus_dxi)/square(dxi) << ")\n";
  // std::cout << "(numerator, denom, deriv) = ("
  // 	    << ll_plus_dmu << " " <<  ll << " " <<  ll_minus_dmu << ", "
  // 	    << dmu << ", "
  // 	    << (ll_plus_dmu - 2*ll + ll_minus_dmu)/square(dmu) << ")\n";

  double rho_var = -1.0*square(dr)/(ll_plus_drho - 2*ll + ll_minus_drho);
  if (rho_var <=0 ) {
    rho_var = 1.0;
  }
  double xi_square_var = -1.0*square(dxi)/(ll_plus_dxi - 2*ll + ll_minus_dxi);
  if (xi_square_var <= 0) {
    xi_square_var = 1.0;
  }

  double mu_var = -1.0*square(dmu)/(ll_plus_dmu - 2*ll + ll_minus_dmu);
  if (mu_var <= 0) {
    mu_var = 0.1;
  }

  std::vector<double> out {rho_tilde_mle[0], rho_tilde_mle[1], rho_tilde_mle[2],
      10*rho_var, 10*xi_square_var, 10*mu_var};

  // std::cout << "rho_mle = "
  // 	    << get_ou_model_fast()->get_rho().tilde_to_nominal(rho_tilde_mle[0])
  // 	    << "; rho_var = "
  // 	    << rho_var << "\n";
  // std::cout << "xi2_mle = "
  // 	    << get_observational_model()->get_xi_square().tilde_to_nominal(rho_tilde_mle[1])
  // 	    << "; xi2_var = "
  // 	    << xi_square_var << "\n";
  return out;
}

double SVModelWithJumps::MLE_min_rho_integrated_prices(const std::vector<double> &x,
						       std::vector<double> &grad)
{
  double rho_tilde = x[0];
  double rho = 2*(exp(rho_tilde) / (exp(rho_tilde)+1)) - 1;

  if (!grad.empty()) {
  }

  double ll = -1.0* (log_likelihood_ous_integrated_filtered_prices(rho)
		     + log(2) + rho_tilde
 		     - 2*log(exp(rho_tilde)+1));

  // std::cout << "rho = " << rho << "; ll = " << ll << "\n";
  return ll;
}

double SVModelWithJumps::MLE_min_rho_xi_square_mu_integrated_prices(const std::vector<double> &x,
								   std::vector<double> &grad)
{
  double rho_tilde = x[0];
  double rho = 2*(exp(rho_tilde) / (exp(rho_tilde)+1)) - 1;
  double xi_square_tilde = x[1];
  double xi_square = exp(xi_square_tilde);
  double mu = x[2];

  if (!grad.empty()) {}

  double ll = -1.0* (log_likelihood_ous_integrated_filtered_prices(rho,
								   xi_square,
								   mu)
		     + log(2) + rho_tilde - 2*log(exp(rho_tilde)+1)
		     + xi_square_tilde);

  // std::cout << "rho = " << rho << "; ll = " << ll << "\n";
  return ll;
}

double SVModelWithJumps::wrapper_rho(const std::vector<double> &x,
				     std::vector<double> &grad,
				     void * data)
{
  SVModelWithJumps * params =
    reinterpret_cast<SVModelWithJumps*>(data);
  return  params->MLE_min_rho_integrated_prices(x,grad);
}

double SVModelWithJumps::wrapper_rho_xi_square_mu(const std::vector<double> &x,
						  std::vector<double> &grad,
						  void * data)
{
  SVModelWithJumps * params =
    reinterpret_cast<SVModelWithJumps*>(data);
  return  params->MLE_min_rho_xi_square_mu_integrated_prices(x,grad);
}

// GENERATES DATA ON LOG SCALE
void SVModelWithJumps
::generate_data(double time,
		gsl_rng * rng,
		long int dt_record,
		std::string save_location)
{
  long int N = floor(time / get_delta_t());
  long int dt = get_delta_t();
  std::cout << dt << std::endl;
  std::cout << N << std::endl;

  double tau_square_slow = get_ou_model_slow()->get_tau_square().
    get_discrete_time_parameter(get_delta_t());
  double tau_square_fast = get_ou_model_fast()->get_tau_square().
    get_discrete_time_parameter(get_delta_t());
  double alpha = get_ou_model_fast()->get_alpha().
    get_discrete_time_parameter(get_delta_t());
  double alpha_hat = get_ou_model_fast()->get_alpha().
    get_continuous_time_parameter();
  double rho = get_ou_model_fast()->get_rho().
    get_discrete_time_parameter(get_delta_t());
  double mu = get_constant_vol_model()->get_mu().
    get_discrete_time_parameter(get_delta_t());
  double noise_size = get_observational_model()->get_xi_square().get_continuous_time_parameter();

  std::cout << "mu = " << mu << "\n";
  std::cout << "xi_square = " << noise_size << "\n";

  double lambda = get_constant_vol_model()->get_jump_rate().get_lambda();
  double probability_of_jump = 1.0 - exp(-1.0*lambda*get_delta_t());

  double jump_probabilities[2];
  jump_probabilities[0] = probability_of_jump;
  jump_probabilities[1] = 1.0-probability_of_jump;
  double *P;
  P = jump_probabilities;
  gsl_ran_discrete_t * g =
    gsl_ran_discrete_preproc(2, P);

  double microstructure_probabilities[3];
  microstructure_probabilities[0] = 0.33;
  microstructure_probabilities[1] = 0.33;
  microstructure_probabilities[2] = 0.33;
  double *PP;
  PP = microstructure_probabilities;
  gsl_ran_discrete_t * gg =
    gsl_ran_discrete_preproc(3, PP);
  // double noise_size = 0.0005;

  double jump_size_mean = get_constant_vol_model()->
    get_jump_size_mean().get_continuous_time_parameter();
  double jump_size_var = get_constant_vol_model()->
    get_jump_size_variance().get_sigma_square();
  std::cout << "jump_size_var=" << jump_size_var << "\n";

  double theta_slow = get_ou_model_slow()->get_theta().
    get_discrete_time_parameter(get_delta_t());
  double theta_fast = get_ou_model_fast()->get_theta().
    get_discrete_time_parameter(get_delta_t());

  double current_log_sigma_hat_slow = alpha_hat;
  double current_log_sigma_hat_fast = alpha_hat;
  double current_price = log(100.0);
  double current_price_clean = current_price;

  arma::vec mean = arma::zeros<arma::vec> (2);
  arma::mat cov = arma::ones<arma::mat> (2,2);
  cov(0,1) = rho;
  cov(1,0) = rho;

  std::cout << "cov=" << cov << "\n";

  // RECORD THE RESULTS
  std::ofstream simulation (save_location +
			    "simulated-prices-and-returns-bid-ask-noise-RAW.csv");
  std::ofstream data_gen_params (save_location +
				 "data-generating-parameters.csv");
  std::ofstream simulation_sparse (save_location +
				   "simulated-prices-and-returns-bid-ask-noise-SPARSE.csv");
  std::ofstream noise_record (save_location +
			      "noise-bid-ask-noise.csv");

  // header
  simulation_sparse << "price.true, price, log.sigma.hat.slow, log.sigma.hat.fast, jump\n";

  data_gen_params << "tau_square_hat_slow" << ","
		  << "tau_square_hat_fast" << ","
		  << "theta_hat_slow" << ","
		  << "theta_hat_fast" << ","
		  << "alpha_hat" << ","
		  << "rho" << ","
		  << "mu_hat" << ","
		  << "xi_square" << "\n";
  
  data_gen_params << get_ou_model_slow()->get_tau_square().get_continuous_time_parameter() << ","
		  << get_ou_model_fast()->get_tau_square().get_continuous_time_parameter() << ","
		  << get_ou_model_slow()->get_theta().get_continuous_time_parameter() << ","
		  << get_ou_model_fast()->get_theta().get_continuous_time_parameter() << ","
		  << get_ou_model_fast()->get_alpha().get_continuous_time_parameter() << ","
		  << get_ou_model_fast()->get_rho().get_discrete_time_parameter(get_delta_t()) << ","
		  << get_constant_vol_model()->get_mu().get_continuous_time_parameter() << ","
		  << noise_size << "\n";
    
  simulation << current_price << ","
	     << current_price << "," << current_log_sigma_hat_slow
	     << "," << current_log_sigma_hat_fast
	     << "," << false << "\n";
  
  simulation_sparse << current_price << ","
		    << current_price << "," << current_log_sigma_hat_slow
		    << "," << current_log_sigma_hat_fast
		    << "," << false << "\n";

  for (long int i=1; i<N+1; ++i) {
    double nu_slow = gsl_ran_gaussian(rng, 1.0);
    arma::vec epsilon_nu_fast = rmvnorm(rng,2,mean,cov);
    double epsilon = epsilon_nu_fast(0);
    double nu_fast = epsilon_nu_fast(1);

    double sigma_hat_slow =
      exp(current_log_sigma_hat_slow);
    double sigma_hat_fast =
      exp(current_log_sigma_hat_fast);

    int jump_indicator = gsl_ran_discrete (rng, g);
    double jump = 0;
    if (jump_indicator==0) {
      std::cout << "Jump occurred\n";
      jump = jump_size_mean + sqrt(jump_size_var)*gsl_ran_gaussian(rng, 1.0);
      std::cout << "jump=" << jump << "\n";
    }
    // TODO: jumps turned off
    jump_indicator = 1;
    jump = 0.0;

    int noise_indicator = gsl_ran_discrete (rng, gg);
    double noise = 0;
    if (noise_indicator == 0) {
      noise = 1.0*noise_size;
    }
    if (noise_indicator == 1) {
      noise = 0;
    }
    if (noise_indicator == 2) {
      noise = -1.0*noise_size;
    }
    noise = sqrt(noise_size)*gsl_ran_gaussian(rng, 1.0);
    current_price_clean = (current_price_clean + mu +
			   sqrt(get_delta_t())*
			   sqrt(sigma_hat_fast)*
			   sqrt(sigma_hat_slow)*
			   epsilon +
			   jump);

    current_price = (current_price_clean + noise);

    double log_sigma_slow = current_log_sigma_hat_slow + 0.5*log(get_delta_t());
    double log_sigma_fast = current_log_sigma_hat_fast + 0.5*log(get_delta_t());

    log_sigma_slow =
      alpha + theta_slow*(log_sigma_slow - alpha) + sqrt(tau_square_slow)*nu_slow;

    log_sigma_fast =
      alpha + theta_fast*(log_sigma_fast - alpha) + sqrt(tau_square_fast)*nu_fast;

    current_log_sigma_hat_slow = log_sigma_slow - 0.5*log(get_delta_t());
    current_log_sigma_hat_fast = log_sigma_fast - 0.5*log(get_delta_t());

    simulation << current_price_clean << ","
	       << current_price << "," << current_log_sigma_hat_slow
	       << "," << current_log_sigma_hat_fast
	       << "," << (1-jump_indicator)  << "\n";

    // std::cout << current_log_price << " "
    // 	      << current_log_sigma_hat_slow << " "
    // 	      << current_log_sigma_hat_fast << std::endl;
    if (i*dt % dt_record == 0) {
      simulation_sparse
	<< current_price_clean << ","
	<< current_price << "," << current_log_sigma_hat_slow
	<< "," << current_log_sigma_hat_fast
	<< "," << (1-jump_indicator)  << "\n";

      noise_record << noise << "\n";
    }

    if ( i % 1000000 == 0 ) {
      std::cout << i << std::endl;
    }
  }
  simulation.close();
  simulation_sparse.close();
  noise_record.close();
  gsl_ran_discrete_free(g);
  gsl_ran_discrete_free(gg);
}
