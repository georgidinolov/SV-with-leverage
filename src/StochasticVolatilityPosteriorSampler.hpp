#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <random>
#include "StochasticVolatilityModel.hpp"
#include "ProposalTypes.hpp"

// ======================== OBSERVATIONAL MODEL SAMPLER ============
class ObservationalPosteriorSampler
{
public:
  ObservationalPosteriorSampler(ObservationalModel * obs_mod,
				gsl_rng * rng);
  void draw_xi_square();
  void draw_deltas();
  void draw_nu();
  inline ObservationalModel* get_observational_model() {
    return observational_model_;
  }

private:
  ObservationalModel * observational_model_;
  gsl_rng * rng_;
};

// ================= CONST VOL SAMPELR ============================
class ConstantVolatilityPosteriorSampler
{
public:
  ConstantVolatilityPosteriorSampler(ConstantVolatilityModel * const_vol_mod,
				     gsl_rng * rng);
  void draw_mu_hat();

private:
  ConstantVolatilityModel * const_vol_model_;
  gsl_rng * rng_;
};

// ================= CONST VOL SAMPELR WITH JUMPS  ==============
class ConstantVolatilityWithJumpsPosteriorSampler
{
public:
  ConstantVolatilityWithJumpsPosteriorSampler(
            ConstantMultifactorVolatilityModelWithJumps * const_vol_mod,
	    gsl_rng * rng);

  inline ConstantMultifactorVolatilityModelWithJumps * 
  get_constant_vol_model()
  {
    return const_vol_model_;
  }

  void draw_mu_hat();
  void draw_jump_size_mean();
  void draw_jump_size_variance();
  void draw_jump_rate();
  void draw_jump_indicators();

private:
  ConstantMultifactorVolatilityModelWithJumps * const_vol_model_;
  gsl_rng * rng_;
};

// ======================== OU SAMPLER =============================
class OUPosteriorSampler
{
public:
  OUPosteriorSampler(OUModel * ou_model,
		     gsl_rng * rng,
		     const gsl_matrix * proposal_covariance_matrix);
  virtual ~OUPosteriorSampler();

  void draw_alpha_hat();

  void draw_rho_tnorm();
  void draw_rho_norm();
  void draw_rho_norm_walk();

  void draw_theta_hat();
  
  void draw_tau_square_hat();

  void draw_theta_tau_square_rho();
  void draw_theta_tau_square();
  
  void set_theta_lower_bound(double lb);
  void set_theta_upper_bound(double ub);

  inline OUModel* get_ou_model() {
    return ou_model_;
  }
  
private:
  OUModel * ou_model_;
  gsl_rng * rng_;
  ThetaTauSquareRhoProposal theta_tau_square_rho_proposal_;
  double acceptance_ratio_;
  double number_iterations_;
};

// ======================= FAST OU SAMPLER =========================
// Doesn't sample alpha; that's the job of the sampler holding this
// sampler
class FastOUPosteriorSampler
{
public:
  FastOUPosteriorSampler(FastOUModel * ou_model_fast,
			 gsl_rng * rng,
			 const gsl_matrix * proposal_covariance_matrix);

  void draw_theta_hat();
  void draw_theta_hat_norm();

  void draw_rho_norm();
  void draw_rho_tnorm();
  void draw_rho_student();

  void draw_tau_square_hat();
  void draw_tau_square_hat_norm_MLE();

  void draw_theta_tau_square_rho();
  // everything is done on the tilde scale
  void draw_theta_tau_square_rho_MLE();
  
  void set_theta_lower_bound(double lb);
  void set_theta_upper_bound(double ub);

  inline FastOUModel* get_ou_model_fast() {
    return ou_model_fast_;
  }

  inline const ThetaTauSquareRhoProposal& get_proposal() const
  {
    return theta_tau_square_rho_proposal_;
  }

private:
  FastOUModel * ou_model_fast_;
  gsl_rng * rng_;
  ThetaTauSquareRhoProposal theta_tau_square_rho_proposal_;
};

// ======================= SV SAMPLER ==============================
class StochasticVolatilityPosteriorSampler
{
public:
  StochasticVolatilityPosteriorSampler(StochasticVolatilityModel *sv_model,
				       gsl_rng * rng,
				       const gsl_matrix * proposal_covariance_matrix);

  inline void draw() {
    ou_sampler_.draw_alpha_hat();
    ou_sampler_.draw_theta_hat();
    ou_sampler_.draw_tau_square_hat();
    ou_sampler_.draw_rho_norm();

    // ou_sampler_.draw_theta_tau_square_rho();

    draw_gammas_gsl();
    constant_vol_sampler_.draw_mu_hat();
    draw_sigmas();
    // draw_filtered_log_prices();
    observational_model_sampler_.draw_xi_square();
  }
  void draw_gammas();
  void draw_gammas_gsl();
  void draw_sigmas();
  void draw_filtered_log_prices();

private:
  StochasticVolatilityModel * sv_model_;
  gsl_rng * rng_;

  ObservationalPosteriorSampler observational_model_sampler_;
  ConstantVolatilityPosteriorSampler constant_vol_sampler_;
  OUPosteriorSampler ou_sampler_;
};

// ====================== SV MULTIFACTOR SAMPLER ==================
class MultifactorStochasticVolatilityPosteriorSampler
{
public:
  MultifactorStochasticVolatilityPosteriorSampler(MultifactorStochasticVolatilityModel *model,
						  gsl_rng * rng,
						  const gsl_matrix * proposal_covariance_matrix);
  virtual ~MultifactorStochasticVolatilityPosteriorSampler();

  inline void draw() {
    draw_alpha_hat();

    // Drawing theta_SLOW
    ou_sampler_slow_.draw_theta_hat();
    ou_sampler_slow_.draw_tau_square_hat();

    // Drawing theta_hat_FAST
    ou_sampler_fast_.draw_theta_hat();
    ou_sampler_fast_.draw_tau_square_hat();
    ou_sampler_fast_.draw_rho_student();

    draw_gammas_gsl();
    constant_vol_sampler_.draw_mu_hat();
    draw_sigmas();
    // draw_filtered_log_prices();
    // observational_model_sampler_.draw_xi_square();
  }
  void draw_gammas();
  void draw_gammas_gsl();
  void draw_alpha_hat();
  void draw_sigmas();
  void draw_filtered_log_prices();
  
  inline const ObservationalPosteriorSampler& get_observational_sampler() const {
    return observational_model_sampler_;
  }
  inline const ConstantVolatilityPosteriorSampler& get_const_vol_sampler() const {
    return constant_vol_sampler_;
  }
  inline const OUPosteriorSampler& get_ou_sampler_slow() const {
    return ou_sampler_slow_;
  }
  inline const FastOUPosteriorSampler& get_ou_sampler_fast() const {
    return ou_sampler_fast_;
  }

private:
  MultifactorStochasticVolatilityModel * sv_model_;
  gsl_rng * rng_;

  ObservationalPosteriorSampler observational_model_sampler_;
  ConstantVolatilityPosteriorSampler constant_vol_sampler_;
  OUPosteriorSampler ou_sampler_slow_;
  FastOUPosteriorSampler ou_sampler_fast_;
};

// ================ SV MULTIFACTOR SAMPLER WITH JUMPS ===============
class SVWithJumpsPosteriorSampler
{
public:
  SVWithJumpsPosteriorSampler(SVModelWithJumps *model,
			      gsl_rng * rng,
			      const gsl_matrix * proposal_covariance_matrix,
			      const gsl_matrix * proposal_covariance_matrix_all);
  virtual ~SVWithJumpsPosteriorSampler();

  inline void draw() {
    draw_rho_xi_mu_integrated_prices();
    draw_filtered_log_prices();

    // draw_nu_integrated_deltas();
    // observational_model_sampler_.draw_deltas();

    // constant_vol_sampler_.draw_lambda_jump_size_jump_variance_integraged_jumps();
    // constant_vol_sampler_.draw_jumps();

    // constant_vol_sampler_.draw_jump_indicators();
    // constant_vol_sampler_.draw_jump_size_mean();
    // constant_vol_sampler_.draw_jump_size_variance();
    // constant_vol_sampler_.draw_jump_rate();

    draw_gammas_gsl();
    // draw_sv_models_minus_rho_params_integrated_vol();
    draw_sigmas();
  }

  void draw_gammas_gsl();
  void draw_alpha_hat();

  // Draws rho, theta, tau2 for slow and fast models, where we
  // integrate out h_t, conditioning on log(S_t).
  void draw_sv_models_params_integrated_vol();
  // Draws theta, tau2 for slow and fast models, where we
  // integrate out h_t, conditioning on log(S_t).
  void draw_sv_models_minus_rho_params_integrated_vol();
  // Draws rho, theta, tau2 for slow and fast models, where we
  // integrate out log(S_t), conditioning on h_t.
  void draw_sv_models_params_integrated_prices();
  // Draws rho only, where we integrate out log(S_t) and \gamma_t
  void draw_rho_integrated_prices(double rho_tilde_sd);
  // Draws rho only, where we integrate out log(S_t) and \gamma_t,
  // proposing centered on the MLE and with variance equal to the
  // neg. second deriv at the MLE
  void draw_rho_integrated_prices_MLE();
  // Draws \rho, \xi, \mu where we integrate out log(S_t) and \gamma_t
  void draw_rho_xi_mu_integrated_prices();
  // Draws \rho, \xi, \mu where we integrate out log(S_t) and
  // \gamma_t. Proposal for rho is centered on the MLE for rho.
  void draw_rho_xi_mu_integrated_prices_MLE();
  // Draws \xi, \mu where we integrate out log(S_t) and \gamma_t
  void draw_xi_mu_integrated_prices();
  // Draws nu by integrating out the deltas
  void draw_nu_integrated_deltas();

  void draw_sv_fast_params();
  void draw_sigmas();
  void draw_filtered_log_prices();
  void draw_xi_square(double log_xi_square_prop_sd);

  inline void draw_rho() {
    ou_sampler_fast_.draw_rho_student();
  }
  inline void draw_theta_hat_slow() {
    ou_sampler_slow_.draw_theta_hat();
  }
  inline std::vector<double> rho_mle() {
    std::vector<double> out = ou_sampler_fast_.get_ou_model_fast()->rho_MLE_mean_var();
    return out;
  }
private:
  SVModelWithJumps * sv_model_;
  gsl_rng * rng_;

  ObservationalPosteriorSampler observational_model_sampler_;
  ConstantVolatilityWithJumpsPosteriorSampler constant_vol_sampler_;
  OUPosteriorSampler ou_sampler_slow_;
  FastOUPosteriorSampler ou_sampler_fast_;
  
  NormalRWProposal normal_rw_proposal_;
  NormalRWProposal normal_rw_proposal_obs_model_params_;
};
