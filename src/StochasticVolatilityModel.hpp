#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <vector>
#include "DataTypes.hpp"
#include "ParamTypes.hpp"
#include "PriorTypes.hpp"
#include "src/nlopt/api/nlopt.hpp"

class OUModel;
class FastOUModel;
class ConstantVolatilityModel;

// ================== BASE MODEL ====================
class BaseModel
{
public:
  BaseModel();
  BaseModel(double delta_t);
  virtual ~BaseModel();
  double get_delta_t() const;
  bool const_delta_t() const;
  virtual double log_likelihood()const =0;
  virtual unsigned data_length()const =0;

private:
  double delta_t_;
  bool const_delta_t_;
};

// =================== OBSERVATIONAL MODEL ===============
class ObservationalModel
  : public BaseModel,
    public ObservationalParams,
    public ObservationalPriors,
    public OpenCloseData
{
public:
  ObservationalModel(const OpenCloseData& data,
		     double delta_t);

  inline const ConstantVolatilityModel* get_const_vol_model() const
  {
    return constant_vol_model_;
  }
  const std::vector<double>& get_filtered_log_prices() const;
  const std::vector<double>& get_deltas() const;
  double get_nu() const;

  inline void set_const_vol_model(const ConstantVolatilityModel* const_vol_mod)
  {
    constant_vol_model_ = const_vol_mod;
  }
  void set_nu(double nu);
  void set_delta_element(unsigned i, double delta);

  virtual unsigned data_length() const;
  virtual double log_likelihood() const;
  virtual double log_likelihood_integrated_filtered_prices(
	   double xi_square,
	   double alpha,
	   double theta_fast,
	   double tau_square_fast,
	   double rho,
	   const std::vector<SigmaSingletonParameter>& sigmas_slow,
	   const std::vector<SigmaSingletonParameter>& sigmas_fast,
	   const std::vector<double>& h_fast,
	   const std::vector<double>& jump_sizes,
	   const std::vector<double>& deltas) const;

private:
  const ConstantVolatilityModel* constant_vol_model_;
  
  std::vector<double> deltas_;
  DeltaPrior delta_prior_;
  double nu_;
};

// =================== CONSTANT VOL MODEL ==============
class ConstantVolatilityModel
  : public BaseModel,
    public ConstantVolatilityParams,
    public ConstantVolatilityPriors
{
public:
  ConstantVolatilityModel(const ObservationalModel* observational_model,
			  double delta_t);

  inline const ObservationalModel * get_observational_model() const {
    return observational_model_;
  }
  inline const OUModel * get_ou_model() const {
    return ou_model_;
  }
  const std::vector<double>& get_y_star() const;
  inline const std::vector<double>& get_filtered_log_prices() const {
    return filtered_log_prices_;
  }
  const std::vector<int>& get_ds() const;
  const std::vector<double>& get_as() const;
  const std::vector<double>& get_bs() const;

  void set_observational_model(const ObservationalModel* obs_model) {
    observational_model_ = obs_model;
  }
  void set_ou_model(const OUModel * ou_model) {
    ou_model_ = ou_model;
  }
  virtual unsigned data_length() const;
  virtual void set_y_star_ds();
  virtual void set_filtered_log_prices(const std::vector<double>& flps);

  inline void set_y_star(const std::vector<double>& y_star_new) {
    y_star_ = y_star_new;
  }
  inline void set_ds(const std::vector<int>& ds_new) {
    ds_ = ds_new;
  }
  virtual double log_likelihood() const;
  
private:
  const ObservationalModel * observational_model_;
  const OUModel * ou_model_;
  std::vector<double> y_star_;
  std::vector<int> ds_;
  std::vector<double> filtered_log_prices_;

  std::vector<double> as_correction_;
  std::vector<double> bs_correction_;
};

// =================== CONSTANT MULTIFACTOR VOL MODEL =================
class ConstantMultifactorVolatilityModel
  : public ConstantVolatilityModel
{
public:
  ConstantMultifactorVolatilityModel(const ObservationalModel* observational_model,
				     double delta_t);

  inline const FastOUModel* get_ou_model_fast() const {
    return ou_model_fast_;
  }
  inline const OUModel* get_ou_model_slow() const {
    return get_ou_model();
  }
  inline void set_ou_model_fast(const FastOUModel* ou_mod_fast) {
    ou_model_fast_ = ou_mod_fast;
  }
  inline void set_ou_model_slow(const OUModel* ou_mod_slow) {
    set_ou_model(ou_mod_slow);
  }
  
  virtual double log_likelihood() const;
private:
  const FastOUModel * ou_model_fast_;
};

// =================== CONSTANT MULTIFACTOR VOL MODEL WITH JUMPS  =================
class ConstantMultifactorVolatilityModelWithJumps
  : public ConstantMultifactorVolatilityModel
{
public:
  ConstantMultifactorVolatilityModelWithJumps(const ObservationalModel* observational_model,
					      double delta_t);
  virtual void set_y_star_ds();

  inline long unsigned get_number_jumps() const {
    return number_jumps_;
  }
  inline const std::vector<bool>& get_jump_indicators() const {
    return jump_indicators_;
  }
  inline const std::vector<double>& get_jump_sizes() const {
    return jump_sizes_;
  }

  inline void set_number_jumps(long unsigned number_jumps) {
    number_jumps_ = number_jumps;
  }
  inline void set_jump_indicator(const std::vector<bool>& jump_indicators) {
    jump_indicators_ = jump_indicators;
  }
  inline void set_jump_indicator(unsigned i, bool JUMP) {
    jump_indicators_[i] = JUMP;
  }
  inline void set_jump_sizes(const std::vector<double>& jump_sizes) {
    jump_sizes_ = jump_sizes;
  }
  inline void set_jump_size(unsigned i, double jump_size) {
    jump_sizes_[i] = jump_size;
  }

  inline const MuPrior& get_jump_size_prior() const {
    return jump_size_prior_;
  }
  inline const SigmaSquarePrior& get_jump_size_variance_prior() const {
    return jump_size_variance_prior_;
  }
  inline const LambdaPrior& get_jump_rate_prior() const {
    return jump_rate_prior_;
  }

  inline const MuParameter& get_jump_size_mean() const {
    return jump_size_mean_;
  }
  inline const SigmaSquareParam& get_jump_size_variance() const {
    return jump_size_variance_;
  }
  inline const LambdaParam& get_jump_rate() const {
    return jump_rate_;
  }

  inline void set_jump_size_mean(double jump_size_mean) {
    jump_size_mean_.set_continuous_time_parameter(jump_size_mean);
  }
  inline void set_jump_size_variance(double jump_size_var) {
    jump_size_variance_.set_sigma_square(jump_size_var);
  }
  inline void set_jump_rate(double lambda) {
    jump_rate_.set_lambda(lambda);
  }

private:
  long unsigned number_jumps_;
  std::vector<bool> jump_indicators_;
  std::vector<double> jump_sizes_;

  MuPrior jump_size_prior_;
  SigmaSquarePrior jump_size_variance_prior_;
  LambdaPrior jump_rate_prior_;

  MuParameter jump_size_mean_;
  SigmaSquareParam jump_size_variance_;
  LambdaParam jump_rate_;
};

// ================== OU MODEL ======================
class OUModel
  : public BaseModel
{
public:
  OUModel(const ConstantVolatilityModel* const_vol_model);
  OUModel(const ConstantVolatilityModel* const_vol_model,
	  double delta_t);

  const ConstantVolatilityModel* get_const_vol_model() const;
  const ThetaParameter& get_theta() const;
  const TauSquareParameter& get_tau_square() const;
  const AlphaParameter& get_alpha() const;
  const RhoParameter& get_rho() const;
  const SigmaParameter& get_sigmas() const; 
  unsigned data_length() const;

  ThetaPrior& get_theta_prior();
  const TauSquarePrior& get_tau_square_prior() const;
  const AlphaPrior& get_alpha_prior() const;
  const RhoPrior& get_rho_prior() const;

  void set_const_vol_model(const ConstantVolatilityModel* const_vol_model);
  void set_sigmas(const SigmaParameter& sigmas);
  void set_sigmas_element(unsigned i, double sigma_hat, double log_sigma);
  void set_tau_square_hat(double tau_square_hat);
  void set_theta_hat(double theta_hat);
  void set_alpha_hat(double alpha_hat);
  void set_rho(double rho);

  inline void set_theta_hat_mean(double theta_hat_mean) {
    theta_prior_.set_theta_hat_mean(theta_hat_mean);
  }
  inline void set_theta_hat_std_dev(double std) {
    theta_prior_.set_theta_hat_std_dev(std);
  }

  inline void set_tau_square_hat_mean(double tau_square_hat_mean) {
    tau_square_prior_.set_tau_square_hat_mean(tau_square_hat_mean);
  }
  inline void set_tau_square_hat_std_dev(double std) {
    tau_square_prior_.set_tau_square_hat_std_dev(std);
  }

  inline void set_alpha_hat_mean(double alpha_hat_mean) {
    alpha_prior_.set_alpha_hat_mean(alpha_hat_mean);
  }
  inline void set_alpha_hat_std_dev(double alpha_hat_sd) {
    alpha_prior_.set_alpha_hat_std_dev(alpha_hat_sd);
  }

  double theta_j(unsigned i_data_index,
		 unsigned j_mixture_index) const;
  
  double alpha_j(unsigned i_data_index,
		 unsigned j_mixture_index) const;

  virtual double log_likelihood() const;

  virtual std::vector<double> alpha_posterior_mean_var() const;
  virtual std::vector<double> rho_posterior_mean_var() const;
  // shape and rate for sampling tau^2 from INV GAMMA proposal
  virtual std::vector<double> tau_square_posterior_shape_rate() const;


private:
  const ConstantVolatilityModel * const_vol_model_;
  
  ThetaTauSquareParameter theta_tau_square_parameter_;
  AlphaParameter alpha_parameter_;
  RhoParameter rho_parameter_;
  SigmaParameter sigmas_parameter_;

  ThetaPrior theta_prior_;
  TauSquarePrior tau_square_prior_;
  AlphaPrior alpha_prior_;
  RhoPrior rho_prior_;
};

// =================== FAST OU MODEL =================
class FastOUModel
  : public OUModel
{
public:
  FastOUModel(const ConstantVolatilityModel* const_vol_model,
	      const OUModel* ou_model_slow);

  FastOUModel(const ConstantVolatilityModel* const_vol_model,
	      const OUModel* ou_model_slow,
	      double delta_t);
  ~FastOUModel();
  
  double theta_j_one(unsigned i_data_index,
  		     unsigned j_mixture_index) const;

  double theta_j_two(unsigned i_data_index,
  		     unsigned j_mixture_index) const;
 
  virtual double log_likelihood() const;
  virtual double log_likelihood(double rho, double theta, double tau_squared); 
  virtual double log_likelihood_tilde(double rho_tilde, 
				      double theta_tilde, 
				      double tau_squared_tilde); 
  virtual double log_likelihood_rho(double rho);
  virtual double log_likelihood_tau_square(double tau_square);

  inline const OUModel* get_ou_model_slow() const 
  {
    return ou_model_slow_;
  }

  // numerical derivatives on nominal scale
  virtual double rho_deriv_numeric_nominal_scale(double rho,
						 double theta,
						 double tau_square,
						 double drho);

  virtual double rho_deriv_analytic_nominal_scale(double rho,
  						  double theta,
  						  double tau_square);

  virtual double rho_double_deriv_numeric_nominal_scale(double rho,
							double theta,
							double tau_square,
							double drho);

  virtual double rho_theta_deriv_numeric_nominal_scale(double rho,
						       double theta,
						       double tau_square,
						       double drho,
						       double dtheta);

  virtual double rho_tau_square_deriv_numeric_nominal_scale(double rho,
							    double theta,
							    double tau_square,
							    double drho,
							    double dtau_sq);

  virtual double theta_deriv_numeric_nominal_scale(double rho,
						   double theta,
						   double tau_square,
						   double dtheta);

  virtual double theta_double_deriv_numeric_nominal_scale(double rho,
							  double theta,
							  double tau_square,
							  double dtheta);

  virtual double theta_tau_square_deriv_numeric_nominal_scale(double rho,
							      double theta,
							      double tau_square,
							      double dtheta,
							      double dtau_sq);

  virtual double tau_square_deriv_numeric_nominal_scale(double rho,
  							double theta,
  							double tau_square,
  							double dtau_sq);

  virtual double tau_square_double_deriv_numeric_nominal_scale(double rho,
  							       double theta,
  							       double tau_square,
  							       double dtau_sq);
  
  // numerical derivatives on tilde scale
  virtual double rho_deriv_numeric_tilde_scale(double rho_tilde,
					       double theta_tilde,
					       double tau_square_tilde,
					       double drho_tilde);

  virtual double rho_deriv_analytic_tilde_scale(double rho_tilde,
						double theta_tilde,
						double tau_square_tilde);

  virtual double rho_deriv_analytic_tilde_scale(double rho_tilde);
  
  virtual double rho_double_deriv_numeric_tilde_scale(double rho_tilde,
  						      double theta_tilde,
  						      double tau_square_tilde,
  						      double drho_tilde);
  
  virtual double rho_theta_deriv_numeric_tilde_scale(double rho_tilde,
  						     double theta_tilde,
  						     double tau_square_tilde,
  						     double drho_tilde,
  						     double dtheta_tilde);
  
  virtual double rho_tau_square_deriv_numeric_tilde_scale(double rho_tilde,
  							  double theta_tilde,
  							  double tau_square_tilde,
  							  double drho_tilde,
  							  double dtau_sq_tilde);
  
  virtual double theta_deriv_numeric_tilde_scale(double rho_tilde,
  						   double theta_tilde,
  						   double tau_square_tilde,
  						   double dtheta_tilde);

  virtual double theta_double_deriv_numeric_tilde_scale(double rho_tilde,
  							  double theta_tilde,
  							  double tau_square_tilde,
  							  double dtheta_tilde);

  virtual double theta_tau_square_deriv_numeric_tilde_scale(double rho_tilde,
							    double theta_tilde,
							    double tau_square_tilde,
							    double dtheta_tilde,
							    double dtau_sq_tilde);

  virtual double tau_square_deriv_numeric_tilde_scale(double rho_tilde,
						      double theta_tilde,
						      double tau_square_tilde,
						      double dtau_sq_tilde);

  virtual double tau_square_double_deriv_numeric_tilde_scale(double rho_tilde,
  							       double theta_tilde,
  							       double tau_square_tilde,
  							       double dtau_sq_tilde);
  // end of derivs

  virtual std::vector<double> alpha_posterior_mean_var() const;
  virtual std::vector<double> tau_square_posterior_shape_rate() const;
  virtual std::vector<double> tau_square_MLE_shape_rate();
  virtual std::vector<double> tau_square_MLE_mean_variance_tilde_scale();
  virtual std::vector<double> rho_posterior_mean_var() const;
  virtual std::vector<double> rho_MLE_mean_var();
  virtual std::vector<double> rho_MLE_mean_var_tilde();

  // returns parameters on tilde scale
  virtual std::vector<double> rho_theta_tau_square_tilde_MLE();

private:
  const OUModel* ou_model_slow_;
  double A_;
  double B_;
  double C_;
  double D_;

  double MLE_min_rho(const std::vector<double> &x,
		     std::vector<double> &grad);
  double MLE_min_tau_square(const std::vector<double> &x,
			    std::vector<double> &grad);
  double MLE_min_all(const std::vector<double> &x,
			    std::vector<double> &grad);

  double rho_cube_poly(double rho) const;
  double rho_cube_poly_prime(double rho) const;

  static double wrapper_rho(const std::vector<double> &x, 
			    std::vector<double> &grad,
			    void * data);

  static double wrapper_tau_square(const std::vector<double> &x, 
				   std::vector<double> &grad,
				   void * data);

  static double wrapper_all(const std::vector<double> &x, 
			    std::vector<double> &grad,
			    void * data);
};

// =================== SV MODEL =======================
class StochasticVolatilityModel 
  : public BaseModel
{
public:
  StochasticVolatilityModel(const OpenCloseData& data,
			    double delta_t);
  ~StochasticVolatilityModel();

  inline ObservationalModel* get_observational_model()
  {
    return observational_model_;
  }
  inline ConstantVolatilityModel* get_const_vol_model()
  {
    return const_vol_model_;
  }
  inline OUModel* get_ou_model() 
  {
    return ou_model_;
  }

  virtual double log_likelihood() const;
  virtual unsigned data_length() const;
  virtual void generate_data(double time_length,
			     gsl_rng * rng);
private:
  ObservationalModel * observational_model_;
  ConstantVolatilityModel * const_vol_model_;
  OUModel * ou_model_;
};

// ======================= SV MODEL MULTIFACTOR ================
class MultifactorStochasticVolatilityModel 
  : public virtual BaseModel
{
public:
  MultifactorStochasticVolatilityModel(const OpenCloseData& data,
				       double delta_t,
				       double theta_hat_fast_mean,
				       double theta_hat_fast_std_dev,
				       double theta_hat_slow_mean,
				       double theta_hat_slow_std);

  virtual ~MultifactorStochasticVolatilityModel();

  inline ObservationalModel* get_observational_model() {
    return observational_model_;
  }
  inline ConstantMultifactorVolatilityModel* get_constant_vol_model() {
    return const_multifactor_vol_model_;
  }
  inline OUModel* get_ou_model_slow() {
    return ou_model_slow_;
  }
  inline FastOUModel* get_ou_model_fast() {
    return ou_model_fast_;
  }
  // const OUModel * get_ou_model_slow();
  // const FastOUModel * get_ou_model_fast();
  
  virtual unsigned data_length() const;
  virtual double log_likelihood() const;

  virtual void generate_data(double time_length,
			     gsl_rng * rng);
private:
  ObservationalModel * observational_model_;
  ConstantMultifactorVolatilityModel * const_multifactor_vol_model_;
  
  // theta_hat_fast > that_hat_slow
  // thata_fast < theta_slow
  OUModel * ou_model_slow_;
  FastOUModel * ou_model_fast_;
};

// ============= SV MODEL MULTIFACTOR WITH JUMP =============
class SVModelWithJumps
  : public BaseModel
{
public:
  SVModelWithJumps(const OpenCloseData& data,
		   double delta_t,
		   double theta_hat_fast_mean,
		   double theta_hat_fast_std_dev,
		   double theta_hat_slow_mean,
		   double theta_hat_slow_std);

  SVModelWithJumps(const OpenCloseData& data,
		   double delta_t,
		   double theta_hat_fast_mean,
		   double theta_hat_fast_std_dev,
		   double theta_hat_slow_mean,
		   double theta_hat_slow_std,
		   double tau_square_hat_fast_mean,
		   double tau_square_hat_fast_sd,
		   double tau_square_hat_slow_mean,
		   double tau_square_hat_slow_sd);

  virtual ~SVModelWithJumps();

  inline ObservationalModel* get_observational_model() {
    return observational_model_;
  }
  inline ConstantMultifactorVolatilityModelWithJumps* get_constant_vol_model() {
    return const_multifactor_vol_model_with_jumps_;
  }
  inline OUModel* get_ou_model_slow() {
    return ou_model_slow_;
  }
  inline FastOUModel* get_ou_model_fast() {
    return ou_model_fast_;
  }
 
  virtual unsigned data_length() const;
  virtual double log_likelihood() const;
  virtual double log_likelihood_ous_integrated_vol(double alpha,
						   double rho,
						   double theta_slow,
						   double theta_fast,
						   double tau_square_slow,
						   double tau_square_fast) const;

  virtual double log_likelihood_ous_integrated_filtered_prices(
       double alpha,
       double rho,
       double theta_slow,
       double theta_fast,
       double tau_square_slow,
       double tau_square_fast,
       double xi_square,
       double mu) const;

  virtual double log_likelihood_ous_integrated_filtered_prices(double rho);
  virtual double log_likelihood_ous_integrated_filtered_prices(double rho,
							       double xi_square,
							       double mu);
  
  virtual std::vector<double> rho_tilde_MLE_mean_var();
  virtual std::vector<double> rho_tilde_xi_square_tilde_mu_MLE_mean_var();

  virtual void generate_data(double time_length,
  			     gsl_rng * rng,
			     long int dt_record,
			     std::string save_location);
private:
  ObservationalModel * observational_model_;
  ConstantMultifactorVolatilityModelWithJumps * 
  const_multifactor_vol_model_with_jumps_;
  
  // theta_hat_fast > that_hat_slow
  // thata_fast < theta_slow
  OUModel * ou_model_slow_;
  FastOUModel * ou_model_fast_;

  double MLE_min_rho_integrated_prices(const std::vector<double> &x,
				       std::vector<double> &grad);  

  double MLE_min_rho_xi_square_mu_integrated_prices(const std::vector<double> &x,
						    std::vector<double> &grad);  

  static double wrapper_rho(const std::vector<double> &x,
			    std::vector<double> &grad,
			    void * data);

  static double wrapper_rho_xi_square_mu(const std::vector<double> &x,
					 std::vector<double> &grad,
					 void * data);
};
