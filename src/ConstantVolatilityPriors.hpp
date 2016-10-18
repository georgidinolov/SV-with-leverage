// =================== CONSTANT VOLATILITY PRIORS ===============

class ConstantVolatilityPriors
{
public:
  ConstantVolatilityPriors();
  ConstantVolatilityPriors(double delta_t);

  // ConstantVolatilityPriors(double mu_hat_mean,
  // 			     double mu_hat_std_dev,
  // 			     double theta_hat_mean,
  // 			     double theta_hat_std_dev,
  // 			     double alpha_hat_mean, 
  // 			     double alpha_hat_std_dev,
  // 			     double tau_square_hat_mean,
  // 			     double tau_square_hat_std_dev,
  // 			     double xi_square_mean,
  // 			     double xi_square_std_dev,
  // 			     double rho_mean,
  // 			     double rho_std_dev);

  // ConstantVolatilityPriors(double mu_hat_mean,
  // 			     double mu_hat_std_dev,
  // 			     double theta_hat_mean,
  // 			     double theta_hat_std_dev,
  // 			     double alpha_hat_mean, 
  // 			     double alpha_hat_std_dev,
  // 			     double tau_square_hat_mean,
  // 			     double tau_square_hat_std_dev,
  // 			     double xi_square_mean,
  // 			     double xi_square_std_dev,
  // 			     double rho_mean,
  // 			     double rho_std_dev,
  // 			     double delta_t);

  const MuPrior& get_mu_prior() const;
private:
  MuPrior mu_prior_;
};
