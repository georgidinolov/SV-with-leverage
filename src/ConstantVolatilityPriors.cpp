// =================== CONSTANT VOLATILITY PRIORS ===============

ConstantVolatilityPriors::ConstantVolatilityPriors()
  : mu_prior_(MuPrior())
{}

ConstantVolatilityPriors::ConstantVolatilityPriors(double delta_t)
  : mu_prior_(MuPrior(delta_t))
{}

const MuPrior & ConstantVolatilityPriors::get_mu_prior() const
{
  return mu_prior_;
}
