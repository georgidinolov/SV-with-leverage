// ===================== CONSTANT VOL PARAMS ==============================
ConstantVolatilityParams::ConstantVolatilityParams()
  : mu_(MuParameter(1.0)),
    gammas_(GammaParameter()),
{}

ConstantVolatilityParams::ConstantVolatilityParams(double delta_t)
  : mu_(MuParameter(delta_t)),
    gammas_(GammaParameter()),
{}

const MuParameter& ConstantVolatilityParams::get_mu() const
{
  return mu_;
}

const GammaParameter& ConstantVolatilityParams::get_gammas() const
{
  return gammas_;
}

void ConstantVolatilityParams::set_gammas(const std::vector<int>& gammas)
{
  gammas_.set_gammas(gammas);
}

void ConstantVolatilityParams::set_gammas(const GammaParameter& gammas)
{
  gammas_.set_gammas(gammas.get_gammas());
}
