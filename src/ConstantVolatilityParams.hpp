#include <vector>
#include "DataTypes.hpp"
#include "ParamTypes.hpp"
#include "PriorTypes.hpp"

// ===================== CONSTANT VOL PARAMS ===========================
class ConstantVolatilityParams
{
public:
  ConstantVolatilityParams();
  ConstantVolatilityParams(double delta_t);

  ConstantVolatilityParams * clone()const;
  // ConstantVolatilityParams & operator=(const ConstantVolatilityParams &rhs);

  const MuParameter& get_mu() const;
  const GammaParameter& get_gammas() const; 

  void set_gammas(const std::vector<int>& gammas);
  void set_gammas(const GammaParameter& gammas);

private:
  MuParameter mu_;
  GammaParameter gammas_;
};
