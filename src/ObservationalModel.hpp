#include "DataTypes.hpp"
#include "ParamTypes.hpp"
#include "PriorTypes.hpp"
#include "StochasticVolatilityModel.hpp"

class ObservationalModel
  : BaseModel,
    ObservationalParams,
    ObservationalPriors,
    OpenCloseData
{
public:
  ObservationalModel();
  ObservationalModel(double delta_t);

private:
  
}
