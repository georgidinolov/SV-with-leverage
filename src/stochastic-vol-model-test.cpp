#include <iostream>
#include "StochasticVolatilityModel.hpp"
#include "src/nlopt/api/nlopt.hpp"
using namespace std;

int main ()
{
  double dt = 1000.0;
  OpenCloseDatum datum = OpenCloseDatum(0,1,dt,0);
  OpenCloseDatum datum_two = OpenCloseDatum(datum.get_close(),-10,dt,
					    datum.get_t_beginning()+dt);
  OpenCloseData data = OpenCloseData();
  data.add_data(datum);
  data.add_data(datum_two);
  
  std::vector<double> log_filtered_prices 
  {data.get_data_element(0).get_open(), 
      data.get_data_element(0).get_close(),
      data.get_data_element(1).get_close()};

  // OBSERVATIONAL MODEL 
  ObservationalModel observational_model = ObservationalModel(data,
							      dt);

  std::cout << observational_model.data_length() << std::endl;

  ConstantVolatilityModel const_vol_model = ConstantVolatilityModel(&observational_model,
  								    dt);
  observational_model.set_const_vol_model(&const_vol_model);
  
  OUModel ou_model_slow = OUModel(&const_vol_model,
				  dt);
  ou_model_slow.set_const_vol_model(&const_vol_model);

  FastOUModel ou_model_fast = FastOUModel(&const_vol_model,
					  &ou_model_slow,
					  dt);
  ou_model_fast.set_const_vol_model(&const_vol_model);
  
  MultifactorStochasticVolatilityModel model = 
    MultifactorStochasticVolatilityModel(data, dt,
					 1.0/(10*60*1000),
					 1.0/(10*60*1000*10),
					 1.0/(6.5*60*60*1000),
					 1.0/(6.5*60*60*1000*10));

  ConstantMultifactorVolatilityModelWithJumps const_vol_model_jumps =
    ConstantMultifactorVolatilityModelWithJumps(&observational_model,
						dt);
  // model.print_data();
  // model.print_parameters();
  return 0;
}
