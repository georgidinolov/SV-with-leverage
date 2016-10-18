#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "ParamTypes.hpp"
#include "DataTypes.hpp"

using namespace std;

int main ()
{
  double dt = 1.0;
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::normal_distribution<double> normal(0,0.01);

  int data_size = 100;
  OpenCloseData data(data_size);

  for (unsigned i=0; i<data.data_length(); ++i) {
    OpenCloseDatum datum = OpenCloseDatum(100, 100*(1+normal(mt)), dt, 0);
    
    if (i>0) {
      datum.set_open(data.get_data_element(i-1).get_close());
      datum.set_close(data.get_data_element(i-1).get_close()*(1+normal(mt)));
      datum.set_delta_t(dt);
      datum.set_t_beginning(data.get_data_element(i-1).get_t_beginning() + 
			    data.get_data_element(i-1).get_delta_t());
    }
    data.set_data_element(i, datum);
  }
  StochasticVolatilityParams params = StochasticVolatilityParams(dt);
  GammaParameter gammas = GammaParameter(data.data_length());
  gammas.set_gammas(std::vector<int> (data.data_length(), 1));
  params.set_gammas(gammas);
  const std::vector<int>& new_gammas = params.get_gammas().get_gammas();
  for (unsigned i=0; i<new_gammas.size(); ++i) {
    std::cout << new_gammas[i] << std::endl;
  }
  return 0;
}
