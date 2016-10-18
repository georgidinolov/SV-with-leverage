#include <iostream>
#include <vector>
#include "DataTypes.hpp"

using namespace std;

int main ()
{
  unsigned N = 10;
  std::vector<OpenCloseDatum> data_vector;

  for (unsigned i=0; i<N; ++i) {
    OpenCloseDatum datum = OpenCloseDatum(i,i,1,0);
    data_vector.push_back(datum);
  }

  // Clone datum
  OpenCloseDatum * next_element = data_vector[N-1].clone();
  std::cout << "Cloned element: " << *next_element << std::endl;
  delete next_element;
  
  // Equals datum
  OpenCloseDatum next_next_element = data_vector[N-1];
  std::cout << "Equalled element: " << next_next_element << std::endl;
  
  OpenCloseData data = OpenCloseData(data_vector);

  // Clone data
  OpenCloseData * cloned_data = data.clone();
  std::cout << "Cloned data\n";
  std::cout << *cloned_data << std::endl;

  // Eqauls data
  OpenCloseData equals_data = data;
  std::cout << "Equalled data\n";
  std::cout << equals_data << std::endl;

  // Testing the data elements
  std::cout << "Testing members of the data elements\n";
  for (unsigned i=0; i<data.data_length(); ++i) {
    std::cout << data.get_data_element(i);
  }
  std::cout << std::endl;

  // Testing members of the data set
  std::cout << "Testing members of the data set\n";
  std::cout << data;
  std::cout << std::endl;

  // Testing print_data()
  std::cout << "Testing print data set\n";
  data.print_data();
  std::cout << std::endl;

  return 0;
}
