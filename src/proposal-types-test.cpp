#include <iostream>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include "ProposalTypes.hpp"
#include <stdio.h>
#include <vector>

using namespace std;

int main ()
{
  gsl_rng_env_setup();
  const gsl_rng_type * T = gsl_rng_default;
  gsl_rng * r = gsl_rng_alloc (T);

  gsl_matrix * proposal_matrix = gsl_matrix_alloc(3,3);
  gsl_matrix_set_zero(proposal_matrix);

  // rho, theta, tau
  std::vector<double> mean {0, 0.1, 1e-7};

  gsl_matrix_set(proposal_matrix, 0, 0, 1e-5);
  gsl_matrix_set(proposal_matrix, 1, 1, 1e-5);
  gsl_matrix_set(proposal_matrix, 2, 2, 1e-5);

  ThetaTauSquareRhoProposal prop = ThetaTauSquareRhoProposal(proposal_matrix);
  
  std::vector<double> proposal = prop.propose_parameters(r,mean);
			  
  for (int j=0; j < 100; ++j) {
    proposal = prop.propose_parameters(r,mean);
    for (int i=0; i<3; ++i) {
      std::cout << proposal[i] << " ";
    }
    std::cout << std::endl;
  }

  gsl_matrix_free(proposal_matrix);
  return 0;
}
