#include <iostream>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include <stdexcept>
#include <vector>

class ThetaTauSquareRhoProposal
{
public:
  ThetaTauSquareRhoProposal();
  ThetaTauSquareRhoProposal(const gsl_matrix * proposal_covariance_matrix);
  ~ThetaTauSquareRhoProposal();

  const gsl_matrix * get_proposal_covariance_matrix() const;

  void set_theta_upper_bound(double upper_bound);
  void set_theta_lower_bound(double lower_bound);

  // returns a vector of proposed rho, theta(delta),
  // tau(delta)^2 on the NOMINAL SCALE
  //
  // mean is ALSO ON THE NOMINAL
  std::vector<double> propose_parameters(const gsl_rng *r,
					 const std::vector<double>& mean) const;

private:
  // order of parameters is
  // rho^tilde = logit((rho+1)/2),
  // theta_tilde = logit(theta(delta)),
  // tau_tilde = log(tau(delta)^2)
  gsl_matrix * proposal_covariance_matrix_;

  double theta_upper_bound_;
  double theta_lower_bound_;
};

// ============================================
class RhoThetaTauSquareTildeProposal
{
public:
  RhoThetaTauSquareTildeProposal(const gsl_vector * proposal_tilde_mean,
				 const gsl_matrix * proposal_tilde_covariance_matrix,
				 int dof);
  ~RhoThetaTauSquareTildeProposal();

  const gsl_vector * get_proposal_mean() const;
  const gsl_matrix * get_proposal_covariance_matrix() const;

  // returns a vector of proposed rho_tilde, theta(delta)_tilde,
  // tau(delta)^2_tilde on the TILDE SCALE
  //
  // mean is ALSO ON THE TILDE SCALE
  std::vector<double> propose_parameters(const gsl_rng *r) const;
  double q_log_likelihood(const std::vector<double>& proposed_tilde) const;

private:
  // order of parameters is
  // rho^tilde = logit((rho+1)/2),
  // theta_tilde = logit(theta(delta)),
  // tau_tilde = log(tau(delta)^2)

  gsl_vector * proposal_tilde_mean_;
  gsl_matrix * proposal_tilde_covariance_matrix_;
  int dof_;
};

// =============================================
class NormalRWProposal
{
public:
  NormalRWProposal(int d, const gsl_matrix * proposal_covariance_matrix);
  ~NormalRWProposal();

  // returns a vector of proposed parameters on the scale of the
  // proposal matrix
  virtual std::vector<double> propose_parameters(const gsl_rng *r,
						 const std::vector<double>& mean) const;

  std::vector<double> propose_parameters(const gsl_rng *r,
					 const int d,
					 const std::vector<double>& mean,
					 const gsl_matrix * proposal_covariance_matrix) const;

  // proposes with student t with nu dof
  std::vector<double> propose_parameters(const gsl_rng *r,
					 const int d,
					 const std::vector<double>& mean,
					 const gsl_matrix * proposal_covariance_matrix,
					 const int dof) const;
  inline const gsl_matrix* get_proposal_covariance_matrix() const {
    return proposal_covariance_matrix_;
  }
  inline int get_d() const {
    return d_;
  }
  void set_proposal_covariance_matrix(const gsl_matrix* new_cov_mat);

private:
  int d_;
  gsl_matrix * proposal_covariance_matrix_;
};

// =============================================
// from http://probability.ca/jeff/ftpdir/adaptex.pdf, page 3 equation 3
class AdaptiveNormalRWProposal
  : public NormalRWProposal
{
public:
  AdaptiveNormalRWProposal(int d,
			   const gsl_matrix * proposal_covariance_matrix,
			   double beta);
  ~AdaptiveNormalRWProposal();

  void update_proposal(const std::vector<double>& new_sample);
  std::vector<double> propose_parameters(const gsl_rng *r,
					 const std::vector<double>& mean) const;

private:
  gsl_vector* sum_vector_;
  gsl_matrix* sum_sq_matrix_;
  gsl_matrix* empirical_proposal_matrix_;
  int n_; // keeping track of how many samples have been computed
  double beta_;
};
