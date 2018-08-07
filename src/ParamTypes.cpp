#include <cmath>
#include "ParamTypes.hpp"

// ==================== MU PARAM ========================
MuParameter::MuParameter()
  : mu_hat_(1.7e-10)
{}

MuParameter::MuParameter(double delta_t)
  : mu_hat_(1.7e-10),
    delta_t_(delta_t),
    constant_delta_t_(true)
{}

MuParameter::MuParameter(double mu_hat,
			 double delta_t)
  : mu_hat_(mu_hat),
    delta_t_(delta_t),
    constant_delta_t_(true)
{}

double MuParameter::get_continuous_time_parameter() const
{
  return mu_hat_;
}

double MuParameter::get_discrete_time_parameter(double delta_t) const
{
  return mu_hat_ * delta_t;
}

void MuParameter::set_continuous_time_parameter(double mu_hat)
{
  mu_hat_ = mu_hat;
}

// ==================== THETA PARAM ====================
ThetaParameter::ThetaParameter()
  : theta_hat_(5.6e-7)
{}

ThetaParameter::ThetaParameter(double delta_t)
  : theta_hat_(5.6e-7),
    delta_t_(delta_t),
    constant_delta_t_(true)
{}

double ThetaParameter::get_continuous_time_parameter() const
{
  return theta_hat_;
}

double ThetaParameter::get_discrete_time_parameter(double delta_t) const
{
  return exp(-theta_hat_ * delta_t);
}

void ThetaParameter::set_continuous_time_parameter(double theta_hat)
{
  theta_hat_ = theta_hat;
}

// =============== TAU SQ PARAM =================================
TauSquareParameter::TauSquareParameter(const ThetaParameter& theta_parameter)
  : tau_square_hat_(1.3e-7),
    theta_parameter_(theta_parameter)
{}

TauSquareParameter::TauSquareParameter(const ThetaParameter& theta_parameter,
				       double delta_t)
  : tau_square_hat_(1.3e-7),
    theta_parameter_(theta_parameter),
    delta_t_(delta_t),
    constant_delta_t_(true)
{}

double TauSquareParameter::get_continuous_time_parameter() const
{
  return tau_square_hat_;
}

double TauSquareParameter::get_discrete_time_parameter(double delta_t) const
{
  return tau_square_hat_ *
    (1 - exp(-2.0*theta_parameter_.get_continuous_time_parameter()*delta_t)) / 
    (2.0 * theta_parameter_.get_continuous_time_parameter());
}

void TauSquareParameter::set_continuous_time_parameter(double tau_square_hat) 
{
  tau_square_hat_ = tau_square_hat;
}

// ===================== THETA TAU SQ PARAM ===================
ThetaTauSquareParameter::ThetaTauSquareParameter()
  : theta_parameter_(ThetaParameter()),
    tau_square_parameter_(theta_parameter_)
{}

ThetaTauSquareParameter::ThetaTauSquareParameter(double delta_t)
  : theta_parameter_(ThetaParameter(delta_t)),
    tau_square_parameter_(theta_parameter_,
			  delta_t),
    delta_t_(delta_t),
    constant_delta_t_(true)
{}

const ThetaParameter& ThetaTauSquareParameter::get_theta_parameter() const
{
  return theta_parameter_;
}

const TauSquareParameter& ThetaTauSquareParameter::get_tau_square_parameter() const
{
  return tau_square_parameter_;
}

void ThetaTauSquareParameter::set_tau_square_hat(double tau_sq_hat)
{
  tau_square_parameter_.set_continuous_time_parameter(tau_sq_hat);
}

void ThetaTauSquareParameter::set_theta_hat(double theta_hat)
{
  theta_parameter_.set_continuous_time_parameter(theta_hat);
}

// ====================== ALPHA PARAM ==========================
AlphaParameter::AlphaParameter()
  : alpha_hat_(-13)
{}

AlphaParameter::AlphaParameter(double delta_t)
  : alpha_hat_(-13),
    delta_t_(delta_t),
    constant_delta_t_(true)
{}

double AlphaParameter::get_continuous_time_parameter() const
{
  return alpha_hat_;
}

double AlphaParameter::get_discrete_time_parameter(double delta_t) const
{
  return alpha_hat_ + 0.5*log(delta_t);
}

void AlphaParameter::set_continuous_time_parameter(double alpha_hat)
{
  alpha_hat_ = alpha_hat;
}

// ===================== SIGMA SINGLETON PARAM ================
SigmaSingletonParameter::SigmaSingletonParameter(double sigma_hat)
  : sigma_hat_(sigma_hat)
{}

SigmaSingletonParameter::SigmaSingletonParameter(double sigma_hat,
						 double delta_t)
  : sigma_hat_(sigma_hat),
    delta_t_(delta_t),
    constant_delta_t_(true)
{}

double SigmaSingletonParameter::get_continuous_time_parameter() const
{
  return sigma_hat_;
}

double SigmaSingletonParameter::get_discrete_time_parameter(double delta_t) const
{
  return sigma_hat_ * sqrt(delta_t);
}

void SigmaSingletonParameter::set_continuous_time_parameter(double sigma_hat)
{
  sigma_hat_ = sigma_hat;
}

std::ostream& operator <<(std::ostream& output_stream, 
			  const SigmaSingletonParameter& param)
{
  output_stream << "(sigma_hat=" 
		<< param.sigma_hat_ 
		<< ", delta_t=" 
		<< param.delta_t_ << ")";
  return output_stream;
}

// ====================== SIGMA PARAMETER ======================
SigmaParameter::SigmaParameter()
  : sigmas_(std::vector<SigmaSingletonParameter> ()),
    discrete_time_log_sigmas_(std::vector<double> ()),
    delta_t_(0.0),
    constant_delta_t_(false)
{}

SigmaParameter::SigmaParameter(double delta_t)
  : sigmas_(std::vector<SigmaSingletonParameter> ()),
    discrete_time_log_sigmas_(std::vector<double> ()),
    delta_t_(delta_t),
    constant_delta_t_(true)
{}

SigmaParameter::SigmaParameter(const std::vector<double>& sigma_hats,
			       double delta_t)
{
  set_sigmas(sigma_hats,
	     delta_t);
}

SigmaParameter::SigmaParameter(const std::vector<double>& sigma_hats,
			       const std::vector<double>& delta_ts)
{
  set_sigmas(sigma_hats,
	     delta_ts);
}

const std::vector<SigmaSingletonParameter>& SigmaParameter::get_sigmas() const
{
  return sigmas_;
}

const std::vector<double>& SigmaParameter::get_discrete_time_log_sigmas() const
{
  return discrete_time_log_sigmas_;
}

void SigmaParameter::set_sigmas(const std::vector<double>& sigma_hats,
				double delta_t)
{
  delta_t_ = delta_t;
  constant_delta_t_ = true;

  sigmas_ = std::vector<SigmaSingletonParameter> ();
  discrete_time_log_sigmas_ = std::vector<double> ();

  for (unsigned i=0; i<sigma_hats.size(); ++i) {
    SigmaSingletonParameter sigma = 
      SigmaSingletonParameter(sigma_hats[i], delta_t);
    sigmas_.push_back(sigma);
    discrete_time_log_sigmas_.
      push_back(log(sigma.get_discrete_time_parameter(delta_t)));
  }
}

void SigmaParameter::set_sigma_hat_element(unsigned i, double sigma_hat)
{
  sigmas_[i].set_continuous_time_parameter(sigma_hat);
}

void SigmaParameter::set_discrete_time_log_sigma_element(unsigned i, double log_sigma)
{
  discrete_time_log_sigmas_[i] = log_sigma;
}

void SigmaParameter::set_sigmas(const std::vector<double>& sigma_hats,
				const std::vector<double>& delta_ts)
{
  delta_t_ = 0.0;
  constant_delta_t_ = false;

  sigmas_ = std::vector<SigmaSingletonParameter> ();
  discrete_time_log_sigmas_ = std::vector<double> ();

  for (unsigned i=0; i<sigma_hats.size(); ++i) {
    SigmaSingletonParameter sigma = 
      SigmaSingletonParameter(sigma_hats[i], delta_ts[i]);
    sigmas_.push_back(sigma);
    discrete_time_log_sigmas_.
      push_back(log(sigma.get_discrete_time_parameter(delta_ts[i])));
  }
}

std::ostream& operator <<(std::ostream& output_stream, 
			  const SigmaParameter& param)
{
  for (unsigned i=0; i<param.sigmas_.size(); ++i) {
    output_stream << param.sigmas_[i] << "\n";
  }
  return output_stream;
}

// ======================= RHO PARAMETER ============================
RhoParameter::RhoParameter()
  : rho_(0)
{}

RhoParameter::RhoParameter(double rho)
  : rho_(rho)
{}

double RhoParameter::get_continuous_time_parameter() const
{
  return rho_;
}

double RhoParameter::get_discrete_time_parameter(double delta_t) const
{
  return rho_;
}

void RhoParameter::set_continuous_time_parameter(double rho) 
{
  rho_ = rho;
}

double RhoParameter::nominal_to_tilde(double rho) const
{
  return -1.0*log(2.0/(rho+1) - 1);
}

double RhoParameter::tilde_to_nominal(double rho_tilde) const
{
  return 2*exp(rho_tilde)/(exp(rho_tilde) + 1.0) - 1;
}

// ========================== XI PARAMTER =========================
XiSquareParameter::XiSquareParameter()
  : xi_square_(2.5e-11)
{}

XiSquareParameter::XiSquareParameter(double xi_square)
  : xi_square_(xi_square)
{}

double XiSquareParameter::get_continuous_time_parameter() const
{
  return xi_square_;
}

double XiSquareParameter::get_discrete_time_parameter(double delta_t) const
{
  return xi_square_;
}

void XiSquareParameter::set_continuous_time_parameter(double xi_square) 
{
  xi_square_ = xi_square;
}

double XiSquareParameter::tilde_to_nominal(double xi_square_tilde) const
{
  return exp(xi_square_tilde);
}

// ===================== GAMMA PARAMTER =========================
GammaParameter::GammaParameter()
  : J_(10),
    mixture_probabilities_(std::vector<double> 
			   {0.00609,
			       0.04775,
			       0.13057,
			       0.20674,
			       0.22715,
			       0.18842,
			       0.12047,
			       0.05591,
			       0.01575,
			       0.00115}),
    mixture_means_(std::vector<double> 
		   {1.92677,
		       1.34744,
		       0.73504,
		       0.02266,
		       -0.85173,
		       -1.97278,
		       -3.46788,
		       -5.55246,
		       -8.68384,
		       -14.65000}),
    mixture_variances_(std::vector<double> 
		       {0.11265,
			   0.17788,
			   0.26768,
			   0.40611,
			   0.62699,
			   0.98583,
			   1.57469,
			   2.54498,
			   4.16591,
			   7.33342})
{}

GammaParameter::GammaParameter(unsigned data_length)
  : J_(10),
    mixture_probabilities_(std::vector<double> 
			   {0.00609,
			       0.04775,
			       0.13057,
			       0.20674,
			       0.22715,
			       0.18842,
			       0.12047,
			       0.05591,
			       0.01575,
			       0.00115}),
    mixture_means_(std::vector<double> 
		   {1.92677,
		       1.34744,
		       0.73504,
		       0.02266,
		       -0.85173,
		       -1.97278,
		       -3.46788,
		       -5.55246,
		       -8.68384,
		       -14.65000}),
    mixture_variances_(std::vector<double> 
		       {0.11265,
			   0.17788,
			   0.26768,
			   0.40611,
			   0.62699,
			   0.98583,
			   1.57469,
			   2.54498,
			   4.16591,
			   7.33342}),
   data_length_(data_length),
   gammas_(std::vector<int> (data_length_))
{}

const std::vector<int>& GammaParameter::get_gammas() const
{
  return gammas_;
}

const std::vector<double>& GammaParameter::get_mixture_means() const
{
  return mixture_means_;
}

const std::vector<double>& GammaParameter::get_mixture_variances() const
{
  return mixture_variances_;
}

const std::vector<double>& GammaParameter::get_mixture_probabilities() const
{
  return mixture_probabilities_;
}

int GammaParameter::get_J() const
{
  return J_;
}

void GammaParameter::set_gammas(const std::vector<int>& gammas)
{
  gammas_ = gammas;
  data_length_ = gammas.size();
}

void GammaParameter::set_gamma_element(unsigned i, int new_gamma)
{
  gammas_[i] = new_gamma;
}

// ======================== LAMBDA PARAMTER ========================
LambdaParam::LambdaParam()
  : lambda_(1.0)
{}

LambdaParam::LambdaParam(double lambda) 
  : lambda_(lambda)
{
    if (lambda_ < 0.0) {
      std::cout << "WARNING: lambda < 0.0" << std::endl;
    }
}

// ===================== OBSERVATIONAL MODEL PARAMS ======================
ObservationalParams::ObservationalParams()
  : xi_square_(XiSquareParameter())
{}

const XiSquareParameter& ObservationalParams::get_xi_square() const
{
  return xi_square_;
}

void ObservationalParams::set_xi_square(const XiSquareParameter& xi_square)
{
  xi_square_ = xi_square;
}

// ===================== CONSTANT VOL PARAMS ==============================
ConstantVolatilityParams::ConstantVolatilityParams()
  : mu_(MuParameter(1.0)),
    gammas_(GammaParameter())
{}

ConstantVolatilityParams::ConstantVolatilityParams(double delta_t)
  : mu_(MuParameter(delta_t)),
    gammas_(GammaParameter())
{}

ConstantVolatilityParams::ConstantVolatilityParams(unsigned data_length,
						   double delta_t)
  : mu_(MuParameter(delta_t)),
    gammas_(GammaParameter(data_length))
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

void ConstantVolatilityParams::set_gamma_element(unsigned i, int new_gamma)
{
  gammas_.set_gamma_element(i, new_gamma);
}

// ===================== STOCHASTIC VOL PARAMS ========================
StochasticVolatilityParams::StochasticVolatilityParams()
  : mu_(MuParameter(1.0)),
    theta_tau_square_(ThetaTauSquareParameter(1.0)),
    alpha_(AlphaParameter(1.0)),
    sigmas_(SigmaParameter(1.0)),
    gammas_(GammaParameter()),
    xi_square_(XiSquareParameter()),
    rho_(RhoParameter())
{};

StochasticVolatilityParams::StochasticVolatilityParams(double delta_t)
  : mu_(MuParameter(delta_t)),
    theta_tau_square_(ThetaTauSquareParameter(delta_t)),
    alpha_(AlphaParameter(delta_t)),
    sigmas_(SigmaParameter(delta_t)),
    gammas_(GammaParameter()),
    xi_square_(XiSquareParameter()),
    rho_(RhoParameter())
{};

StochasticVolatilityParams * StochasticVolatilityParams::clone()const 
{
  return new StochasticVolatilityParams(*this);
}

// StochasticVolatilityParams & 
// StochasticVolatilityParams::operator=(const StochasticVolatilityParams &rhs)
// {
//   if(&rhs != this){
//     mu_hat_ = rhs.mu_hat_;
//     theta_hat_ = rhs.theta_hat_;
//     alpha_hat_ = rhs.alpha_hat_;
//     tau_square_hat_ = rhs.tau_square_hat_;
//     xi_square_ = rhs.xi_square_;
//     rho_ = rhs.rho_;
//   }
//   return *this;
// }

const MuParameter& StochasticVolatilityParams::get_mu() const
{
  return mu_;
}

const ThetaParameter& StochasticVolatilityParams::get_theta() const
{
  return theta_tau_square_.get_theta_parameter();
}

const TauSquareParameter& StochasticVolatilityParams::get_tau_square() const
{
  return theta_tau_square_.get_tau_square_parameter();
}

const AlphaParameter& StochasticVolatilityParams::get_alpha() const
{
  return alpha_;
}

const SigmaParameter& StochasticVolatilityParams::get_sigmas() const
{
  return sigmas_;
}

const GammaParameter& StochasticVolatilityParams::get_gammas() const
{
  return gammas_;
}

const XiSquareParameter& StochasticVolatilityParams::get_xi_square() const
{
  return xi_square_;
}
 
const RhoParameter& StochasticVolatilityParams::get_rho() const
{
  return rho_;
}

void StochasticVolatilityParams::set_sigmas(const SigmaParameter& sigmas)
{
  sigmas_ = sigmas;
}

void StochasticVolatilityParams::set_sigmas_element(unsigned i, 
						    double sigma_hat, 
						    double log_sigma)
{
  sigmas_.
    set_discrete_time_log_sigma_element(i, log_sigma);

  sigmas_.
    set_sigma_hat_element(i, sigma_hat);
    
}

void StochasticVolatilityParams::set_gammas(const std::vector<int>& gammas)
{
  gammas_.set_gammas(gammas);
}

void StochasticVolatilityParams::set_gammas(const GammaParameter& gammas)
{
  gammas_.set_gammas(gammas.get_gammas());
}

void StochasticVolatilityParams::set_tau_square_hat(double tau_square_hat)
{
  theta_tau_square_.set_tau_square_hat(tau_square_hat);
}

void StochasticVolatilityParams::set_theta_hat(double theta_hat)
{
  theta_tau_square_.set_theta_hat(theta_hat);
}

void StochasticVolatilityParams::set_alpha_hat(double alpha_hat)
{
  alpha_.set_continuous_time_parameter(alpha_hat);
}

void StochasticVolatilityParams::set_mu_hat(double mu_hat)
{
  mu_.set_continuous_time_parameter(mu_hat);
}

void StochasticVolatilityParams::set_rho(double rho)
{
  rho_.set_continuous_time_parameter(rho);
}

void StochasticVolatilityParams::set_xi_square(double xi_square)
{
  xi_square_.set_continuous_time_parameter(xi_square);
}

void StochasticVolatilityParams::print_parameters() const
{
  std::cout << sigmas_;
  std::cout << "\n" << std::endl;
}

// std::ostream& operator <<(std::ostream& output_stream, 
// 			  const StochasticVolatilityParams& params)
// {
//   output_stream << "mu_hat = " << params.mu_hat_ << "\n"
// 		<< "theta_hat = " << params.theta_hat_ << "\n"
// 		<< "alpha_hat = " << params.alpha_hat_ << "\n"
// 		<< "tau2_hat = " << params.tau_square_hat_ << "\n"
// 		<< "xi2 = " << params.xi_square_ << "\n" 
// 		<< "rho = " << params.rho_ << "\n"
// 		<< "sigma_hat = ";

//   for (unsigned i=0; i<params.sigma_hat_.size(); ++i) {
//     output_stream << params.sigma_hat_[i] << ", ";
//   }
//   output_stream << "\n";
//   return output_stream;
// }


// ===================== MULTIFACTOR SV PARAMS ==============================
MultifactorStochasticVolatilityParams::MultifactorStochasticVolatilityParams()
  : mu_(MuParameter(1.0)),
    gammas_(GammaParameter()),
    xi_square_(XiSquareParameter())
{};

MultifactorStochasticVolatilityParams::MultifactorStochasticVolatilityParams(double delta_t)
  : mu_(MuParameter(delta_t)),
    gammas_(GammaParameter()),
    xi_square_(XiSquareParameter())
{};

MultifactorStochasticVolatilityParams * MultifactorStochasticVolatilityParams::clone()const 
{
  return new MultifactorStochasticVolatilityParams(*this);
}

// MultifactorStochasticVolatilityParams & 
// MultifactorStochasticVolatilityParams::operator=(const MultifactorStochasticVolatilityParams &rhs)
// {
//   if(&rhs != this){
//     mu_hat_ = rhs.mu_hat_;
//     theta_hat_ = rhs.theta_hat_;
//     alpha_hat_ = rhs.alpha_hat_;
//     tau_square_hat_ = rhs.tau_square_hat_;
//     xi_square_ = rhs.xi_square_;
//     rho_ = rhs.rho_;
//   }
//   return *this;
// }

const MuParameter& MultifactorStochasticVolatilityParams::get_mu() const
{
  return mu_;
}

const GammaParameter& MultifactorStochasticVolatilityParams::get_gammas() const
{
  return gammas_;
}

const XiSquareParameter& MultifactorStochasticVolatilityParams::get_xi_square() const
{
  return xi_square_;
}
 
void MultifactorStochasticVolatilityParams::set_gammas(const std::vector<int>& gammas)
{
  gammas_.set_gammas(gammas);
}

void MultifactorStochasticVolatilityParams::set_gammas(const GammaParameter& gammas)
{
  gammas_.set_gammas(gammas.get_gammas());
}

void MultifactorStochasticVolatilityParams::set_mu_hat(double mu_hat)
{
  mu_.set_continuous_time_parameter(mu_hat);
}

void MultifactorStochasticVolatilityParams::set_xi_square(double xi_square)
{
  xi_square_.set_continuous_time_parameter(xi_square);
}

