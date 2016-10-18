#include <iostream>
#include <vector>

class ParamBase
{
public:
  virtual double get_continuous_time_parameter()const = 0;
  virtual double get_discrete_time_parameter(double delta_t)const = 0;

  virtual void set_continuous_time_parameter(double)=0;
};

class MuParameter
  : public ParamBase
{
public:
  MuParameter();
  MuParameter(double delta_t);
  MuParameter(double mu_hat,
	      double delta_t);
  virtual double get_continuous_time_parameter() const;
  virtual double get_discrete_time_parameter(double delta_t) const;

  virtual void set_continuous_time_parameter(double mu_hat);
private:
  double mu_hat_;
  double delta_t_;
  bool constant_delta_t_;
};

class ThetaParameter
  : public ParamBase
{
public:
  ThetaParameter();
  ThetaParameter(double delta_t);
  virtual double get_continuous_time_parameter() const;
  virtual double get_discrete_time_parameter(double delta_t) const;

  virtual void set_continuous_time_parameter(double theta_hat);
private:
  double theta_hat_;
  double delta_t_;
  bool constant_delta_t_;
};


// ============ TAU SQUARE PARAMETER ============
class TauSquareParameter
  : public ParamBase
{
public:
  TauSquareParameter(const ThetaParameter& theta_parameter);
  TauSquareParameter(const ThetaParameter& theta_parameter,
		     double delta_t);

  virtual double get_continuous_time_parameter() const;
  virtual double get_discrete_time_parameter(double delta_t) const;

  virtual void set_continuous_time_parameter(double tau_square_hat);

private:
  double tau_square_hat_;
  const ThetaParameter& theta_parameter_;

  double delta_t_;
  bool constant_delta_t_;
};

// ============ THETA TAU SQUARE PARAMETER ============
class ThetaTauSquareParameter
{
public:
  ThetaTauSquareParameter();
  ThetaTauSquareParameter(double delta_t);

  const ThetaParameter& get_theta_parameter() const;
  const TauSquareParameter& get_tau_square_parameter() const;

  void set_tau_square_hat(double tau_sq_hat);
  void set_theta_hat(double theta_hat);
private:
  ThetaParameter theta_parameter_;
  TauSquareParameter tau_square_parameter_;

  double delta_t_;
  bool constant_delta_t_;
};

// ============= ALPHA PARAMETER =====================
class AlphaParameter
  : public ParamBase
{
public:
  AlphaParameter();
  AlphaParameter(double delta_t);

  virtual double get_continuous_time_parameter() const;
  virtual double get_discrete_time_parameter(double delta_t) const;

  virtual void set_continuous_time_parameter(double alpha_hat);
private:
  double alpha_hat_;
  
  double delta_t_;
  bool constant_delta_t_;
};

// ============== SIGMA SINGLETON PARAMETER =========
class SigmaSingletonParameter
  : public ParamBase
{
public:
  SigmaSingletonParameter(double sigma_hat);
  SigmaSingletonParameter(double sigma_hat,
			  double delta_t);

  virtual double get_continuous_time_parameter() const;
  virtual double get_discrete_time_parameter(double delta_t) const;

  virtual void set_continuous_time_parameter(double sigma_hat);

  friend std::ostream& operator <<(std::ostream& output_stream, 
  				   const SigmaSingletonParameter& param);
private:
  double sigma_hat_;
  
  double delta_t_;
  bool constant_delta_t_;
};


// ============== SIGMA PARAMETER =========
class SigmaParameter
{
public:
  SigmaParameter();
  SigmaParameter(double delta_t);
  SigmaParameter(const std::vector<double>& sigma_hats,
		 double delta_t);

  SigmaParameter(const std::vector<double>& sigma_hats,
  		 const std::vector<double>& delta_ts);

  const std::vector<SigmaSingletonParameter>& get_sigmas() const;
  const std::vector<double>& get_discrete_time_log_sigmas() const;

  void set_sigmas(const std::vector<double>& sigma_hats,
  		  double delta_t);
  
  void set_sigma_hat_element(unsigned i, double sigma_hat);
  void set_discrete_time_log_sigma_element(unsigned i, double log_sigma);

  void set_sigmas(const std::vector<double>& sigma_hats,
  		  const std::vector<double>& delta_ts);

  friend std::ostream& operator <<(std::ostream& output_stream, 
  				   const SigmaParameter& params);

private:
  std::vector<SigmaSingletonParameter> sigmas_;
  std::vector<double> discrete_time_log_sigmas_;

  double delta_t_;
  bool constant_delta_t_;
};

// ================ RHO PARAMETER =================
class RhoParameter
  : public ParamBase
{
public:
  RhoParameter();
  RhoParameter(double rho);

  virtual double get_continuous_time_parameter() const;
  virtual double get_discrete_time_parameter(double delta_t) const;
  virtual void set_continuous_time_parameter(double rho);
  virtual double nominal_to_tilde(double rho) const;
  virtual double tilde_to_nominal(double rho_tilde) const;
private:
  double rho_;
};

// ================ XI SQUARE PARAMETER =================
class XiSquareParameter
  : public ParamBase
{
public:
  XiSquareParameter();
  XiSquareParameter(double xi_square);

  virtual double get_continuous_time_parameter() const;
  virtual double get_discrete_time_parameter(double delta_t) const;
  virtual void set_continuous_time_parameter(double xi_square);

  virtual double tilde_to_nominal(double xi_square_tilde) const;
private:
  double xi_square_;
};

// ============= GAMMA PARAMETER ===========================
class GammaParameter
{
public:
  GammaParameter();
  GammaParameter(unsigned length);

  const std::vector<int>& get_gammas() const;
  const std::vector<double>& get_mixture_means() const;
  const std::vector<double>& get_mixture_variances() const;
  const std::vector<double>& get_mixture_probabilities() const;
  int get_J() const;
  
  void set_gammas(const std::vector<int>& new_gammas);
  void set_gamma_element(unsigned i, int new_gamma);

private:
  int J_;
  std::vector<double> mixture_probabilities_;
  std::vector<double> mixture_means_;
  std::vector<double> mixture_variances_;

  unsigned data_length_;
  std::vector<int> gammas_;
};

// =============== LAMBDA PARAMETER =========================
class LambdaParam
{
public: 
  LambdaParam();
  LambdaParam(double lambda);

  inline double get_lambda() const {
    return lambda_;
  }
  inline void set_lambda(double lambda) {
    lambda_ = lambda;
    if (lambda_ < 0.0) {
      std::cout << "WARNING: lambda < 0.0" << std::endl;
    }
  }
  
private:
  double lambda_;
};

// ======================== SIGMA SQ PRIOR =============================
class SigmaSquareParam
{
public:
  inline SigmaSquareParam()
    : sigma_square_(1.0)
  {}
  inline SigmaSquareParam(double sigma_square)
    : sigma_square_(sigma_square)
  {
    if (sigma_square_ < 0.0) {
      std::cout << "WARNING: sigma_square < 0.0" << std::endl;
    }
  }

  inline double get_sigma_square() const {
    return sigma_square_;
  }
  inline void set_sigma_square(double sigma_square) {
    sigma_square_ = sigma_square;
    if (sigma_square_ < 0.0) {
      std::cout << "WARNING: sigma_square < 0.0" << std::endl;
    }
  }

private:
  double sigma_square_;
};

// ===================== CONSTANT VOL PARAMS ===========================
class ConstantVolatilityParams
{
public:
  ConstantVolatilityParams();
  ConstantVolatilityParams(double delta_t);
  ConstantVolatilityParams(unsigned data_length,
			   double delta_t);

  ConstantVolatilityParams * clone()const;
  // ConstantVolatilityParams & operator=(const ConstantVolatilityParams &rhs);

  const MuParameter& get_mu() const;
  const GammaParameter& get_gammas() const; 

  inline void set_mu_hat(double mu_hat) 
  {
    mu_.set_continuous_time_parameter(mu_hat);
  }
  void set_gammas(const std::vector<int>& gammas);
  void set_gammas(const GammaParameter& gammas);
  void set_gamma_element(unsigned i, int new_gamma);

private:
  MuParameter mu_;
  GammaParameter gammas_;
};

// ===================== OBSERVATIONAL MODEL PARAMS ========================
class ObservationalParams
{
public:
  ObservationalParams();

  const XiSquareParameter& get_xi_square() const;
  void set_xi_square(const XiSquareParameter& xi_square_param);

  inline void set_xi_square(double xi_square) {
    xi_square_.set_continuous_time_parameter(xi_square);
  }

private:
  XiSquareParameter xi_square_;
};

// ===================== SV PARAMS ===========================
class StochasticVolatilityParams
{
public:
  StochasticVolatilityParams();
  StochasticVolatilityParams(double delta_t);

  StochasticVolatilityParams * clone()const;
  // StochasticVolatilityParams & operator=(const StochasticVolatilityParams &rhs);

  const MuParameter& get_mu() const;
  const ThetaParameter& get_theta() const;
  const TauSquareParameter& get_tau_square() const;
  const AlphaParameter& get_alpha() const;
  const SigmaParameter& get_sigmas() const; 
  const GammaParameter& get_gammas() const; 
  const XiSquareParameter&  get_xi_square() const;
  const RhoParameter& get_rho() const;

  void set_sigmas(const SigmaParameter& sigmas);
  void set_sigmas_element(unsigned i, double sigma_hat, double log_sigma);
  void set_tau_square_hat(double tau_square_hat);
  void set_theta_hat(double theta_hat);
  void set_alpha_hat(double alpha_hat);
  void set_mu_hat(double mu_hat);
  void set_rho(double rho);
  void set_xi_square(double xi_square);

  void set_gammas(const std::vector<int>& gammas);
  void set_gammas(const GammaParameter& gammas);

  void print_parameters() const;
  // friend std::ostream& operator <<(std::ostream& output_stream, 
  // 				   const StochasticVolatilityParams& params);

private:
  MuParameter mu_;
  ThetaTauSquareParameter theta_tau_square_;
  AlphaParameter alpha_;
  SigmaParameter sigmas_;
  GammaParameter gammas_;
  XiSquareParameter xi_square_;
  RhoParameter rho_;
};

// ===================== MULTIFACTOR SV PARAMS ===========================
class MultifactorStochasticVolatilityParams
{
public:
  MultifactorStochasticVolatilityParams();
  MultifactorStochasticVolatilityParams(double delta_t);

  MultifactorStochasticVolatilityParams * clone()const;
  // MultifactorStochasticVolatilityParams & operator=(const MultifactorStochasticVolatilityParams &rhs);

  const MuParameter& get_mu() const;
  const GammaParameter& get_gammas() const; 
  const XiSquareParameter&  get_xi_square() const;
  const RhoParameter& get_rho() const;

  void set_mu_hat(double mu_hat);
  void set_xi_square(double xi_square);

  void set_gammas(const std::vector<int>& gammas);
  void set_gammas(const GammaParameter& gammas);

private:
  MuParameter mu_;
  GammaParameter gammas_;
  XiSquareParameter xi_square_;
};
