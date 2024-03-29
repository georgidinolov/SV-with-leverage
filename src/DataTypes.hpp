#include <vector>
#include <iostream>

class OpenCloseDatum
{
public:
  OpenCloseDatum();
  OpenCloseDatum(double open,
		 double close,
		 double delta_t,
		 double t_beginning);

  OpenCloseDatum * clone()const;
  OpenCloseDatum & operator=(const OpenCloseDatum &rhs);

  double get_open()const;
  double get_close()const;
  double get_delta_t()const;
  double get_t_beginning()const;
  double get_t_close()const;

  void set_open(double open);
  void set_close(double close);
  void set_delta_t(double delta_t);
  void set_t_beginning(double t_beginning);

  friend std::ostream& operator <<(std::ostream& output_stream, 
				   const OpenCloseDatum& datum);

private:
  double open_;
  double close_;
  double delta_t_;
  double t_beginning_;
};

class OpenCloseData
{
public:
  OpenCloseData();
  OpenCloseData(const std::vector<OpenCloseDatum>& data);
  OpenCloseData(int data_size);

  OpenCloseData * clone()const;
  OpenCloseData & operator=(const OpenCloseData &rhs);

  void add_data(const OpenCloseDatum& datum);
  void pop_back_data();

  const OpenCloseDatum& get_data_element(unsigned i) const;
  const std::vector<OpenCloseDatum>& get_data() const;
  unsigned data_length() const;
  void print_data() const;

  void set_data_element(unsigned i, const OpenCloseDatum& datum);

  friend std::ostream& operator <<(std::ostream& output_stream, 
				   const OpenCloseData& data);
private:
  std::vector<OpenCloseDatum> data_;
};
