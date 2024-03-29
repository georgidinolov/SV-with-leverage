#include "DataTypes.hpp"

OpenCloseDatum::OpenCloseDatum() 
  : open_(0.0),
    close_(1.0),
    delta_t_(1.0),
    t_beginning_(0.0)
{}

OpenCloseDatum::OpenCloseDatum(double open,
			       double close,
			       double delta_t,
			       double t_beginning)
  : open_(open),
    close_(close),
    delta_t_(delta_t),
    t_beginning_(t_beginning)
{}

OpenCloseDatum * OpenCloseDatum::clone()const 
{
  return new OpenCloseDatum(*this);
}

OpenCloseDatum & OpenCloseDatum::operator=(const OpenCloseDatum &rhs){
  if(&rhs != this){
    open_ = rhs.open_;
    close_ = rhs.close_;
    delta_t_ = rhs.delta_t_;
    t_beginning_ = rhs.t_beginning_;
  }
  return *this;
}

double OpenCloseDatum::get_open()const 
{
  return open_;
}

double OpenCloseDatum::get_close()const
{
  return close_;
}

double OpenCloseDatum::get_delta_t()const
{
  return delta_t_;
}

double OpenCloseDatum::get_t_beginning()const
{
  return t_beginning_;
}

double OpenCloseDatum::get_t_close()const
{
  return t_beginning_ + delta_t_;
}

void OpenCloseDatum::set_open(double open)
{
  open_ = open;
}

void OpenCloseDatum::set_close(double close)
{
  close_ = close;
}

void OpenCloseDatum::set_delta_t(double delta_t)
{
  delta_t_ = delta_t;
}

void OpenCloseDatum::set_t_beginning(double t_beginning)
{
  t_beginning_ = t_beginning;
}

std::ostream& operator<<(std::ostream& output_stream, 
			 const OpenCloseDatum& OpenCloseDatum) 
{
  output_stream << "open = " << OpenCloseDatum.open_
		<< ", close = " << OpenCloseDatum.close_
		<< ", delta_t = " << OpenCloseDatum.delta_t_
		<< ", t_beginning = " << OpenCloseDatum.t_beginning_
		<< "\n";
  return output_stream;
}

// ==============================================================
OpenCloseData::OpenCloseData()
  : data_(std::vector<OpenCloseDatum>())
{}

OpenCloseData::OpenCloseData(const std::vector<OpenCloseDatum>& data)
  : data_(data)
{}

OpenCloseData::OpenCloseData(int data_size)
  : data_(std::vector<OpenCloseDatum>(data_size))
{}

OpenCloseData * OpenCloseData::clone()const 
{
  return new OpenCloseData(*this);
}

OpenCloseData & OpenCloseData::operator=(const OpenCloseData &rhs){
  if(&rhs != this){
    data_ = rhs.data_;
  }
  return *this;
}

void OpenCloseData::add_data(const OpenCloseDatum& datum)
{
  data_.push_back(datum);
}

void OpenCloseData::pop_back_data()
{
  data_.pop_back();
}

const OpenCloseDatum& OpenCloseData::get_data_element(unsigned i) const
{
  return data_[i];
}

const std::vector<OpenCloseDatum>& OpenCloseData::get_data() const
{
  return data_;
}

unsigned OpenCloseData::data_length() const
{
  return data_.size();
}

void OpenCloseData::print_data() const
{
  for (unsigned i=0; i<data_.size(); ++i) {
    std::cout << "i=" << i << "; " << data_[i];
  }
}

void OpenCloseData::set_data_element(unsigned i, const OpenCloseDatum& datum)
{
  data_[i] = datum;
}

std::ostream& operator<<(std::ostream& output_stream, 
			 const OpenCloseData& data) 
{
  for (unsigned i=0; i<data.data_.size(); ++i) {
    output_stream << "i=" << i << "; " << data.data_[i];
  }
  return output_stream;
}

