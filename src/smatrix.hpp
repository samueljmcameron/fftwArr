#ifndef FFTWMPI_SMATRIX_HPP
#define FFTWMPI_SMATRIX_HPP

#include <array>
#include <memory>
#include <iostream>

namespace fftwArr {
template <typename T>
class sMatrix
{
public:

  typedef T* iterator;
  typedef const T* const_iterator;
  typedef T value_type;
  
  sMatrix() 
  {
    create();
    _sizeax.at(0) = 0;
    _sizeax.at(1) = 0;
  };
  sMatrix(size_t Nx, size_t Ny, const T& t = T())
  {
    create(Nx*Ny,t);
    _sizeax.at(0) = Nx;
    _sizeax.at(1) = Ny;
  }

  sMatrix (const sMatrix & m)
  { // copy constructor
    
    create(m.begin(), m.end());
    _sizeax.at(0) = m.axis_size(0);
    _sizeax.at(1) = m.axis_size(1);
    
  } 
  sMatrix& operator= (const sMatrix &);
  

  size_t size() const { return limit - arr; }

  
  T& operator()(size_t i, size_t j) { return arr[j*_sizeax[0] + i]; }
  const T& operator()(size_t i, size_t j) const { return arr[j*_sizeax[0] + i]; }

  iterator begin() { return arr; }
  const_iterator begin() const { return arr; }


  iterator end() { return limit; }
  const_iterator end() const { return limit; }

  ~sMatrix() {
    uncreate(); };
  
  
  T* data() {
      return arr;
  };

  // note that resize destroys the current array elements
  void resize(size_t, size_t);
  
  int axis_size(int i) const
  {
    return _sizeax.at(i);
  }

  
private:
  iterator arr;
  iterator limit;

  std::array<size_t,2> _sizeax;

  std::allocator<T> alloc;
  

  void create();
  void create( size_t, const T&);
  void create(const_iterator, const_iterator);

  void uncreate();

};




};
#endif
