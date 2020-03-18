#ifndef MATRIXASSEMBLER_H
#define MATRIXASSEMBLER_H

#ifdef KOKKOS_ENABLE_CUDA

#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <sys/time.h>

#include <cuda_runtime.h>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifndef LINEAR_SYSTEM_ASSEMBLER_DEBUG
#define LINEAR_SYSTEM_ASSEMBLER_DEBUG
#endif // LINEAR_SYSTEM_ASSEMBLER_DEBUG
//#undef LINEAR_SYSTEM_ASSEMBLER_DEBUG

namespace sierra {
namespace nalu {

template<typename IntType>
class MatrixAssembler {

public:

  /**
   * MatrixAssembler Constructor 
   *
   * @param name of the linear system being assembled
   * @param sort whether or not to sort the CSR matrix (prior to full assembly) based on the element ids
   * @param r0 first row
   * @param c0 first column
   * @param num_rows number of rows
   * @param num_cols number of columns
   * @param nDataPtsToAssemble the number of data points to assemble into a CSR matrix
   */
  MatrixAssembler(std::string name, bool sort, IntType r0, IntType c0,
		  IntType num_rows, IntType num_cols, IntType nDataPtsToAssemble);

  /**
   *  Destructor 
   */
  virtual ~MatrixAssembler();

  /**
   * memoryInGBS computes the amount of device memory used in GBs
   *
   * @return the amount of device memory used in GBs
   */
  double memoryInGBs() const;

  /**
   * memoryInGBS computes the amount of device memory used in GBs
   *
   * @return the amount of device memory used in GBs
   */
  IntType getNumNonzeros() const;

  /**
   * copySrcDataToDevice copies the COO source data to the device
   *
   * @param rows host pointer for the row coordinates
   * @param cols host pointer for the column coordinates
   * @param data host pointer for the data values
   */
  void copySrcDataToDevice(const IntType * rows, const IntType * cols, const double * data);

  /**
   * copySrcDataToDevice copies the COO source data from Kokkos views
   *
   * @param rows host pointer for the row coordinates
   * @param cols host pointer for the column coordinates
   * @param data host pointer for the data values
   */
  void copySrcDataFromKokkos(const IntType * rows, const IntType * cols, const double * data);

  /**
   * copyAssembledCSRMatrixToHost copies the assembled CSR matrix to the host (page locked memory)
   */
  void copyAssembledCSRMatrixToHost();

  /**
   * copyAssembledCSRMatrixToHost copies the assembled CSR matrix to the host
   *
   * @param rows host pointer for the row coordinates
   * @param cols host pointer for the column coordinates
   * @param data host pointer for the data values
   */
  void copyAssembledCSRMatrixToHost(int * rows, IntType * cols, double * data);

  /**
   * assemble : assemble the symbolic and numeric parts of the CSR matrix
   *
   */
  void assemble();

  /**
   * reorder to Hypre format : [D|L|U] from [L|D|U]
   */
  void reorderDLU();
 
  /**
   * reorder to standard CSR formaat : [L|D|U] from [D|L|U]
   */
  void reorderLDU();

  /**
   * get the host row_offsets ptr in page locked memory
   *
   * @return the pointer to the host row_offsets
   */
  int * getHostRowOffsetsPtr() { return _h_row_offsets; }

  /**
   * get the host column indices ptr in page locked memory
   *
   * @return the pointer to the host column indices 
   */
  IntType * getHostColIndicesPtr() { return _h_col_indices; }

  /**
   * get the host values ptr in page locked memory
   *
   * @return the pointer to the host values
   */
  double * getHostValuesPtr() { return _h_values; }

protected:

private:

  /* cuda timers */
  cudaEvent_t _start, _stop;
  float _assembleTime=0.f;
  float _xferTime=0.f;
  float _xferHostTime=0.f;
  int _nAssemble=0;
  
  /* amount of memory being used */
  IntType _memoryUsed=0;

  /* The final csr matrix pointers */
  IntType _num_nonzeros=0;
  int * _d_row_offsets=NULL;
  IntType * _d_col_indices=NULL;
  double *_d_values=NULL;

  int * _h_row_offsets=NULL;
  IntType * _h_col_indices=NULL;
  double *_h_values=NULL;

  /* meta data */
  std::string _name="";
  bool _sort=false;
  IntType _r0=0;
  IntType _c0=0;
  IntType _num_rows=0;
  IntType _num_cols=0;
  IntType _nDataPtsToAssemble=0;
  IntType _nBogusPtsToIgnore=0;

  /* Cuda pointers and allocations for temporaries */
  IntType * _d_rows=NULL, *_d_cols=NULL;
  double * _d_data=NULL, * _d_data_aux=NULL;
  IntType * _d_bin_ptrs=NULL;
  IntType * _d_key=NULL;
  IntType * _d_key_aux=NULL;
  IntType * _d_locations=NULL;
  int *_d_col_index_for_diagonal=NULL;
  bool _col_index_determined=false;
  int * _d_bin_block_count=NULL;
  IntType * _d_bin_ptrs_final = NULL;
  bool _csrMatMemoryAdded=false;
};



template<typename IntType>
class RhsAssembler {

public:

  /**
   * RhsAssembler Constructor 
   *
   * @param name of the linear system being assembled
   * @param sort whether or not to sort the CSR matrix (prior to full assembly) based on the element ids
   * @param r0 first row
   * @param num_rows number of rows
   * @param nDataPtsToAssemble the number of data points to assemble into a CSR matrix
   */
  RhsAssembler(std::string name, bool sort, IntType r0, IntType num_rows, IntType nDataPtsToAssemble);

  /**
   *  Destructor 
   */
  virtual ~RhsAssembler();

  /**
   * memoryInGBS computes the amount of device memory used in GBs
   *
   * @return the amount of device memory used in GBs
   */
  double memoryInGBs() const;

  /**
   * copySrcDataToDevice copies the Rhs source to the device
   *
   * @param rows host pointer for the row coordinates
   * @param data host pointer for the data values
   */
  void copySrcDataToDevice(const IntType * rows, const double * data);

  /**
   * copySrcDataFromKokkos copies the Rhs source data from Kokkos views
   *
   * @param rows host pointer for the row coordinates
   * @param data host pointer for the data values
   */
  void copySrcDataFromKokkos(const IntType * rows, const double * data);

  /**
   * copyAssembledRhsVectorToHost copies the assembled RhsVector to the host (page locked memory)
   */
  void copyAssembledRhsVectorToHost();

  /**
   * copyAssembledRhsVectorToHost copies the assembled RhsVector to the host
   *
   * @param rhs host pointer for the rhs values
   */
  void copyAssembledRhsVectorToHost(double * rhs);

  /**
   * assemble : assemble the symbolic and numeric parts of the CSR matrix
   *
   */
  void assemble();

  /**
   * get the host rhs ptr in page locked memory
   *
   * @return the pointer to the host rhs
   */
  double * getHostRhsPtr() { return _h_rhs; }

protected:

private:

  /* cuda timers */
  cudaEvent_t _start, _stop;
  float _assembleTime=0.f;
  float _xferTime=0.f;
  float _xferHostTime=0.f;
  int _nAssemble=0;

  /* amount of memory being used */
  IntType _memoryUsed=0;

  /* The final rhs vector */
  double *_d_rhs=NULL;
  double *_h_rhs=NULL;

  /* meta data */
  std::string _name="";
  bool _sort=false;
  IntType _r0=0;
  IntType _num_rows=0;
  IntType _nDataPtsToAssemble=0;
  IntType _nBogusPtsToIgnore=0;

  /* Cuda pointers and allocations for temporaries */
  IntType * _d_rows=NULL;
  IntType * _d_bin_ptrs=NULL;
  IntType * _d_locations=NULL;
  double * _d_data=NULL;
  double * _d_data_aux=NULL;
  int * _d_bin_block_count=NULL;
  IntType * _d_bin_ptrs_final = NULL;
};

}  // nalu
}  // sierra

#endif

#endif /* MATRIXASSEMBLER_H */
