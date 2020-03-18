#include "HypreLinearSystem.h"
#include "LinearSystemAssembler.h"

#ifdef KOKKOS_ENABLE_CUDA

namespace sierra {
namespace nalu {

#define MATRIX_ASSEMBLER_CUDA_SAFE_CALL(call) do {                                                             \
   cudaError_t err = call;                                                                                     \
   if (cudaSuccess != err) {                                                                                   \
      printf("CUDA ERROR (code = %d, %s) at %s:%d\n", err, cudaGetErrorString(err), __FILE__, __LINE__);       \
      exit(1);                                                                                                 \
   } } while(0)

int nextPowerOfTwo(int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

struct saxpy_functor
{
  const int64_t nc_;
  saxpy_functor(int64_t nc) : nc_(nc) {}
  __host__ __device__
  int64_t operator()(const int& x, const int& y) const
  { 
    return nc_ * ((int64_t)x) + ((int64_t)y);
  }
};

struct lessThanOrdering32
{
  lessThanOrdering32() {}
  __host__ __device__
  bool operator()(const thrust::tuple<int32_t,double>& x, const thrust::tuple<int32_t,double>& y) const
  { 
    int32_t x1 = thrust::get<0>(x);
    double x2 = thrust::get<1>(x);
    int32_t y1 = thrust::get<0>(y);
    double y2 = thrust::get<1>(y);
    if (x1<y1) return true;
    else if (x1>y1) return false;
    else {
      if (abs(x2)<abs(y2)) return true;
      else return false;
    }
  }
};

struct lessThanOrdering64
{
  lessThanOrdering64() {}
  __host__ __device__
  bool operator()(const thrust::tuple<int64_t,double>& x, const thrust::tuple<int64_t,double>& y) const
  { 
    HypreIntType x1 = thrust::get<0>(x);
    double x2 = thrust::get<1>(x);
    HypreIntType y1 = thrust::get<0>(y);
    double y2 = thrust::get<1>(y);
    if (x1<y1) return true;
    else if (x1>y1) return false;
    else {
      if (abs(x2)<abs(y2)) return true;
      else return false;
    }
  }
};

#define THREADBLOCK_SIZE 128

__global__ void diffKernel(const HypreIntType N, const HypreIntType * x, int * y) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid<N) y[tid] = (int)(x[tid+1] - x[tid]);
}

__global__ void shiftKernel(const HypreIntType N, const HypreIntType shift, const HypreIntType * x, HypreIntType * y) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid<N) {
    y[tid] = x[tid+1]-shift;
  }
}

__global__ void binPointersFinalKernel(const int N1, const HypreIntType N2, const HypreIntType * x, const HypreIntType * y, HypreIntType * z) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  HypreIntType xx = 0, yy=0;
  if (tid<=N2) {
    xx = x[tid];
    yy = y[tid];
    if (xx>0 && yy>0 && yy<=N1) z[yy] = xx;
  }
}

__global__ void binPointersKernel(const HypreIntType * x, const HypreIntType N, HypreIntType * y, HypreIntType * z, int * bin_block_count) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  
  /* load data into shmem */
  __shared__ HypreIntType shmem[THREADBLOCK_SIZE+1];
  int t = threadIdx.x;
  while (t<THREADBLOCK_SIZE+1 && blockIdx.x*blockDim.x+t<N) {
    shmem[t] = x[blockIdx.x*blockDim.x+t];
    t+=blockDim.x;
  }
  __syncthreads();

  /* Compute the differences and store. If on the very last entry, store N */
  if (tid<N-1) {
    y[tid+1] = (tid+1)*(shmem[threadIdx.x+1]>shmem[threadIdx.x]?1:0);
    z[tid+1] = shmem[threadIdx.x+1]>shmem[threadIdx.x]?1:0;
    if (shmem[threadIdx.x+1]>shmem[threadIdx.x]) atomicAdd(&(bin_block_count[blockIdx.x]),1);
  } else if (tid==N-1) {
    y[tid+1]=N;
    z[tid+1]=1;
    atomicAdd(&(bin_block_count[blockIdx.x]),1);
  }
}

__global__ void fillCSRMatrix(int num_rows, int threads_per_row,
			      const HypreIntType * bins_in, const HypreIntType * rows_in,
			      const HypreIntType *cols_in, double * data_in, 
			      int * rows_out, HypreIntType * cols_out, double * data_out) {
  
  int tid = threadIdx.x%threads_per_row;
  int rowSmall = threadIdx.x/threads_per_row;
  int row = (blockDim.x/threads_per_row)*blockIdx.x + rowSmall;

  if (row<num_rows) {
   /* read the row pointers by the first thread in the warp */
    HypreIntType begin, end;
    if (tid==0) {
      begin = bins_in[row];
      end = bins_in[row+1];
    }
    /* broadcast across the warp */
    begin = __shfl_sync(0xffffffff, begin, 0, threads_per_row);
    end = __shfl_sync(0xffffffff, end, 0, threads_per_row);
    
    /* This compute the loop size to the next multiple of threads_per_row */
    int roundUpSize = (((int)(end-begin) + threads_per_row - 1)/threads_per_row)*threads_per_row;
    double value=0.;
    double sum=0.;

    for (int t=tid; t<roundUpSize; t+=threads_per_row) {
      if (t>=end-begin) value=0.;
      else value = data_in[begin+t];
      for (int offset = threads_per_row/2; offset > 0; offset/=2)
      	value += __shfl_xor_sync(0xffffffff, value, offset, threads_per_row);      
      sum += value;
    }
    if (tid==0) {
      cols_out[row] = cols_in[begin];
      data_out[row] = sum;
      int * out = rows_out + rows_in[begin];
      atomicAdd(out,1);
    }
  }
}


__global__ void fillCSRMatrixData(int num_rows, int threads_per_row, 
				  const HypreIntType * bins_in, double * data_in, double * data_out) {
  
  int tid = threadIdx.x%threads_per_row;
  int rowSmall = threadIdx.x/threads_per_row;
  int row = (blockDim.x/threads_per_row)*blockIdx.x + rowSmall;

  if (row<num_rows) {
    /* read the row pointers by the first thread in the warp */
    HypreIntType begin, end;
    if (tid==0) {
      begin = bins_in[row];
      end = bins_in[row+1];
    }
    /* broadcast across the warp */
    begin = __shfl_sync(0xffffffff, begin, 0, threads_per_row);
    end = __shfl_sync(0xffffffff, end, 0, threads_per_row);
    
    /* This compute the loop size to the next multiple of threads_per_row */
    int roundUpSize = (((int)(end-begin) + threads_per_row - 1)/threads_per_row)*threads_per_row;
    double value=0.;
    double sum=0.;

    for (int t=tid; t<roundUpSize; t+=threads_per_row) {
      if (t>=end-begin) value=0.;
      else value = data_in[begin+t];
      for (int offset = threads_per_row/2; offset > 0; offset/=2)
      	value += __shfl_xor_sync(0xffffffff, value, offset, threads_per_row);      
      sum += value;
    }
    if (tid==0)
      data_out[row] = sum;
  }
}

__global__ void findDiagonalElementKernel(const int num_rows, const int threads_per_row, const int * rows, 
					  const HypreIntType * cols, int * colIndexForDiagonal) {
  int tid = threadIdx.x%threads_per_row;
  int rowSmall = threadIdx.x/threads_per_row;
  int row = (blockDim.x/threads_per_row)*blockIdx.x + rowSmall;

  if (row<num_rows) {
    /* read the row pointers by the first thread in the warp */
    int rbegin, rend;
    if (tid==0) {
      rbegin = rows[row];
      rend = rows[row+1];
    }

    /* broadcast across the warp */
    rbegin = __shfl_sync(0xffffffff, rbegin, 0, threads_per_row);
    rend = __shfl_sync(0xffffffff, rend, 0, threads_per_row);

    /* This compute the loop size to the next multiple of threads_per_row */
    int roundUpSize = ((rend-rbegin + threads_per_row - 1)/threads_per_row)*threads_per_row;
    
    int colIndexForDiag=0;
    for (int t=tid; t<roundUpSize; t+=threads_per_row) {
      /* make a value for large threads that is guaranteed to be bigger than all others */
      int column = 2*num_rows;

      /* read the actual column for valid threads/columns */
      if (t<rend-rbegin) column = cols[rbegin+t];
      /* makt it absolute value so we can search for 0 */
      int val = abs(column-row);

      /* Try to find the location of the diagonal */
      colIndexForDiag = t;
      for (int offset = threads_per_row/2; offset > 0; offset/=2) {
      	int tmp1 = __shfl_down_sync(0xffffffff, val, offset, threads_per_row);
      	int tmp2 = __shfl_down_sync(0xffffffff, colIndexForDiag, offset, threads_per_row);
      	if (tmp1 < val) {
      	  val = tmp1;
      	  colIndexForDiag = tmp2;
      	}
      }
      /* broadcast in order to exit successfully for all threads in the warp */
      val = __shfl_sync(0xffffffff, val, 0, threads_per_row);
      if (val==0) break;
    }

    if (tid==0) {
      colIndexForDiagonal[row] = colIndexForDiag;
    }
  }
}

__global__ void shuffleDiagonalDLUKernel(const int num_rows, const int threads_per_row, const int * rows, 
					 const int * colIndexForDiagonal, HypreIntType * cols, double * values) {
  int tid = threadIdx.x%threads_per_row;
  int rowSmall = threadIdx.x/threads_per_row;
  int row = (blockDim.x/threads_per_row)*blockIdx.x + rowSmall;

  if (row<num_rows) {
    /* read the row pointers by the first thread in the warp */
    int rbegin, rend, colIndex;
    if (tid==0) {
      rbegin = rows[row];
      rend = rows[row+1];
      colIndex = colIndexForDiagonal[row];
    }

    /* broadcast across the warp */
    rbegin = __shfl_sync(0xffffffff, rbegin, 0, threads_per_row);
    rend = __shfl_sync(0xffffffff, rend, 0, threads_per_row);
    colIndex = __shfl_sync(0xffffffff, colIndex, 0, threads_per_row);

    if (colIndex>0) {

      HypreIntType diag_column;
      double diag_value;
      if (tid==0) {
	diag_column = cols[rbegin+colIndex];
	diag_value = values[rbegin+colIndex];
      }

      /* This compute the loop size to the next multiple of threads_per_row */
      int roundUpSize = ((colIndex + threads_per_row - 1)/threads_per_row)*threads_per_row;

      int column;
      double value;
      for (int t=tid; t<roundUpSize; t+=threads_per_row) {
	int t1 = colIndex-1-t;
	if (t1>=0) {
	  value = values[rbegin+t1];
	  column = cols[rbegin+t1];
	  values[rbegin+t1+1] = value;
	  cols[rbegin+t1+1] = column;
	}    
      }
      /* write the column to the front of the row */
      if (tid==0) {
	values[rbegin] = diag_value;
	cols[rbegin] = diag_column;
      }
    }
  }
}


__global__ void shuffleDiagonalLDUKernel(const int num_rows, const int threads_per_row, const int * rows, 
					 const int * colIndexForDiagonal, HypreIntType * cols, double * values) {
  int tid = threadIdx.x%threads_per_row;
  int rowSmall = threadIdx.x/threads_per_row;
  int row = (blockDim.x/threads_per_row)*blockIdx.x + rowSmall;

  if (row<num_rows) {
    /* read the row pointers by the first thread in the warp */
    int rbegin, rend, colIndex;
    if (tid==0) {
      rbegin = rows[row];
      rend = rows[row+1];
      colIndex = colIndexForDiagonal[row];
    }

    /* broadcast across the warp */
    rbegin = __shfl_sync(0xffffffff, rbegin, 0, threads_per_row);
    rend = __shfl_sync(0xffffffff, rend, 0, threads_per_row);
    colIndex = __shfl_sync(0xffffffff, colIndex, 0, threads_per_row);

    if (colIndex>0) {

      HypreIntType diag_column;
      double diag_value;
      if (tid==0) {
	diag_column = cols[rbegin];
	diag_value = values[rbegin];
      }

      /* This compute the loop size to the next multiple of threads_per_row */
      int roundUpSize = ((colIndex + threads_per_row - 1)/threads_per_row)*threads_per_row;

      int column;
      double value;
      for (int t=tid; t<roundUpSize; t+=threads_per_row) {
	if (t<colIndex) {
	  value = values[rbegin+t+1];
	  column = cols[rbegin+t+1];
	  values[rbegin+t] = value;
	  cols[rbegin+t] = column;
	}    
      }
      /* write the column to the front of the row */
      if (tid==0) {
	values[rbegin+colIndex] = diag_value;
	cols[rbegin+colIndex] = diag_column;
      }
    }
  }
}

template<typename IntType>
void sortCoo(thrust::device_ptr<IntType> _d_key_ptr,
	     thrust::device_ptr<IntType> _d_key_ptr_end,
             thrust::device_ptr<IntType> _d_rows_ptr,
	     thrust::device_ptr<IntType> _d_cols_ptr,
             thrust::device_ptr<double> _d_data_ptr);

template<>
void sortCoo(thrust::device_ptr<HypreIntType> _d_key_ptr,
             thrust::device_ptr<HypreIntType> _d_key_ptr_end,
             thrust::device_ptr<HypreIntType> _d_rows_ptr,
             thrust::device_ptr<HypreIntType> _d_cols_ptr,
             thrust::device_ptr<double> _d_data_ptr) {

    typedef thrust::device_vector<HypreIntType>::iterator IntIterator;
    typedef thrust::device_vector<double>::iterator DoubleIterator;
    typedef thrust::tuple<IntIterator, IntIterator, DoubleIterator> IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
    ZipIterator iter(thrust::make_tuple(_d_rows_ptr, _d_cols_ptr, _d_data_ptr));
    thrust::stable_sort_by_key(thrust::device, _d_key_ptr, _d_key_ptr_end, iter);
}

template<typename IntType>
void sortRhs(thrust::device_ptr<IntType> _d_key_ptr,
             thrust::device_ptr<IntType> _d_key_ptr_end,
             thrust::device_ptr<double> _d_dkey_ptr,
             thrust::device_ptr<double> _d_dkey_ptr_end,
             thrust::device_ptr<double> _d_data_ptr);

template<>
void sortRhs(thrust::device_ptr<HypreIntType> _d_key_ptr,
             thrust::device_ptr<HypreIntType> _d_key_ptr_end,
             thrust::device_ptr<double> _d_dkey_ptr,
             thrust::device_ptr<double> _d_dkey_ptr_end,
             thrust::device_ptr<double> _d_data_ptr) {

    typedef thrust::device_vector<HypreIntType>::iterator IntIterator;
    typedef thrust::device_vector<double>::iterator DoubleIterator;
    typedef thrust::tuple<IntIterator, DoubleIterator> IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
    ZipIterator iter_begin(thrust::make_tuple(_d_key_ptr, _d_dkey_ptr));
    ZipIterator iter_end(thrust::make_tuple(_d_key_ptr_end, _d_dkey_ptr_end));
    thrust::stable_sort_by_key(thrust::device, iter_begin, iter_end, _d_data_ptr, lessThanOrdering64());
}


template<typename IntType>
void sortCooAscending(thrust::device_ptr<IntType> _d_key_ptr,
                      thrust::device_ptr<IntType> _d_key_ptr_end,
                      thrust::device_ptr<double> _d_dkey_ptr,
                      thrust::device_ptr<double> _d_dkey_ptr_end,
                      thrust::device_ptr<IntType> _d_rows_ptr,
                      thrust::device_ptr<IntType> _d_cols_ptr,
                      thrust::device_ptr<double> _d_data_ptr);

template<>
void sortCooAscending(thrust::device_ptr<HypreIntType> _d_key_ptr,
                      thrust::device_ptr<HypreIntType> _d_key_ptr_end,
                      thrust::device_ptr<double> _d_dkey_ptr,
                      thrust::device_ptr<double> _d_dkey_ptr_end,
                      thrust::device_ptr<HypreIntType> _d_rows_ptr,
                      thrust::device_ptr<HypreIntType> _d_cols_ptr,
                      thrust::device_ptr<double> _d_data_ptr) {

    typedef thrust::device_vector<HypreIntType>::iterator IntIterator;
    typedef thrust::device_vector<double>::iterator DoubleIterator;
    typedef thrust::tuple<IntIterator, DoubleIterator> IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
    typedef thrust::tuple<IntIterator, IntIterator, DoubleIterator> IteratorTuple3;
    typedef thrust::zip_iterator<IteratorTuple3> ZipIterator3;
    ZipIterator iter_begin(thrust::make_tuple(_d_key_ptr, _d_dkey_ptr));
    ZipIterator iter_end(thrust::make_tuple(_d_key_ptr_end, _d_dkey_ptr_end));
    ZipIterator3 iter3_begin(thrust::make_tuple(_d_rows_ptr, _d_cols_ptr, _d_data_ptr));
    thrust::stable_sort_by_key(thrust::device, iter_begin, iter_end, iter3_begin, lessThanOrdering64());
}


template<typename IntType>
MatrixAssembler<IntType>::MatrixAssembler(std::string name, bool sort, IntType r0, IntType c0,
					  IntType num_rows, IntType num_cols, IntType nDataPtsToAssemble)
  : _name(name), _sort(sort), _r0(r0), _c0(c0), _num_rows(num_rows), _num_cols(num_cols), 
    _nDataPtsToAssemble(nDataPtsToAssemble)
{
#ifdef LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("\n%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif

  /* allocate some space */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_rows, _nDataPtsToAssemble*sizeof(IntType)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_cols, _nDataPtsToAssemble*sizeof(IntType)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_key, _nDataPtsToAssemble*sizeof(IntType)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_key_aux, _nDataPtsToAssemble*sizeof(IntType)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_bin_ptrs, (_nDataPtsToAssemble+1)*sizeof(IntType)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_locations, (_nDataPtsToAssemble+1)*sizeof(IntType)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_data, _nDataPtsToAssemble*sizeof(double)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_data_aux, _nDataPtsToAssemble*sizeof(double)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_row_offsets, (_num_rows+1)*sizeof(int)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_col_index_for_diagonal, _num_rows*sizeof(int)));
  int num_threads=128;
  int num_blocks = (_nDataPtsToAssemble + num_threads - 1)/num_threads;  
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void**)&_d_bin_block_count, (num_blocks+1)*sizeof(int)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_bin_ptrs_final, (_nDataPtsToAssemble+1)*sizeof(IntType)));

  /* Host Assemblies */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_row_offsets, (_num_rows+1)*sizeof(int)));

  _memoryUsed = 7*sizeof(IntType)*_nDataPtsToAssemble + 2*sizeof(double)*_nDataPtsToAssemble 
    + 3*sizeof(IntType) + (2*_num_rows+1)*sizeof(int) + (num_blocks+1)*sizeof(int);
  
  /* create events */
  cudaEventCreate(&_start);
  cudaEventCreate(&_stop);
  _assembleTime=0.f;
  _nAssemble=0;
  _xferTime=0.f;

#ifdef LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("Done %s %s %d : name=%s : nDataPtsToAssemble=%lld, Device Memory GBs=%1.6lf\n",
	 __FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,memoryInGBs());
#endif
}

template<typename IntType>
MatrixAssembler<IntType>::~MatrixAssembler() {
#ifdef LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("\n%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,_name.c_str());
  if (_nAssemble>0)
    printf("Mean Symbolic/Numeric Assembly Time (%d samples)=%1.5f msec, Data Xfer Time From Kokkos=%1.5f msec, Data Xfer Time To Host=%1.5f msec\n",
	   _nAssemble,_assembleTime/_nAssemble,_xferTime/_nAssemble,_xferHostTime/_nAssemble);
#endif
  
  /* free the data */
  if (_d_rows) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_rows)); _d_rows=NULL; }
  if (_d_cols) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_cols)); _d_cols=NULL; }
  if (_d_key) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_key)); _d_key=NULL; }
  if (_d_key_aux) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_key_aux)); _d_key_aux=NULL; }
  if (_d_bin_ptrs) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_bin_ptrs)); _d_bin_ptrs=NULL; }
  if (_d_locations) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_locations)); _d_locations=NULL; }
  if (_d_data) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_data)); _d_data=NULL; }
  if (_d_data_aux) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_data_aux)); _d_data_aux=NULL; }
  if (_d_col_index_for_diagonal) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_col_index_for_diagonal)); _d_col_index_for_diagonal=NULL; }
  if (_d_bin_block_count) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_bin_block_count)); _d_bin_block_count=NULL; }
  if (_d_bin_ptrs_final) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_bin_ptrs_final)); _d_bin_ptrs_final=NULL; }

  /* csr matrix */
  if (_d_row_offsets) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_row_offsets)); _d_row_offsets=NULL; }
  if (_d_col_indices) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_col_indices)); _d_col_indices=NULL; }
  if (_d_values) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_values)); _d_values=NULL; }

  if (_h_row_offsets) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_row_offsets)); _h_row_offsets=NULL; }
  if (_h_col_indices) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_col_indices)); _h_col_indices=NULL; }
  if (_h_values) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_values)); _h_values=NULL; }

  /* create events */
  cudaEventDestroy(_start);
  cudaEventDestroy(_stop);
}

template<typename IntType>
double MatrixAssembler<IntType>::memoryInGBs() const {
  return 1.0*_memoryUsed/(1024.*1024.*1024.);
}

template<typename IntType>
IntType MatrixAssembler<IntType>::getNumNonzeros() const {
  return _num_nonzeros;
}

template<typename IntType>
void MatrixAssembler<IntType>::copySrcDataToDevice(const IntType * rows, const IntType * cols, const double * data) {
  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_rows, rows, _nDataPtsToAssemble*sizeof(IntType), cudaMemcpyHostToDevice));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_cols, cols, _nDataPtsToAssemble*sizeof(IntType), cudaMemcpyHostToDevice));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_data, data, _nDataPtsToAssemble*sizeof(double), cudaMemcpyHostToDevice));
  
  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferTime+=t;
}

template<typename IntType>
void MatrixAssembler<IntType>::copySrcDataFromKokkos(const IntType * rows, const IntType * cols, const double * data) {
  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_rows, rows, _nDataPtsToAssemble*sizeof(IntType), cudaMemcpyDeviceToDevice));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_cols, cols, _nDataPtsToAssemble*sizeof(IntType), cudaMemcpyDeviceToDevice));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_data, data, _nDataPtsToAssemble*sizeof(double), cudaMemcpyDeviceToDevice));
  
  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferTime+=t;
}


template<typename IntType>
void MatrixAssembler<IntType>::copyAssembledCSRMatrixToHost() {
  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_row_offsets, _d_row_offsets, (_num_rows+1)*sizeof(int), cudaMemcpyDeviceToHost));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_col_indices, _d_col_indices, _num_nonzeros*sizeof(IntType), cudaMemcpyDeviceToHost));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_values, _d_values, _num_nonzeros*sizeof(double), cudaMemcpyDeviceToHost));

  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
}

template<typename IntType>
void MatrixAssembler<IntType>::copyAssembledCSRMatrixToHost(int * rows, IntType * cols, double * data) {
  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(rows, _d_row_offsets, (_num_rows+1)*sizeof(int), cudaMemcpyDeviceToHost));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(cols, _d_col_indices, _num_nonzeros*sizeof(IntType), cudaMemcpyDeviceToHost));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(data, _d_values, _num_nonzeros*sizeof(double), cudaMemcpyDeviceToHost));

  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
}


template<typename IntType>
void MatrixAssembler<IntType>::assemble() {

  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  /* reset */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_bin_ptrs, 0, (_nDataPtsToAssemble+1)*sizeof(IntType)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_locations, 0, (_nDataPtsToAssemble+1)*sizeof(IntType)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_row_offsets, 0, (_num_rows+1)*sizeof(int)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_bin_ptrs_final, 0, (_nDataPtsToAssemble+1)*sizeof(IntType)));

  /* thrust pointers ... useful to define up front */
  thrust::device_ptr<IntType> _d_rows_ptr = thrust::device_pointer_cast(_d_rows);
  thrust::device_ptr<IntType> _d_rows_ptr_end = thrust::device_pointer_cast(_d_rows + _nDataPtsToAssemble);
  thrust::device_ptr<IntType> _d_cols_ptr = thrust::device_pointer_cast(_d_cols);
  thrust::device_ptr<IntType> _d_key_ptr = thrust::device_pointer_cast(_d_key);
  thrust::device_ptr<IntType> _d_key_ptr_end = thrust::device_pointer_cast(_d_key + _nDataPtsToAssemble);
  thrust::device_ptr<double> _d_data_ptr = thrust::device_pointer_cast(_d_data);

  /* Step 1 : compute the dense index */
  thrust::transform(thrust::device, _d_rows_ptr, _d_rows_ptr_end, _d_cols_ptr, _d_key_ptr, saxpy_functor(_num_cols));
  
  /* Step 2 : do a stable sort by key on the dense index. Apply the sorting to the 4pt tuple defined next */
  if (_sort) {
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_data_aux, _d_data, _nDataPtsToAssemble*sizeof(double), cudaMemcpyDeviceToDevice));
    thrust::device_ptr<double> _d_data_aux_ptr     = thrust::device_pointer_cast(_d_data_aux);
    thrust::device_ptr<double> _d_data_aux_ptr_end = thrust::device_pointer_cast(_d_data_aux + _nDataPtsToAssemble);
    sortCooAscending<IntType>(_d_key_ptr, _d_key_ptr_end, _d_data_aux_ptr, _d_data_aux_ptr_end, _d_rows_ptr, _d_cols_ptr, _d_data_ptr);
  } else {
    sortCoo<IntType>(_d_key_ptr, _d_key_ptr_end, _d_rows_ptr, _d_cols_ptr, _d_data_ptr);
  }

  /* Step 3 : Create the bin_ptrs vector by looking at differences between the key_sorted vector */

  /* this choice has to be the same as what's in the construtor. Do not change unless you know what you're doing */
  int num_threads=128;
  int num_blocks = (_nDataPtsToAssemble + num_threads - 1)/num_threads;    
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_bin_block_count, 0, (num_blocks+1)*sizeof(int)));
  binPointersKernel<<<num_blocks,num_threads>>>(_d_key, _nDataPtsToAssemble, _d_bin_ptrs, _d_locations, _d_bin_block_count);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
    
  /* Step 4 : exclusive scan on the block count gives the relative positions of where to write the row pointers */
  thrust::inclusive_scan(thrust::device, 
			 thrust::device_pointer_cast(_d_locations),
			 thrust::device_pointer_cast(_d_locations+_nDataPtsToAssemble+1),
			 thrust::device_pointer_cast(_d_locations));

  /* Step 5: reduce to get the count, i.e. the number of nonzeros */
  _num_nonzeros = thrust::reduce(thrust::device, 
				 thrust::device_pointer_cast(_d_bin_block_count),
				 thrust::device_pointer_cast(_d_bin_block_count+num_blocks+1));

#ifdef LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("%s %s %d : name=%s : _num_nonzeros=%lld\n",__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_num_nonzeros);
#endif

  /* Step 6 : Compute the final row pointers array */
  num_blocks = (_nDataPtsToAssemble + 1 + num_threads - 1)/num_threads;  
  binPointersFinalKernel<<<num_blocks,num_threads>>>(_num_nonzeros, _nDataPtsToAssemble,
						     _d_bin_ptrs, _d_locations, _d_bin_ptrs_final);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  

  /* Step 7 : check for bogus indices */
  IntType key;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&key, _d_key, sizeof(IntType), cudaMemcpyDeviceToHost));
  if (key<0) {
    IntType firstValidIndex[2];
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(firstValidIndex, _d_bin_ptrs_final, 2*sizeof(IntType), cudaMemcpyDeviceToHost));
    _nBogusPtsToIgnore = (firstValidIndex[1]-firstValidIndex[0]);
    int numBlocks = (_num_nonzeros + num_threads - 1)/num_threads;
    shiftKernel<<<numBlocks,num_threads>>>(_num_nonzeros, _nBogusPtsToIgnore, _d_bin_ptrs_final, _d_bin_ptrs);
    
    _num_nonzeros -= 1;
#ifdef LINEAR_SYSTEM_ASSEMBLER_DEBUG
    printf("%s %s %d : name=%s : _num_nonzeros=%lld, _nBogusPtsToIgnore=%lld\n",
	   __FILE__,__FUNCTION__,__LINE__,_name.c_str(),_num_nonzeros,_nBogusPtsToIgnore);
#endif
  } else {
    /* copy the temporary over to the permanent */
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_bin_ptrs, _d_bin_ptrs_final, (_num_nonzeros+1)*sizeof(IntType), cudaMemcpyDeviceToDevice));
  }

  /* Allocate space for the CSR matrix */
  if (!_d_col_indices) {
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_col_indices, _num_nonzeros*sizeof(IntType)));
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_col_indices, _num_nonzeros*sizeof(IntType)));
  }
  if (!_d_values) {
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_values, _num_nonzeros*sizeof(double)));
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_values, _num_nonzeros*sizeof(double)));
  }
  /* accumulate */
  if (!_csrMatMemoryAdded) {
    _memoryUsed += (_num_nonzeros)*(sizeof(IntType) + sizeof(double));
    _csrMatMemoryAdded = true;
  }

#ifdef LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("%s %s %d : name=%s : Device Memory GBs=%1.6lf\n",__FILE__,__FUNCTION__,__LINE__,_name.c_str(),memoryInGBs());
#endif

  /* Step 8 : reduce the array and create the "True" CSR matrix */
  num_threads=128;
  int threads_per_row = (_nDataPtsToAssemble - _nBogusPtsToIgnore + _num_nonzeros - 1)/_num_nonzeros;
  threads_per_row = nextPowerOfTwo(threads_per_row);
  int num_rows_per_block = num_threads/threads_per_row;
  num_blocks = (_num_nonzeros + num_rows_per_block - 1)/num_rows_per_block;

  /* fill the matrix */
  fillCSRMatrix<<<num_blocks,num_threads>>>(_num_nonzeros, threads_per_row,
					    _d_bin_ptrs,
					    _d_rows + _nBogusPtsToIgnore,
					    _d_cols + _nBogusPtsToIgnore,
					    _d_data + _nBogusPtsToIgnore, 
					    _d_row_offsets, _d_col_indices, _d_values);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  

  /* step 9: final exclusive scan on the row offsets gives the true row_offsets vector */
  thrust::exclusive_scan(thrust::device, 
			 thrust::device_pointer_cast(_d_row_offsets),
			 thrust::device_pointer_cast(_d_row_offsets+_num_rows+1),
			 thrust::device_pointer_cast(_d_row_offsets));

  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _assembleTime+=t;
  _nAssemble++;
}


template<typename IntType>
void MatrixAssembler<IntType>::reorderDLU() {

  int num_threads=128;
  int threads_per_row = (_num_nonzeros + _num_rows - 1)/_num_rows;
  threads_per_row = std::min(nextPowerOfTwo(threads_per_row),32);    
  int num_rows_per_block = num_threads/threads_per_row;
  int num_blocks = (_num_rows + num_rows_per_block - 1)/num_rows_per_block;
  
  /* compute the location of the diagonal in each row */
  findDiagonalElementKernel<<<num_blocks,num_threads>>>(_num_rows, threads_per_row, _d_row_offsets, 
							_d_col_indices, _d_col_index_for_diagonal);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
  _col_index_determined = true;
  
  /* shuffle the rows to put the diagonal first */
  shuffleDiagonalDLUKernel<<<num_blocks,num_threads>>>(_num_rows, threads_per_row, _d_row_offsets, 
						       _d_col_index_for_diagonal, _d_col_indices, _d_values);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
}

template<typename IntType>
void MatrixAssembler<IntType>::reorderLDU() {
  if (!_col_index_determined) {
    printf("column index for diagonal not computed. Must call reorderDLU first\n");
    return;
  }
  int num_threads=128;
  int threads_per_row = (_num_nonzeros + _num_rows - 1)/_num_rows;
  threads_per_row = std::min(nextPowerOfTwo(threads_per_row),32);    
  int num_rows_per_block = num_threads/threads_per_row;
  int num_blocks = (_num_rows + num_rows_per_block - 1)/num_rows_per_block;
  shuffleDiagonalLDUKernel<<<num_blocks,num_threads>>>(_num_rows, threads_per_row, _d_row_offsets, 
						       _d_col_index_for_diagonal, _d_col_indices, _d_values);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
}

//////////////////////////////////////////////////////////////////
// Explicit template instantiation
template class MatrixAssembler<HypreIntType>;




template<typename IntType>
RhsAssembler<IntType>::RhsAssembler(std::string name, bool sort, IntType r0, IntType num_rows, IntType nDataPtsToAssemble)
  : _name(name), _sort(sort), _r0(r0), _num_rows(num_rows), _nDataPtsToAssemble(nDataPtsToAssemble)
{
#ifdef LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("\n%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,_name.c_str());
#endif
  
  /* allocate some space */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_rows, _nDataPtsToAssemble*sizeof(IntType)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_bin_ptrs, (_nDataPtsToAssemble+1)*sizeof(IntType)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_locations, (_nDataPtsToAssemble+1)*sizeof(IntType)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_data, _nDataPtsToAssemble*sizeof(double)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_data_aux, _nDataPtsToAssemble*sizeof(double)));
  int num_threads=128;
  int num_blocks = (_nDataPtsToAssemble + num_threads - 1)/num_threads;  
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void**)&_d_bin_block_count, (num_blocks+1)*sizeof(int)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_bin_ptrs_final, (_nDataPtsToAssemble+1)*sizeof(IntType)));

  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMalloc((void **)&_d_rhs, _num_rows*sizeof(double)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMallocHost((void **)&_h_rhs, _num_rows*sizeof(double)));
  
  _memoryUsed = 4*sizeof(IntType)*_nDataPtsToAssemble + 2*sizeof(double)*_nDataPtsToAssemble + 3*sizeof(IntType) + sizeof(double)*_num_rows + (num_blocks+1)*sizeof(int);
  
  
  /* create events */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventCreate(&_start));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventCreate(&_stop));
  _assembleTime=0.f;
  _xferTime=0.f;

#ifdef LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("Done %s %s %d : name=%s : nDataPtsToAssemble=%lld, Device Memory GBs=%1.6lf\n",
	 __FILE__,__FUNCTION__,__LINE__,_name.c_str(),_nDataPtsToAssemble,memoryInGBs());
#endif
}

template<typename IntType>
RhsAssembler<IntType>::~RhsAssembler() {
#ifdef LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("\n%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,_name.c_str());
  if (_nAssemble>0)
    printf("Mean RHS Assembly Time (%d samples)=%1.5f msec, Data Xfer Time From Kokkos=%1.5f msec, Data Xfer Time To Host=%1.5f msec\n",
	   _nAssemble,_assembleTime/_nAssemble,_xferTime/_nAssemble,_xferHostTime/_nAssemble);
#endif

  /* free the data */
  if (_d_rows) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_rows)); _d_rows=NULL; }
  if (_d_bin_ptrs) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_bin_ptrs)); _d_bin_ptrs=NULL; }
  if (_d_locations) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_locations)); _d_locations=NULL; }
  if (_d_data) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_data)); _d_data=NULL; }
  if (_d_data_aux) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_data_aux)); _d_data_aux=NULL; }
  if (_d_bin_block_count) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_bin_block_count)); _d_bin_block_count=NULL; }
  if (_d_bin_ptrs_final) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_bin_ptrs_final)); _d_bin_ptrs_final=NULL; }

  if (_d_rhs) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFree(_d_rhs)); _d_rhs=NULL; }
  if (_h_rhs) { MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaFreeHost(_h_rhs)); _d_rhs=NULL; }
  
  /* create events */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventDestroy(_start));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventDestroy(_stop));
}

template<typename IntType>
double RhsAssembler<IntType>::memoryInGBs() const {
  return 1.0*_memoryUsed/(1024.*1024.*1024.);
}

template<typename IntType>
void RhsAssembler<IntType>::copySrcDataToDevice(const IntType * rows, const double * data) {

  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));
  
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_rows, rows, _nDataPtsToAssemble*sizeof(IntType), cudaMemcpyHostToDevice));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_data, data, _nDataPtsToAssemble*sizeof(double), cudaMemcpyHostToDevice));
  
  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferTime+=t;
}

template<typename IntType>
void RhsAssembler<IntType>::copySrcDataFromKokkos(const IntType * rows, const double * data) {
  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));
  
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_rows, rows, _nDataPtsToAssemble*sizeof(IntType), cudaMemcpyDeviceToDevice));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_data, data, _nDataPtsToAssemble*sizeof(double), cudaMemcpyDeviceToDevice));
  
  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferTime+=t;
}

template<typename IntType>
void RhsAssembler<IntType>::copyAssembledRhsVectorToHost() {
  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));
  
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_h_rhs, _d_rhs, _num_rows*sizeof(double), cudaMemcpyDeviceToHost));
  
  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
}

template<typename IntType>
void RhsAssembler<IntType>::copyAssembledRhsVectorToHost(double * rhs) {
  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));
  
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(rhs, _d_rhs, _num_rows*sizeof(double), cudaMemcpyDeviceToHost));
  
  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _xferHostTime+=t;
}

template<typename IntType>
void RhsAssembler<IntType>::assemble() {

  /* record the start time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_start));

  /* reset */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_bin_ptrs, 0, (_nDataPtsToAssemble+1)*sizeof(IntType)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_locations, 0, (_nDataPtsToAssemble+1)*sizeof(IntType)));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_bin_ptrs_final, 0, (_nDataPtsToAssemble+1)*sizeof(IntType)));

  /* thrust pointers ... useful to define up front */
  thrust::device_ptr<IntType> _d_rows_ptr = thrust::device_pointer_cast(_d_rows);
  thrust::device_ptr<IntType> _d_rows_ptr_end = thrust::device_pointer_cast(_d_rows + _nDataPtsToAssemble);
  thrust::device_ptr<double> _d_data_ptr = thrust::device_pointer_cast(_d_data);
  thrust::device_ptr<double> _d_data_aux_ptr = thrust::device_pointer_cast(_d_data_aux);
  thrust::device_ptr<double> _d_data_aux_ptr_end = thrust::device_pointer_cast(_d_data_aux + _nDataPtsToAssemble);
  
  /* Step 1 : do a stable sort by key */
  if (_sort) {
    /* here we sort on the tuple row,data pair. */
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_data_aux, _d_data, _nDataPtsToAssemble*sizeof(double), cudaMemcpyDeviceToDevice));      
    sortRhs<IntType>(_d_rows_ptr, _d_rows_ptr_end, _d_data_aux_ptr, _d_data_aux_ptr_end, _d_data_ptr);
  } else {
    /* here we only sort on the row */
    thrust::stable_sort_by_key(thrust::device, _d_rows_ptr, _d_rows_ptr_end, _d_data);
  }
  
  /* Step 2 : Create the bin_ptrs vector by looking at differences between the key_sorted vector */

  /* this choice has to be the same as what's in the construtor. Do not change unless you know what you're doing */
  int num_threads=128;
  int num_blocks = (_nDataPtsToAssemble + num_threads - 1)/num_threads;  
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemset(_d_bin_block_count, 0, (num_blocks+1)*sizeof(int)));
  binPointersKernel<<<num_blocks,num_threads>>>(_d_rows, _nDataPtsToAssemble, _d_bin_ptrs, _d_locations, _d_bin_block_count);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
  
  /* Step 3 : exclusive scan on the block count gives the relative positions of where to write the row pointers */
  thrust::inclusive_scan(thrust::device, 
			 thrust::device_pointer_cast(_d_locations),
			 thrust::device_pointer_cast(_d_locations+_nDataPtsToAssemble+1),
			 thrust::device_pointer_cast(_d_locations));
  
  /* Step 4: reduce to get the count, i.e. the number of nonzeros */
  IntType num_rows_computed = thrust::reduce(thrust::device, 
					     thrust::device_pointer_cast(_d_bin_block_count),
					     thrust::device_pointer_cast(_d_bin_block_count+num_blocks+1));
#ifdef LINEAR_SYSTEM_ASSEMBLER_DEBUG
  printf("%s %s %d : name=%s : num rows=%lld, computed num rows=%lld\n",__FILE__,__FUNCTION__,__LINE__,_name.c_str(),_num_rows,num_rows_computed);
#endif
  
  /* Step 5 : Compute the final row pointers array */
  num_blocks = (_nDataPtsToAssemble + 1 + num_threads - 1)/num_threads;  
  binPointersFinalKernel<<<num_blocks,num_threads>>>(num_rows_computed, _nDataPtsToAssemble, _d_bin_ptrs,
						     _d_locations, _d_bin_ptrs_final);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());  
  
  /* check for bogus indices */
  IntType key;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(&key, _d_rows, sizeof(IntType), cudaMemcpyDeviceToHost));
  if (key<0) {
    IntType firstValidIndex[2];
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(firstValidIndex, _d_bin_ptrs_final, 2*sizeof(IntType), cudaMemcpyDeviceToHost));
    _nBogusPtsToIgnore = (firstValidIndex[1]-firstValidIndex[0]);
    int numBlocks = (num_rows_computed + num_threads - 1)/num_threads;
    shiftKernel<<<numBlocks,num_threads>>>(num_rows_computed, _nBogusPtsToIgnore, _d_bin_ptrs_final, _d_bin_ptrs);
    
#ifdef LINEAR_SYSTEM_ASSEMBLER_DEBUG
    printf("%s %s %d : name=%s : _num_rows=%lld, _nBogusPtsToIgnore=%lld\n",__FILE__,__FUNCTION__,__LINE__,_name.c_str(),
	   _num_rows,_nBogusPtsToIgnore);
#endif
  } else {
    /* copy the temporary over to the permanent */
    MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaMemcpy(_d_bin_ptrs, _d_bin_ptrs_final, (_num_rows+1)*sizeof(IntType), cudaMemcpyDeviceToDevice));
  }
  
  /* Step 6 : reduce the array and create the RHS Vector */
  num_threads=128;
  //int threads_per_row = (_nDataPtsToAssemble + _num_rows - 1)/_num_rows;
  int threads_per_row = 1;
  //threads_per_row = nextPowerOfTwo(threads_per_row);    
  int num_rows_per_block = num_threads/threads_per_row;
  num_blocks = (_num_rows + num_rows_per_block - 1)/num_rows_per_block;

  fillCSRMatrixData<<<num_blocks,num_threads>>>(_num_rows, threads_per_row,
						_d_bin_ptrs, _d_data + _nBogusPtsToIgnore,
						_d_rhs);
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaGetLastError());
  
  /* record the stop time */
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventRecord(_stop));
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventSynchronize(_stop));
  float t=0;
  MATRIX_ASSEMBLER_CUDA_SAFE_CALL(cudaEventElapsedTime(&t, _start, _stop));
  _assembleTime+=t;
  _nAssemble++;
}

//////////////////////////////////////////////////////////////////
// Explicit template instantiation
template class RhsAssembler<HypreIntType>;


}  // nalu
}  // sierra

#endif
