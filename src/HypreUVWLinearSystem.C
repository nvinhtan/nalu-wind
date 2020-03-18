// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "HypreUVWLinearSystem.h"
#include "HypreUVWSolver.h"
#include "NaluEnv.h"
#include "Realm.h"
#include "EquationSystem.h"

#include <utils/CreateDeviceExpression.h>

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_util/parallel/ParallelReduce.hpp"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE.h"
#include "HYPRE_config.h"

#include <limits>
#include <vector>
#include <string>
#include <cmath>

namespace sierra {
namespace nalu {

HypreUVWLinearSystem::HypreUVWLinearSystem(
  Realm& realm,
  const unsigned numDof,
  EquationSystem* eqSys,
  LinearSolver* linearSolver
) : HypreLinearSystem(realm, 1, eqSys, linearSolver),
    rhs_(numDof, nullptr),
    sln_(numDof, nullptr),
    nDim_(numDof)
{}

HypreUVWLinearSystem::~HypreUVWLinearSystem()
{
  if (systemInitialized_) {
    HYPRE_IJMatrixDestroy(mat_);

    for (unsigned i=0; i<nDim_; ++i) {
      HYPRE_IJVectorDestroy(rhs_[i]);
      HYPRE_IJVectorDestroy(sln_[i]);
    }
  }
  systemInitialized_ = false;
}

void
HypreUVWLinearSystem::finalizeSolver()
{

  MPI_Comm comm = realm_.bulk_data().parallel();
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  HYPRE_IJMatrixCreate(comm, iLower_, iUpper_, jLower_, jUpper_, &mat_);
  HYPRE_IJMatrixSetObjectType(mat_, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  for (unsigned i=0; i<nDim_; ++i) {
    HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &rhs_[i]);
    HYPRE_IJVectorSetObjectType(rhs_[i], HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(rhs_[i]);
    HYPRE_IJVectorGetObject(rhs_[i], (void**)&(solver->parRhsU_[i]));

    HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &sln_[i]);
    HYPRE_IJVectorSetObjectType(sln_[i], HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(sln_[i]);
    HYPRE_IJVectorGetObject(sln_[i], (void**)&(solver->parSlnU_[i]));
  }
}

void
HypreUVWLinearSystem::loadComplete()
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

  // All algorithms have called sumInto and populated LHS/RHS. Now we are ready
  // to finalize the matrix at the HYPRE end. However, before we do that we need
  // to process unfilled rows and process them appropriately. Any row acted on
  // by sumInto method will have toggled the rowFilled_ array to RS_FILLED
  // status. Before finalizing assembly, we process rows that still have an
  // RS_UNFILLED status and set their diagonal entries to 1.0 (dummy row)
  //
  // TODO: Alternate design to eliminate dummy rows. This will require
  // load-balancing on HYPRE end.
  numAssembles_++;

#ifdef KOKKOS_ENABLE_CUDA

  std::vector<void *> rhs(nDim_);
  for (unsigned i=0; i<nDim_; ++i) rhs[i] = (void*)(&rhs_[i]);

  hostCoeffApplier->finishAssembly((void*)&mat_, rhs, numAssembles_, name_);

#else

  HypreIntType hnrows = 1;
  HypreIntType hncols = 1;
  double getval;
  double setval = 1.0;
  for (HypreIntType i=0; i < numRows_; i++) {
    if (rowFilled_[i] == RS_FILLED) continue;
    HypreIntType lid = iLower_ + i;
    HYPRE_IJMatrixGetValues(mat_, hnrows, &hncols, &lid, &lid, &getval);
    if (std::fabs(getval) < 1.0e-12) {
      HYPRE_IJMatrixSetValues(mat_, hnrows, &hncols, &lid, &lid, &setval);
    }
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
    rows_.push_back(lid);
    cols_.push_back(lid);
    vals_.push_back(setval);
    for (unsigned j=0; j<nDim_; ++j) {
      rhs_rows_[j].push_back(lid);
      rhs_vals_[j].push_back(0.0);
    }
#endif
  }

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
  char fname[100];
  sprintf(fname,"%s_rowIndices%d.bin",name_.c_str(),numAssembles_);
  std::ofstream hrfile(fname, std::ios::out | std::ios::binary);
  hrfile.write((char*)&rows_[0], rows_.size() * sizeof(HypreIntType));
  hrfile.close();
  
  sprintf(fname,"%s_colIndices%d.bin",name_.c_str(),numAssembles_);
  std::ofstream hcfile(fname, std::ios::out | std::ios::binary);
  hcfile.write((char*)&cols_[0], cols_.size() * sizeof(HypreIntType));
  hcfile.close();
  
  sprintf(fname,"%s_values%d.bin",name_.c_str(),numAssembles_);
  std::ofstream hdfile(fname, std::ios::out | std::ios::binary);
  hdfile.write((char*)&vals_[0], vals_.size() * sizeof(double));
  hdfile.close();
  
  for (unsigned j=0; j<nDim_; ++j) {
    sprintf(fname,"%s_rhsRowIndices%d_%d.bin",name_.c_str(),numAssembles_,j);
    std::ofstream hrhsfile(fname, std::ios::out | std::ios::binary);
    hrhsfile.write((char*)(rhs_rows_[j].data()), rhs_rows_[j].size() * sizeof(HypreIntType));
    hrhsfile.close();
    
    sprintf(fname,"%s_rhsValues%d_%d.bin",name_.c_str(),numAssembles_,j);
    std::ofstream hrhsvfile(fname, std::ios::out | std::ios::binary);
    hrhsvfile.write((char*)(rhs_vals_[j].data()), rhs_vals_[j].size() * sizeof(double));
    hrhsvfile.close();
  }
  
  std::vector<HypreIntType> hmetaData(0);
  hmetaData.push_back((HypreIntType)(iLower_));
  hmetaData.push_back((HypreIntType)(iUpper_));
  hmetaData.push_back((HypreIntType)(jLower_));
  hmetaData.push_back((HypreIntType)(jUpper_));
  hmetaData.push_back((HypreIntType)(rows_.size()));
  hmetaData.push_back((HypreIntType)(rhs_rows_[0].size()));
  sprintf(fname,"%s_metaData%d.bin",name_.c_str(),numAssembles_);
  std::ofstream hmdfile(fname, std::ios::out | std::ios::binary);
  long pos = hmdfile.tellp();
  int size = sizeof(HypreIntType);
  hmdfile.write((char *)&size, 4);
  hmdfile.seekp(pos+4);
  hmdfile.write((char*)&hmetaData[0], hmetaData.size() * sizeof(HypreIntType));
  hmdfile.close();
#endif

#endif

  loadCompleteSolver();

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
}


void
HypreUVWLinearSystem::loadCompleteSolver()
{
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  HYPRE_IJMatrixAssemble(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  for (unsigned i=0; i<nDim_; ++i) {
    HYPRE_IJVectorAssemble(rhs_[i]);
    HYPRE_IJVectorGetObject(rhs_[i], (void**)&(solver->parRhsU_[i]));

    HYPRE_IJVectorAssemble(sln_[i]);
    HYPRE_IJVectorGetObject(sln_[i], (void**)&(solver->parSlnU_[i]));
  }

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, num_rows=%lld, num_nonzeros=%lld\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),
	 hypre_ParCSRMatrixDiag((HYPRE_ParCSRMatrix) (solver->parMat_))->num_rows,
	 hypre_ParCSRMatrixDiag((HYPRE_ParCSRMatrix) (solver->parMat_))->num_nonzeros);
#endif

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
  hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag((HYPRE_ParCSRMatrix) (solver->parMat_));
  HYPRE_Int * hr = (HYPRE_Int *) hypre_CSRMatrixI(diag);
  HYPRE_Int * hc = (HYPRE_Int *) hypre_CSRMatrixJ(diag);
  double * hd = (double *) hypre_CSRMatrixData(diag);
  HYPRE_Int num_rows = diag->num_rows;
  HYPRE_Int num_nonzeros = diag->num_nonzeros;

  char fname[50];
  sprintf(fname,"%s_HypreRows%d.bin",name_.c_str(),numAssembles_);
  std::ofstream hrfile(fname, std::ios::out | std::ios::binary);
  std::vector<HypreIntType> tmp(num_rows+1);
  for (int i=0; i<num_rows+1; ++i) { tmp[i] = (HypreIntType)hr[i]; }
  hrfile.write((char*)&tmp[0], (num_rows+1) * sizeof(HypreIntType));
  hrfile.close();
    
  sprintf(fname,"%s_HypreCols%d.bin",name_.c_str(),numAssembles_);
  std::ofstream hcfile(fname, std::ios::out | std::ios::binary);
  tmp.resize(num_nonzeros);
  for (int i=0; i<num_nonzeros; ++i) { tmp[i] = (HypreIntType)hc[i]; }
  hcfile.write((char*)&tmp[0], num_nonzeros * sizeof(HypreIntType));
  hcfile.close();
    
  sprintf(fname,"%s_HypreData%d.bin",name_.c_str(),numAssembles_);
  std::ofstream hdfile(fname, std::ios::out | std::ios::binary);
  hdfile.write((char*)&hd[0], num_nonzeros * sizeof(double));
  hdfile.close();

  for (unsigned i=0; i<nDim_; ++i) {
    double * local_data = hypre_VectorData(hypre_ParVectorLocalVector(solver->parRhsU_[i]));
    sprintf(fname,"%s_HypreRHSData%d_%d.bin",name_.c_str(),numAssembles_,i);
    std::ofstream hrhsfile(fname, std::ios::out | std::ios::binary);
    hrhsfile.write((char*)local_data, num_rows * sizeof(double));
    hrhsfile.close();
  }    
  
  std::vector<HypreIntType> hmetaData(0);
  hmetaData.push_back((HypreIntType)num_rows);
  hmetaData.push_back((HypreIntType)num_nonzeros);
  sprintf(fname,"%s_HypreMetaData%d.bin",name_.c_str(),numAssembles_);
  std::ofstream hmdfile(fname, std::ios::out | std::ios::binary);
  long pos = hmdfile.tellp();
  int size = sizeof(HypreIntType);
  hmdfile.write((char *)&size, 4);
  hmdfile.seekp(pos+4);
  hmdfile.write((char*)&hmetaData[0], hmetaData.size() * sizeof(HypreIntType));
  hmdfile.close();
#endif

  solver->comm_ = realm_.bulk_data().parallel();

  matrixAssembled_ = true;
}

void
HypreUVWLinearSystem::zeroSystem()
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("\n\nZero System\n%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
  rows_.resize(0);
  cols_.resize(0);
  vals_.resize(0);
  rhs_rows_.resize(nDim_);
  rhs_vals_.resize(nDim_);
  for (unsigned i=0; i<nDim_; ++i) {
    rhs_rows_[i].resize(0);
    rhs_vals_[i].resize(0);
  }
#endif

  if (matrixAssembled_) {
    HYPRE_IJMatrixInitialize(mat_);
    for (unsigned i=0; i<nDim_; ++i) {
      HYPRE_IJVectorInitialize(rhs_[i]);
      HYPRE_IJVectorInitialize(sln_[i]);
    }

    matrixAssembled_ = false;
  }

  HYPRE_IJMatrixSetConstantValues(mat_, 0.0);
  for (unsigned i=0; i<nDim_; ++i) {
    HYPRE_ParVectorSetConstantValues((solver->parRhsU_[i]), 0.0);
    HYPRE_ParVectorSetConstantValues((solver->parSlnU_[i]), 0.0);
  }

  // Prepare for matrix assembly and set all entry flags to "unfilled"
  for (HypreIntType i=0; i < numRows_; i++)
    rowFilled_[i] = RS_UNFILLED;

  // Set flag to indicate whether rows must be skipped during normal sumInto
  // process. For this to be activated, the linear system must have Dirichlet or
  // overset rows and they must be present on this processor
  if (hasSkippedRows_ && !skippedRows_.empty())
    checkSkippedRows_ = true;
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
}

void
HypreUVWLinearSystem::sumInto(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const SharedMemView<int*, DeviceShmem>&,
  const SharedMemView<int*, DeviceShmem>&,
  const char*  /* trace_tag */)
{
#ifndef KOKKOS_ENABLE_CUDA
  HypreIntType numRows = numEntities;
  const HypreIntType bufSize = idBuffer_.size();

  ThrowAssertMsg(lhs.span_is_contiguous(), "LHS assumed contiguous");
  ThrowAssertMsg(rhs.span_is_contiguous(), "RHS assumed contiguous");
  if (bufSize < numRows) {
    idBuffer_.resize(numRows);
    scratchRowVals_.resize(numRows);
  }

  for (size_t in=0; in < numEntities; in++) {
    idBuffer_[in] = get_entity_hypre_id(entities[in]);
  }

  for (size_t in=0; in < numEntities; in++) {
    int ix = in * nDim_;
    HypreIntType hid = idBuffer_[in];

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) {
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
	for (unsigned j=0; j<numRows; ++j) {
	  rows_.push_back(-1);
	  cols_.push_back(-1);
	  vals_.push_back(0.0);
	}
	for (size_t d=0; d < numDof_; d++) {
	  rhs_rows_[d].push_back(-1);
	  rhs_vals_[d].push_back(0.0);
	}
#endif
	continue;
      }
    }

    int offset = 0;
    for (int c=0; c < numRows; c++) {
      scratchRowVals_[c] = lhs(ix, offset);
      offset += nDim_;
    }
    HYPRE_IJMatrixAddToValues(
      mat_, 1, &numRows, &hid, &idBuffer_[0], &scratchRowVals_[0]);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
    for (int j=0; j<numRows; ++j) {
      rows_.push_back(hid);
      cols_.push_back(idBuffer_[j]);
      vals_.push_back(scratchRowVals_[j]);
    }
#endif

    for (unsigned d=0; d<nDim_; ++d) {
      int ir = ix + d;
      HYPRE_IJVectorAddToValues(rhs_[d], 1, &hid, &rhs[ir]);
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
      rhs_rows_[d].push_back(hid);
      rhs_vals_[d].push_back(rhs[ir]);
#endif
    }

    if ((hid >= iLower_) && (hid <= iUpper_))
      rowFilled_[hid - iLower_] = RS_FILLED;
  }
#endif
}

void
HypreUVWLinearSystem::sumInto(
  const std::vector<stk::mesh::Entity>& entities,
  std::vector<int>&  /* scratchIds */,
  std::vector<double>& scratchVals,
  const std::vector<double>& rhs,
  const std::vector<double>& lhs,
  const char*  /* trace_tag */)
{
  const size_t n_obj = entities.size();
  HypreIntType numRows = n_obj;
  const HypreIntType bufSize = idBuffer_.size();

#ifndef NDEBUG
  size_t vecSize = numRows * nDim_;
  ThrowAssert(vecSize == rhs.size());
  ThrowAssert(vecSize*vecSize == lhs.size());
#endif
  if (bufSize < numRows) idBuffer_.resize(numRows);

  for (size_t in=0; in < n_obj; in++) {
    idBuffer_[in] = get_entity_hypre_id(entities[in]);
  }

  for (size_t in=0; in < n_obj; in++) {
    int ix = in * nDim_;
    HypreIntType hid = get_entity_hypre_id(entities[in]);

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) {
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
	for (unsigned j=0; j<numRows; ++j) {
	  rows_.push_back(-1);
	  cols_.push_back(-1);
	  vals_.push_back(0.0);
	}
	for (size_t d=0; d < numDof_; d++) {
	  rhs_rows_[d].push_back(-1);
	  rhs_vals_[d].push_back(0.0);
	}
#endif
	continue;
      }
    }

    int offset = 0;
    int ic = ix * numRows * nDim_;
    for (int c=0; c < numRows; c++) {
      scratchVals[c] = lhs[ic + offset];
      offset += nDim_;
    }
    HYPRE_IJMatrixAddToValues(
      mat_, 1, &numRows, &hid, &idBuffer_[0], &scratchVals[0]);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
    for (int j=0; j<numRows; ++j) {
      rows_.push_back(hid);
      cols_.push_back(idBuffer_[j]);
      vals_.push_back(scratchRowVals_[j]);
    }
#endif

    for (unsigned d = 0; d<nDim_; ++d) {
      int ir = ix + d;
      HYPRE_IJVectorAddToValues(rhs_[d], 1, &hid, &rhs[ir]);
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
      rhs_rows_[d].push_back(hid);
      rhs_vals_[d].push_back(rhs[ir]);
#endif
    }

    if ((hid >= iLower_) && (hid <= iUpper_))
      rowFilled_[hid - iLower_] = RS_FILLED;
  }
}

void
HypreUVWLinearSystem::applyDirichletBCs(
  stk::mesh::FieldBase* solutionField,
  stk::mesh::FieldBase* bcValuesField,
  const stk::mesh::PartVector& parts,
  const unsigned,
  const unsigned)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

  double adbc_time = -NaluEnv::self().nalu_time();

#ifdef KOKKOS_ENABLE_CUDA

  hostCoeffApplier->applyDirichletBCs(realm_, solutionField, bcValuesField, parts);

#else 

  auto& meta = realm_.meta_data();

  const stk::mesh::Selector sel = (
    meta.locally_owned_part() &
    stk::mesh::selectUnion(parts) &
    stk::mesh::selectField(*solutionField) &
    !(realm_.get_inactive_selector()));

  const auto& bkts = realm_.get_buckets(
    stk::topology::NODE_RANK, sel);

  HypreIntType ncols = 1;
  double diag_value = 1.0;
  for (auto b: bkts) {
    const double* solution = (double*)stk::mesh::field_data(
      *solutionField, *b);
    const double* bcValues = (double*)stk::mesh::field_data(
      *bcValuesField, *b);

    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);

      HYPRE_IJMatrixSetValues(mat_, 1, &ncols, &hid, &hid, &diag_value);
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
      rows_.push_back(hid);
      cols_.push_back(hid);
      vals_.push_back(diag_value);
#endif

      for (unsigned d=0; d<nDim_; ++d) {
        double bcval = bcValues[in*nDim_ + d] - solution[in*nDim_ + d];
        HYPRE_IJVectorSetValues(rhs_[d], 1, &hid, &bcval);
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
	rhs_rows_[d].push_back(hid);
	rhs_vals_[d].push_back(bcval);
#endif
      }
      rowFilled_[hid - iLower_] = RS_FILLED;
    }
  }
#endif

  adbc_time += NaluEnv::self().nalu_time();

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
}

int
HypreUVWLinearSystem::solve(stk::mesh::FieldBase* slnField)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
    const std::string matFile = eqSysName_ + ".IJM." + writeCounter + ".mat";
    HYPRE_IJMatrixPrint(mat_, matFile.c_str());

    for (unsigned d=0; d<nDim_; ++d) {
      const std::string rhsFile =
        eqSysName_ + std::to_string(d) + ".IJV." + writeCounter + ".rhs";
      HYPRE_IJVectorPrint(rhs_[d], rhsFile.c_str());
    }
  }

  int status = 0;
  std::vector<int> iters(nDim_, 0);
  std::vector<double> finalNorm(nDim_, 1.0);
  std::vector<double> rhsNorm(nDim_, std::numeric_limits<double>::max());

  for (unsigned d=0; d<nDim_; ++d) {
    status = solver->solve(d, iters[d], finalNorm[d], realm_.isFinalOuterIter_);
  }
  copy_hypre_to_stk(slnField, rhsNorm);
  sync_field(slnField);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    for (unsigned d=0; d < nDim_; ++d) {
      std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
      const std::string slnFile = eqSysName_ + std::to_string(d) + ".IJV." + writeCounter + ".sln";
      HYPRE_IJVectorPrint(sln_[d], slnFile.c_str());
      ++eqSys_->linsysWriteCounter_;
    }
  }

  {
    linearSolveIterations_ = 0;
    linearResidual_ = 0.0;
    nonLinearResidual_ = 0.0;
    double linres, nonlinres, scaledres, tmp, scaleFac = 0.0;

    for (unsigned d=0; d<nDim_; ++d) {
      linres = finalNorm[d] * rhsNorm[d];
      nonlinres = realm_.l2Scaling_ * rhsNorm[d];

      if (eqSys_->firstTimeStepSolve_)
        firstNLR_[d] = nonlinres;

      tmp = std::max(std::numeric_limits<double>::epsilon(), firstNLR_[d]);
      scaledres = nonlinres / tmp;
      scaleFac += tmp * tmp;

      linearResidual_ += linres * linres;
      nonLinearResidual_ += nonlinres * nonlinres;
      scaledNonLinearResidual_ += scaledres * scaledres;
      linearSolveIterations_ += iters[d];

      if (provideOutput_) {
        const int nameOffset = eqSysName_.length() + 10;

        NaluEnv::self().naluOutputP0()
          << std::setw(nameOffset) << std::right << eqSysName_+"_"+vecNames_[d]
          << std::setw(32 - nameOffset) << std::right << iters[d] << std::setw(18)
          << std::right << linres << std::setw(15) << std::right
          << nonlinres << std::setw(14) << std::right
          << scaledres << std::endl;
      }
    }
    linearResidual_ = std::sqrt(linearResidual_);
    nonLinearResidual_ = std::sqrt(nonLinearResidual_);
    scaledNonLinearResidual_ = nonLinearResidual_ / std::sqrt(scaleFac);

    if (provideOutput_) {
      const int nameOffset = eqSysName_.length() + 8;
      NaluEnv::self().naluOutputP0()
        << std::setw(nameOffset) << std::right << eqSysName_
        << std::setw(32 - nameOffset) << std::right << linearSolveIterations_ << std::setw(18)
        << std::right << linearResidual_ << std::setw(15) << std::right
        << nonLinearResidual_ << std::setw(14) << std::right
        << scaledNonLinearResidual_ << std::endl;
    }
  }

  eqSys_->firstTimeStepSolve_ = false;
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
  return status;
}


void
HypreUVWLinearSystem::copy_hypre_to_stk(
  stk::mesh::FieldBase* stkField, std::vector<double>& rhsNorm)
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();
  const auto sel = stk::mesh::selectField(*stkField)
    & meta.locally_owned_part()
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  const auto& bkts = bulk.get_buckets(
    stk::topology::NODE_RANK, sel);

  std::vector<double> lclnorm(nDim_, 0.0);
  std::vector<double> gblnorm(nDim_, 0.0);
  double rhsVal = 0.0;

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  int c=0;
#endif

  for (auto b: bkts) {
    double* field = (double*) stk::mesh::field_data(*stkField, *b);
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = get_entity_hypre_id(node);

      for (unsigned d=0; d<nDim_; ++d) {
        int sid = in * nDim_ + d;
        HYPRE_IJVectorGetValues(sln_[d], 1, &hid, &field[sid]);
        HYPRE_IJVectorGetValues(rhs_[d], 1, &hid, &rhsVal);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
	//printf("%d : lid=%lld, sid=%d, sln=%1.16lf, rhs=%1.16lf\n",c,hid,sid,field[sid],rhsVal);
	c++;
#endif
        lclnorm[d] += rhsVal * rhsVal;
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  char fname[200];
#if !defined(HYPRE_LINEAR_SYSTEM_DEBUG_LOAD_FROM_CPU_LISTS) && !defined(HYPRE_LINEAR_SYSTEM_DEBUG_LOAD_FROM_CPU)
  std::string extension="";
#else
#if defined(HYPRE_LINEAR_SYSTEM_DEBUG_LOAD_FROM_CPU_LISTS)
  std::string extension="FromCPULoadLists";
#else
  std::string extension="FromCPULoad";
#endif
#endif
  for (unsigned i=0; i<nDim_; ++i) {
    double * local_data = hypre_VectorData(hypre_ParVectorLocalVector(solver->parSlnU_[i]));
    sprintf(fname,"%s_HypreSolution%s%d_%d.bin",name_.c_str(),extension.c_str(),numAssembles_,i);
    
    std::ofstream slnfile(fname, std::ios::out | std::ios::binary);
    slnfile.write((char*)local_data, numRows_ * sizeof(double));
    slnfile.close();
  }
#endif


  NGPDoubleFieldType ngpField = realm_.ngp_field_manager().get_field<double>(stkField->mesh_meta_data_ordinal());
  ngpField.modify_on_host();
  ngpField.sync_to_device();

  stk::all_reduce_sum(bulk.parallel(), lclnorm.data(), gblnorm.data(), nDim_);

  for (unsigned d=0; d<nDim_; ++d)
    rhsNorm[d] = std::sqrt(gblnorm[d]);
}



sierra::nalu::CoeffApplier* HypreUVWLinearSystem::get_coeff_applier()
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

#ifdef KOKKOS_ENABLE_CUDA

  if (!hostCoeffApplier) {

    unsigned numPartitions = partitionCount_.size();
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
    printf("%s %s %d : name=%s numPartitions=%d\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str(),(int)numPartitions);
#endif

    HypreIntTypeView mat_count = HypreIntTypeView("mat_count",numPartitions);
    HypreIntTypeViewHost mat_count_host = Kokkos::create_mirror_view(mat_count);

    HypreIntTypeView rhs_count = HypreIntTypeView("rhs_count",numPartitions);
    HypreIntTypeViewHost rhs_count_host = Kokkos::create_mirror_view(rhs_count);

    HypreIntTypeView mat_partition_start = HypreIntTypeView("mat_partition_start",numPartitions);
    HypreIntTypeViewHost mat_partition_start_host = Kokkos::create_mirror_view(mat_partition_start);

    HypreIntTypeView rhs_partition_start = HypreIntTypeView("rhs_partition_start",numPartitions);
    HypreIntTypeViewHost rhs_partition_start_host = Kokkos::create_mirror_view(rhs_partition_start);

    HypreIntTypeView2D partition_node_start = HypreIntTypeView2D("partition_node_start",numRows_,numPartitions);
    HypreIntTypeView2DHost partition_node_start_host = Kokkos::create_mirror_view(partition_node_start);

    HypreIntType numMatPtsToAssembleTotal = 0;
    HypreIntType numRhsPtsToAssembleTotal = 0;

    /******************************************************/
    /* Construct the Partition Node Start Data structures */
    for (unsigned i=0; i<numPartitions; ++i) {
      for (unsigned j=0; j<numRows_; ++j) {
	partition_node_start_host(j,i) = partitionNodeStart_[i][j];
      }
    }
    Kokkos::deep_copy(partition_node_start_host, partition_node_start);

    /**************************************************/
    /* Construct the Matrix Partition Data structures */
    for (unsigned i=0; i<numPartitions; ++i) {
      /* set the counts */
      mat_count_host(i) = count_[i];
      /* the number of points to assemble */
      numMatPtsToAssembleTotal += partitionCount_[i]*mat_count_host(i);
      /* compute the start */
      mat_partition_start_host(i) = numMatPtsToAssembleTotal - partitionCount_[i]*mat_count_host(i);
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
      printf("Matrix partition %d : total in partition=%d, count per element=%d, partitionStart=%d, total=%d\n",
	     i,(int)partitionCount_[i],(int)mat_count_host(i),(int)mat_partition_start_host(i),(int)numMatPtsToAssembleTotal);
#endif
    }
    Kokkos::deep_copy(mat_count, mat_count_host);
    Kokkos::deep_copy(mat_partition_start, mat_partition_start_host);

    /***********************************************/
    /* Construct the Rhs Partition Data structures */
    for (unsigned i=0; i<numPartitions; ++i) {
      /* set the counts */
      rhs_count_host(i) = sqrt(count_[i]);
      /* se the number of points to assemble */
      numRhsPtsToAssembleTotal += partitionCount_[i]*rhs_count_host(i);
      /* compute the start */
      rhs_partition_start_host(i) = numRhsPtsToAssembleTotal - partitionCount_[i]*rhs_count_host(i);
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
      printf("Rhs partition %d : total in partition=%d, count per element=%d, partitionStart=%d, total=%d\n",
	     i,(int)partitionCount_[i],(int)rhs_count_host(i),(int)rhs_partition_start_host(i),(int)numRhsPtsToAssembleTotal);
#endif
    }
    Kokkos::deep_copy(rhs_count, rhs_count_host);
    Kokkos::deep_copy(rhs_partition_start, rhs_partition_start_host);

    /*******************************/
    /* skipped rows data structure */
    HypreIntTypeUnorderedMap skippedRowsMap(skippedRows_.size());
    for (auto t : skippedRows_) {
      skippedRowsMap.insert(t,t);
    }
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
    printf("%s %s %d : name=%s skippedRowsMap size=%d\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str(),(int)skippedRowsMap.size());
#endif
    // Total number of global rows in the system
    HypreIntType maxRowID = realm_.hypreNumNodes_ * nDim_ - 1;
    
    hostCoeffApplier.reset(new HypreUVWLinSysCoeffApplier(nDim_, numPartitions, maxRowID,
							  iLower_, iUpper_, jLower_, jUpper_,
							  mat_partition_start, mat_count, numMatPtsToAssembleTotal,
							  rhs_partition_start, rhs_count, numRhsPtsToAssembleTotal,
							  partition_node_start, entityToLID_, skippedRowsMap));
    deviceCoeffApplier = hostCoeffApplier->device_pointer();

    /* clear this data so that the next time a coeffApplier is built, these get rebuilt from scratch */
    partitionNodeStart_.clear();
    partitionCount_.clear();
    count_.clear();
  }
  /* reset the internal counters */
  hostCoeffApplier->resetInternalData();
  
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
  return deviceCoeffApplier;

#else

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
  return LinearSystem::get_coeff_applier();

#endif
}

/********************************************************************************************************/
/*                     Beginning of HypreUVWLinSysCoeffApplier implementations                          */
/********************************************************************************************************/

HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::HypreUVWLinSysCoeffApplier(unsigned numDof, 
									     unsigned numPartitions, HypreIntType maxRowID,
									     HypreIntType iLower, HypreIntType iUpper,
									     HypreIntType jLower, HypreIntType jUpper,
									     HypreIntTypeView mat_partition_start,
									     HypreIntTypeView mat_count,
									     HypreIntType numMatPtsToAssembleTotal,
									     HypreIntTypeView rhs_partition_start,
									     HypreIntTypeView rhs_count,
									     HypreIntType numRhsPtsToAssembleTotal,
									     HypreIntTypeView2D partition_node_start,
									     EntityToHypreIntTypeView entityToLID,
									     HypreIntTypeUnorderedMap skippedRowsMap)
  : HypreLinSysCoeffApplier(numDof, numPartitions, maxRowID,
			    iLower, iUpper, jLower, jUpper,
			    mat_partition_start, mat_count, numMatPtsToAssembleTotal,
			    rhs_partition_start, rhs_count, numRhsPtsToAssembleTotal,
			    partition_node_start, entityToLID, skippedRowsMap) {
  
  nDim_ = numDof;
}

KOKKOS_FUNCTION
void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::sum_into(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  unsigned numDof, HypreIntType iLower, HypreIntType iUpper,
  HypreIntType partitionIndex) {

  unsigned nDim = numDof;
  HypreIntType hid0 = entityToLID_[entities[0].local_offset()];
  HypreIntType counter = Kokkos::atomic_fetch_add(&partition_node_count_(hid0, partitionIndex), 1); 
  HypreIntType nodeStart = partition_node_start_(hid0, partitionIndex);
  HypreIntType matIndex = mat_partition_start_(partitionIndex) + (nodeStart + counter)*mat_count_(partitionIndex);
  HypreIntType rhsIndex = rhs_partition_start_(partitionIndex) + (nodeStart + counter)*rhs_count_(partitionIndex);

  for(unsigned i=0; i<numEntities; i++) {
    localIds[i] = entityToLID_[entities[i].local_offset()];
  }

  for (unsigned i=0; i<numEntities; i++) {
    int ix = i * nDim;
    HypreIntType hid = localIds[i];
    if (checkSkippedRows_()) {
      if (skippedRowsMap_.exists(hid)) continue;
    }

    int offset = 0;
    for (unsigned k=0; k<numEntities; k++) {
      rows_(matIndex+i*numEntities+k) = hid;
      cols_(matIndex+i*numEntities+k) = localIds[k];
      vals_(matIndex+i*numEntities+k) = lhs(ix, offset);
      offset += nDim;
    }

    for (unsigned d=0; d<nDim; d++) {
      int ir = ix + d;
      rhs_rows_(rhsIndex+i,d) = hid;
      rhs_vals_(rhsIndex+i,d) = rhs[ir];
    }

    if ((hid >= iLower) && (hid <= iUpper))
      row_filled_(hid - iLower) = RS_FILLED;
  }
}

KOKKOS_FUNCTION
void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::operator()(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<int*, DeviceShmem>&,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const char* /*trace_tag*/)
{
  sum_into(numEntities,entities,localIds,rhs,lhs,nDim_,iLower_,iUpper_,partition_index_());
}

void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::applyDirichletBCs(Realm & realm, 
								    stk::mesh::FieldBase * solutionField,
								    stk::mesh::FieldBase * bcValuesField,
								    const stk::mesh::PartVector& parts) {
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d\n",__FILE__,__FUNCTION__,__LINE__);
#endif
  resetInternalData();

#if 1

  /************************************************************/
  /* this is a hack to get dirichlet bcs working consistently */

  /* Step 1: copy the row_filled_ to its host mirror */
  Kokkos::deep_copy(row_filled_host, row_filled_);

  /* Step 2: execute the old CPU code */
  auto& meta = realm.meta_data();

  const stk::mesh::Selector sel = (
    meta.locally_owned_part() &
    stk::mesh::selectUnion(parts) &
    stk::mesh::selectField(*solutionField) &
    !(realm.get_inactive_selector()));

  const auto& bkts = realm.get_buckets(
    stk::topology::NODE_RANK, sel);

  double diag_value = 1.0;
  std::vector<HypreIntType> tRows(0);
  std::vector<HypreIntType> tCols(0);
  std::vector<double> tVals(0);
  std::vector<std::vector<HypreIntType> >trhsRows(nDim_);
  std::vector<std::vector<double> >trhsVals(nDim_);
  for (unsigned i=0;i<nDim_;++i) {
    trhsRows[i].resize(0);
    trhsVals[i].resize(0);
  }

  NGPDoubleFieldType ngpSolutionField = realm.ngp_field_manager().get_field<double>(solutionField->mesh_meta_data_ordinal());
  NGPDoubleFieldType ngpBCValuesField = realm.ngp_field_manager().get_field<double>(bcValuesField->mesh_meta_data_ordinal());

  ngpSolutionField.sync_to_host();
  ngpBCValuesField.sync_to_host();

  for (auto b: bkts) {
    const double* solution = (double*)stk::mesh::field_data(
      *solutionField, *b);
    const double* bcValues = (double*)stk::mesh::field_data(
      *bcValuesField, *b);

    for (unsigned in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm.hypreGlobalId_, node);

      /* fill the mirrored version */
      row_filled_host(hid - iLower_) = RS_FILLED;
      
      /* fill these temp values */
      tRows.push_back(hid);
      tCols.push_back(hid);
      tVals.push_back(diag_value);
      
      for (unsigned d=0; d<nDim_; d++) {
        double bcval = bcValues[in*nDim_ + d] - solution[in*nDim_ + d];
	trhsRows[d].push_back(hid);
	trhsVals[d].push_back(bcval);
      }
    }
  }

  /* Step 3 : allocate space in which to push the temporaries */
  HypreIntTypeView r("r",tRows.size());
  HypreIntTypeViewHost rh  = Kokkos::create_mirror_view(r);

  HypreIntTypeView c("c",tCols.size());
  HypreIntTypeViewHost ch  = Kokkos::create_mirror_view(c);

  DoubleView v("v",tVals.size());
  DoubleViewHost vh  = Kokkos::create_mirror_view(v);

  Kokkos::View<HypreIntType**> rr("rr",trhsRows[0].size(),nDim_);
  Kokkos::View<HypreIntType**>::HostMirror rrh  = Kokkos::create_mirror_view(rr);

  Kokkos::View<double**> rv("rv",trhsVals[0].size(),nDim_);
  Kokkos::View<double**>::HostMirror rvh  = Kokkos::create_mirror_view(rv);

  /* Step 4 : next copy the std::vectors into the host mirrors */
  for (unsigned int i=0; i<tRows.size(); ++i) {
    rh(i) = tRows[i];
    ch(i) = tCols[i];
    vh(i) = tVals[i];
    for (unsigned j=0; j<nDim_;++j) {
      rrh(i,j) = trhsRows[j][i];
      rvh(i,j) = trhsVals[j][i];
    }
  }

  /* Step 5 : deep copy this to device */
  Kokkos::deep_copy(row_filled_,row_filled_host);
  Kokkos::deep_copy(r,rh);
  Kokkos::deep_copy(c,ch);
  Kokkos::deep_copy(v,vh);
  Kokkos::deep_copy(rr,rrh);
  Kokkos::deep_copy(rv,rvh);

  /* Step 6 : append this to the existing data structure */
  /* for some reason, Kokkos::parallel_for with a LAMBDA function does not compile. */
  kokkos_parallel_for("bcHackUVW", tRows.size(), [&] (const unsigned& i) {
      HypreIntType matIndex = mat_partition_start_(partition_index_())+i;
      HypreIntType rhsIndex = rhs_partition_start_(partition_index_())+i;
      rows_(matIndex)=r(i);
      cols_(matIndex)=c(i);
      vals_(matIndex)=v(i);
      for (unsigned j=0; j<nDim_;++j) {
	rhs_rows_(rhsIndex,j) = rr(i,j);
	rhs_vals_(rhsIndex,j) = rv(i,j);
      }
    });

#else

#endif

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d\n",__FILE__,__FUNCTION__,__LINE__);
#endif
}

void HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::free_device_pointer()
{
#ifdef KOKKOS_ENABLE_CUDA
  if (this != devicePointer_) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
#endif
}

sierra::nalu::CoeffApplier* HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::device_pointer()
{
#ifdef KOKKOS_ENABLE_CUDA
  if (devicePointer_ != nullptr) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
  devicePointer_ = sierra::nalu::create_device_expression(*this);
#else
  devicePointer_ = this;
#endif
  return devicePointer_;
}


/*********************************************************************************************************/
/*                           End of HypreUVWLinSysCoeffApplier implementations                           */
/*********************************************************************************************************/

void
HypreUVWLinearSystem::buildNodeGraph(const stk::mesh::PartVector & parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
#endif

  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
    & stk::mesh::selectUnion(parts)
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_owned );

  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=-1;
  std::vector<HypreIntType> nodeCount(numRows_);
  std::fill(nodeCount.begin(), nodeCount.end(), 0);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
#endif
  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      /* fill temporaries */
      index++;
      count = count<0 ? 1 : count;

      stk::mesh::Entity node = b[k];
      HypreIntType hid = get_entity_hypre_id(node);
      nodeCount[hid]++;
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
#endif

  /*inclusive scan ... this vector is going to be used to give a hint of where to write in the list
    depending on the hid of the first node */
  std::vector<HypreIntType> nodeStart(numRows_+1);
  std::fill(nodeStart.begin(), nodeStart.end(), 0);
  for (unsigned i=0; i<numRows_; ++i) {
    nodeStart[i+1] = nodeStart[i] + nodeCount[i];
  }

  /* save these */
  partitionNodeStart_.push_back(nodeStart);
  partitionCount_.push_back(index);
  count_.push_back(count);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s, nDim=%d, numPtsInThisPartition=%lld (from inc scan=%lld), numPartitions=%d, numDataPtsToAssemble=%d, count=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,index,nodeStart.back(),(int)partitionCount_.size(),(int)numDataPtsToAssemble(),int(count_.back()));
#endif
}


void
HypreUVWLinearSystem::buildFaceToNodeGraph(const stk::mesh::PartVector & parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
#endif

  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
                                      & stk::mesh::selectUnion(parts)
                                      & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets = realm_.get_buckets(realm_.meta_data().side_rank(), s_owned);

  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=-1;
  std::vector<HypreIntType> nodeCount(numRows_);
  std::fill(nodeCount.begin(), nodeCount.end(), 0);

  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      /* fill temporaries */
      index++;
      const unsigned numNodes = b.num_nodes(k);
      count = count<0 ? (HypreIntType)(numNodes*numNodes) : count;

      /* get the first nodes hid */
      if (numNodes) {
	stk::mesh::Entity const * nodes = b.begin_nodes(k);
	HypreIntType hid = get_entity_hypre_id(nodes[0]);
	nodeCount[hid]++;
      }
    }
  }

  /*inclusive scan ... this vector is going to be used to give a hint of where to write in the list
    depending on the hid of the first node */
  std::vector<HypreIntType> nodeStart(numRows_+1);
  std::fill(nodeStart.begin(), nodeStart.end(), 0);
  for (unsigned i=0; i<numRows_; ++i) {
    nodeStart[i+1] = nodeStart[i] + nodeCount[i];
  }

  /* save these */
  partitionNodeStart_.push_back(nodeStart);
  partitionCount_.push_back(index);
  count_.push_back(count);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s, nDim=%d, numPtsInThisPartition=%lld (from inc scan=%lld), numPartitions=%d, numDataPtsToAssemble=%d, count=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,index,nodeStart.back(),(int)partitionCount_.size(),(int)numDataPtsToAssemble(),int(count_.back()));
#endif
}

void
HypreUVWLinearSystem::buildEdgeToNodeGraph(const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
#endif

  beginLinearSystemConstruction();

  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
                                      & stk::mesh::selectUnion(parts)
                                      & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets = realm_.get_buckets(stk::topology::EDGE_RANK, s_owned);

  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=-1;
  std::vector<HypreIntType> nodeCount(numRows_);
  std::fill(nodeCount.begin(), nodeCount.end(), 0);

  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      /* fill temporaries */
      index++;
      const unsigned numNodes = b.num_nodes(k);
      count = count<0 ? (HypreIntType)(numNodes*numNodes) : count;

      /* get the first nodes hid */
      if (numNodes) {
	stk::mesh::Entity const * nodes = b.begin_nodes(k);
	HypreIntType hid = get_entity_hypre_id(nodes[0]);
	nodeCount[hid]++;
      }
    }
  }

  /*inclusive scan ... this vector is going to be used to give a hint of where to write in the list
    depending on the hid of the first node */
  std::vector<HypreIntType> nodeStart(numRows_+1);
  std::fill(nodeStart.begin(), nodeStart.end(), 0);
  for (unsigned i=0; i<numRows_; ++i) {
    nodeStart[i+1] = nodeStart[i] + nodeCount[i];
  }

  /* save these */
  partitionNodeStart_.push_back(nodeStart);
  partitionCount_.push_back(index);
  count_.push_back(count);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s, nDim=%d, numPtsInThisPartition=%lld (from inc scan=%lld), numPartitions=%d, numDataPtsToAssemble=%d, count=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,index,nodeStart.back(),(int)partitionCount_.size(),(int)numDataPtsToAssemble(),int(count_.back()));
#endif
}

void
HypreUVWLinearSystem::buildElemToNodeGraph(const stk::mesh::PartVector & parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
#endif

  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
                                      & stk::mesh::selectUnion(parts)
                                      & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets = realm_.get_buckets(stk::topology::ELEM_RANK, s_owned);

  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=-1;
  std::vector<HypreIntType> nodeCount(numRows_);
  std::fill(nodeCount.begin(), nodeCount.end(), 0);

  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      /* fill temporaries */
      index++;
      const unsigned numNodes = b.num_nodes(k);
      count = count<0 ? (HypreIntType)(numNodes*numNodes) : count;

      /* get the first nodes hid */
      if (numNodes) {
	stk::mesh::Entity const * nodes = b.begin_nodes(k);
	HypreIntType hid = get_entity_hypre_id(nodes[0]);
	nodeCount[hid]++;
      }
    }
  }

  /*inclusive scan ... this vector is going to be used to give a hint of where to write in the list
    depending on the hid of the first node */
  std::vector<HypreIntType> nodeStart(numRows_+1);
  std::fill(nodeStart.begin(), nodeStart.end(), 0);
  for (unsigned i=0; i<numRows_; ++i) {
    nodeStart[i+1] = nodeStart[i] + nodeCount[i];
  }

  /* save these */
  partitionNodeStart_.push_back(nodeStart);
  partitionCount_.push_back(index);
  count_.push_back(count);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s, nDim=%d, numPtsInThisPartition=%lld (from inc scan=%lld), numPartitions=%d, numDataPtsToAssemble=%d, count=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,index,nodeStart.back(),(int)partitionCount_.size(),(int)numDataPtsToAssemble(),int(count_.back()));
#endif
}

void
HypreUVWLinearSystem::buildFaceElemToNodeGraph(
  const stk::mesh::PartVector & parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
#endif

  beginLinearSystemConstruction();
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
    & stk::mesh::selectUnion(parts)
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( metaData.side_rank(), s_owned );

  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=-1;
  std::vector<HypreIntType> nodeCount(numRows_);
  std::fill(nodeCount.begin(), nodeCount.end(), 0);

  for(size_t ib=0; ib<face_buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *face_buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      const stk::mesh::Entity face = b[k];

      // extract the connected element to this exposed face; should be single in size!
      const stk::mesh::Entity* face_elem_rels = bulkData.begin_elements(face);
      ThrowAssert( bulkData.num_elements(face) == 1 );

      // get connected element and nodal relations
      stk::mesh::Entity element = face_elem_rels[0];
      const stk::mesh::Entity* elem_nodes = bulkData.begin_nodes(element);

      // figure out the global dof ids for each dof on each node
      const size_t numNodes = bulkData.num_nodes(element);
      const unsigned nn = numNodes*numNodes;
      count = count<nn ? (HypreIntType)(nn) : count;
      index++;

      /* get the first nodes hid */
      if (numNodes) {
	HypreIntType hid = get_entity_hypre_id(elem_nodes[0]);
	nodeCount[hid]++;
      }
    }
  }

  /*inclusive scan ... this vector is going to be used to give a hint of where to write in the list
    depending on the hid of the first node */
  std::vector<HypreIntType> nodeStart(numRows_+1);
  std::fill(nodeStart.begin(), nodeStart.end(), 0);
  for (unsigned i=0; i<numRows_; ++i) {
    nodeStart[i+1] = nodeStart[i] + nodeCount[i];
  }

  /* save these */
  partitionNodeStart_.push_back(nodeStart);
  partitionCount_.push_back(index);
  count_.push_back(count);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s, nDim=%d, numPtsInThisPartition=%lld (from inc scan=%lld), numPartitions=%d, numDataPtsToAssemble=%d, count=%d\n",
    __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,index,nodeStart.back(),(int)partitionCount_.size(),(int)numDataPtsToAssemble(),int(count_.back()));
#endif
}

void
HypreUVWLinearSystem::buildReducedElemToNodeGraph(
  const stk::mesh::PartVector&)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

  beginLinearSystemConstruction();

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
}

void
HypreUVWLinearSystem::buildNonConformalNodeGraph(
  const stk::mesh::PartVector&)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

  beginLinearSystemConstruction();

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
}

// void
// HypreUVWLinearSystem::buildOversetNodeGraph(
//   const stk::mesh::PartVector&)
// {
//   printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
// 	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
//   beginLinearSystemConstruction();

//   // Turn on the flag that indicates this linear system has rows that must be
//   // skipped during normal sumInto process
//   hasSkippedRows_ = true;

//   // Mark all the fringe nodes as skipped so that sumInto doesn't add into these
//   // rows during assembly process
//   for(auto* oinfo: realm_.oversetManager_->oversetInfoVec_) {
//     auto node = oinfo->orphanNode_;
//     HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);
//     skippedRows_.insert(hid);
//   } 
//   printf("Done %s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
// 	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
// }

void
HypreUVWLinearSystem::buildDirichletNodeGraph(
  const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
#endif

  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  // Grab nodes regardless of whether they are owned or shared
  const stk::mesh::Selector sel = stk::mesh::selectUnion(parts);
  const auto& bkts = realm_.get_buckets(
    stk::topology::NODE_RANK, sel);

  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=1;
  std::vector<HypreIntType> nodeCount(numRows_);
  std::fill(nodeCount.begin(), nodeCount.end(), 0);

  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);
      skippedRows_.insert(hid);
      index++;

      /* augment the counter */
      nodeCount[hid]++;
    }
  }

  /*inclusive scan ... this vector is going to be used to give a hint of where to write in the list
    depending on the hid of the first node */
  std::vector<HypreIntType> nodeStart(numRows_+1);
  std::fill(nodeStart.begin(), nodeStart.end(), 0);
  for (unsigned i=0; i<numRows_; ++i) {
    nodeStart[i+1] = nodeStart[i] + nodeCount[i];
  }

  /* save these */
  partitionNodeStart_.push_back(nodeStart);
  partitionCount_.push_back(index);
  count_.push_back(count);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
#endif
}

void
HypreUVWLinearSystem::buildDirichletNodeGraph(
  const std::vector<stk::mesh::Entity>& nodeList)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
#endif

  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=1;
  std::vector<HypreIntType> nodeCount(numRows_);
  std::fill(nodeCount.begin(), nodeCount.end(), 0);

  for (const auto& node: nodeList) {
    HypreIntType hid = get_entity_hypre_id(node);
    skippedRows_.insert(hid);
    index++;

    /* augment the counter */
    nodeCount[hid]++;
  }

  /*inclusive scan ... this vector is going to be used to give a hint of where to write in the list
    depending on the hid of the first node */
  std::vector<HypreIntType> nodeStart(numRows_+1);
  std::fill(nodeStart.begin(), nodeStart.end(), 0);
  for (unsigned i=0; i<numRows_; ++i) {
    nodeStart[i+1] = nodeStart[i] + nodeCount[i];
  }

  /* save these */
  partitionNodeStart_.push_back(nodeStart);
  partitionCount_.push_back(index);
  count_.push_back(count);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
#endif
}

void 
HypreUVWLinearSystem::buildDirichletNodeGraph(const stk::mesh::NgpMesh::ConnectedNodes nodeList) {
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
#endif

  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=1;
  std::vector<HypreIntType> nodeCount(numRows_);
  std::fill(nodeCount.begin(), nodeCount.end(), 0);

  for (unsigned i=0; i<nodeList.size();++i) {
    HypreIntType hid = get_entity_hypre_id(nodeList[i]);
    skippedRows_.insert(hid);
    index++;

    /* augment the counter */
    nodeCount[hid]++;
  }

  /*inclusive scan ... this vector is going to be used to give a hint of where to write in the list
    depending on the hid of the first node */
  std::vector<HypreIntType> nodeStart(numRows_+1);
  std::fill(nodeStart.begin(), nodeStart.end(), 0);
  for (unsigned i=0; i<numRows_; ++i) {
    nodeStart[i+1] = nodeStart[i] + nodeCount[i];
  }

  /* save these */
  partitionNodeStart_.push_back(nodeStart);
  partitionCount_.push_back(index);
  count_.push_back(count);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
#endif
}



}  // nalu
}  // sierra
