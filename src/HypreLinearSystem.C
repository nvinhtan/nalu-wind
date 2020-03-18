// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "HypreLinearSystem.h"
#include "HypreDirectSolver.h"
#include "Realm.h"
#include "EquationSystem.h"
#include "LinearSolver.h"
#include "PeriodicManager.h"
#include "NaluEnv.h"
#include "NonConformalManager.h"
#include "overset/OversetManager.h"
#include "overset/OversetInfo.h"

#include <utils/StkHelpers.h>
#include <utils/CreateDeviceExpression.h>

// NGP Algorithms
#include "ngp_utils/NgpLoopUtils.h"

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

#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>

namespace sierra {
namespace nalu {

HypreLinearSystem::HypreLinearSystem(
  Realm& realm,
  const unsigned numDof,
  EquationSystem* eqSys,
  LinearSolver* linearSolver)
  : LinearSystem(realm, numDof, eqSys, linearSolver),
    name_(eqSys->name_), numAssembles_(0),
    rowFilled_(0),
    rowStatus_(0),
    idBuffer_(0)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
  rows_.clear();
  cols_.clear();
  vals_.clear();
  rhs_rows_.clear();
  rhs_vals_.clear();
#endif

  count_.clear();
  partitionCount_.clear();
  partitionNodeStart_.clear();

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
}

HypreLinearSystem::~HypreLinearSystem()
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

  if (systemInitialized_) {
    HYPRE_IJMatrixDestroy(mat_);
    HYPRE_IJVectorDestroy(rhs_);
    HYPRE_IJVectorDestroy(sln_);
    systemInitialized_ = false;
  }

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
}

void
HypreLinearSystem::beginLinearSystemConstruction()
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

  if (inConstruction_) return;
  inConstruction_ = true;

#ifndef HYPRE_BIGINT
  // Make sure that HYPRE is compiled with 64-bit integer support when running
  // O(~1B) linear systems.
  uint64_t totalRows = (static_cast<uint64_t>(realm_.hypreNumNodes_) *
                        static_cast<uint64_t>(numDof_));
  uint64_t maxHypreSize = static_cast<uint64_t>(std::numeric_limits<HypreIntType>::max());

  if (totalRows > maxHypreSize)
    throw std::runtime_error(
      "The linear system size is greater than what HYPRE is compiled for. "
      "Please recompile with bigint support and link to Nalu");
#endif

  const int rank = realm_.bulk_data().parallel_rank();

  if (rank == 0) {
    iLower_ = realm_.hypreILower_;
  } else {
    iLower_ = realm_.hypreILower_ * numDof_ ;
  }

  iUpper_ = realm_.hypreIUpper_  * numDof_ - 1;
  // For now set column indices the same as row indices
  jLower_ = iLower_;
  jUpper_ = iUpper_;

  // The total number of rows handled by this MPI rank for Hypre
  numRows_ = (iUpper_ - iLower_ + 1);
  // Total number of global rows in the system
  maxRowID_ = realm_.hypreNumNodes_ * numDof_ - 1;

#if 1
  if (numDof_ > 0)
    std::cerr << rank << "\t" << numDof_ << "\t"
              << realm_.hypreILower_ << "\t" << realm_.hypreIUpper_ << "\t"
                << iLower_ << "\t" << iUpper_ << "\t"
                << numRows_ << "\t" << maxRowID_ << std::endl;
#endif
  // Allocate memory for the arrays used to track row types and row filled status.
  rowFilled_.resize(numRows_);
  rowStatus_.resize(numRows_);
  skippedRows_.clear();
  // All nodes start out as NORMAL; "build*NodeGraph" methods might alter the
  // row status to modify behavior of sumInto method.
  for (HypreIntType i=0; i < numRows_; i++)
    rowStatus_[i] = RT_NORMAL;

  auto& bulk = realm_.bulk_data();
  std::vector<const stk::mesh::FieldBase*> fVec{realm_.hypreGlobalId_};

  stk::mesh::copy_owned_to_shared(bulk, fVec);
  stk::mesh::communicate_field_data(bulk.aura_ghosting(), fVec);

  if (realm_.oversetManager_ != nullptr &&
      realm_.oversetManager_->oversetGhosting_ != nullptr)
    stk::mesh::communicate_field_data(
      *realm_.oversetManager_->oversetGhosting_, fVec);

  if (realm_.nonConformalManager_ != nullptr &&
      realm_.nonConformalManager_->nonConformalGhosting_ != nullptr)
    stk::mesh::communicate_field_data(
      *realm_.nonConformalManager_->nonConformalGhosting_, fVec);
  
  if (realm_.periodicManager_ != nullptr &&
      realm_.periodicManager_->periodicGhosting_ != nullptr) {
    realm_.periodicManager_->parallel_communicate_field(realm_.hypreGlobalId_);
    realm_.periodicManager_->periodic_parallel_communicate_field(
      realm_.hypreGlobalId_);
  }

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
}

void
HypreLinearSystem::buildNodeGraph(
  const stk::mesh::PartVector & parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
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

  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      /* fill temporaries */
      index++;
      count = count<0 ? numDof_*numDof_ : count;

      stk::mesh::Entity node = b[k];
      HypreIntType hid = get_entity_hypre_id(node);
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
  printf("Done %s %s %d : name=%s, numDof=%d, numPtsInThisPartition=%lld (from inc scan=%lld), numPartitions=%d, numDataPtsToAssemble=%lld\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,index,nodeStart.back(),(int)partitionCount_.size(),numDataPtsToAssemble());
#endif
}


void
HypreLinearSystem::buildFaceToNodeGraph(const stk::mesh::PartVector & parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
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
      count = count<0 ? (HypreIntType)(numNodes*numNodes*numDof_*numDof_) : count;

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
  printf("Done %s %s %d : name=%s, numDof=%d, numPtsInThisPartition=%lld (from inc scan=%lld), numPartitions=%d, numDataPtsToAssemble=%lld\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,index,nodeStart.back(),(int)partitionCount_.size(),numDataPtsToAssemble());
#endif
}

void
HypreLinearSystem::buildEdgeToNodeGraph(const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
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
      count = count<0 ? (HypreIntType)(numNodes*numNodes*numDof_*numDof_) : count;

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
  printf("Done %s %s %d : name=%s, numDof=%d, numPtsInThisPartition=%lld (from inc scan=%lld), numPartitions=%d, numDataPtsToAssemble=%lld\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,index,nodeStart.back(),(int)partitionCount_.size(),numDataPtsToAssemble());
#endif
}

void
HypreLinearSystem::buildElemToNodeGraph(const stk::mesh::PartVector & parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
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
      count = count<0 ? (HypreIntType)(numNodes*numNodes*numDof_*numDof_) : count;

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
  printf("Done %s %s %d : name=%s, numDof=%d, numPtsInThisPartition=%lld (from inc scan=%lld), numPartitions=%d, numDataPtsToAssemble=%lld\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,index,nodeStart.back(),(int)partitionCount_.size(),numDataPtsToAssemble());
#endif
}

void
HypreLinearSystem::buildFaceElemToNodeGraph(
  const stk::mesh::PartVector & parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
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
      const unsigned nn = numNodes*numNodes*numDof_*numDof_;
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
  printf("Done %s %s %d : name=%s, numDof=%d, numPtsInThisPartition=%lld (from inc scan=%lld), numPartitions=%d, numDataPtsToAssemble=%lld\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,index,nodeStart.back(),(int)partitionCount_.size(),numDataPtsToAssemble());
#endif
}

void
HypreLinearSystem::buildReducedElemToNodeGraph(
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
HypreLinearSystem::buildNonConformalNodeGraph(
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
HypreLinearSystem::buildOversetNodeGraph(
  const stk::mesh::PartVector&)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
#endif

  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  // Mark all the fringe nodes as skipped so that sumInto doesn't add into these
  // rows during assembly process
  for(auto* oinfo: realm_.oversetManager_->oversetInfoVec_) {
    auto node = oinfo->orphanNode_;
    HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);
    skippedRows_.insert(hid * numDof_);
  } 

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
#endif
}

void
HypreLinearSystem::buildDirichletNodeGraph(
  const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
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
      skippedRows_.insert(hid * numDof_);
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
  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
#endif
}

void
HypreLinearSystem::buildDirichletNodeGraph(
  const std::vector<stk::mesh::Entity>& nodeList)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
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
    skippedRows_.insert(hid * numDof_);
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
  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
#endif
}

void 
HypreLinearSystem::buildDirichletNodeGraph(const stk::mesh::NgpMesh::ConnectedNodes nodeList) {
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
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
    skippedRows_.insert(hid * numDof_);
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
  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
#endif
}

void
HypreLinearSystem::finalizeLinearSystem()
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

  ThrowRequire(inConstruction_);
  inConstruction_ = false;

  // Prepare for matrix assembly and set all entry flags to "unfilled"
  for (HypreIntType i=0; i < numRows_; i++)
    rowFilled_[i] = RS_UNFILLED;

  finalizeSolver();

  /* create these mappings */
  fill_entity_to_row_mapping();

  // Set flag to indicate whether rows must be skipped during normal sumInto
  // process. For this to be activated, the linear system must have Dirichlet or
  // overset rows and they must be present on this processor
  if (hasSkippedRows_ && !skippedRows_.empty())
    checkSkippedRows_ = true;

  // At this stage the LHS and RHS data structures are ready for
  // sumInto/assembly.
  systemInitialized_ = true;

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
}

void
HypreLinearSystem::finalizeSolver()
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

  MPI_Comm comm = realm_.bulk_data().parallel();
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  HYPRE_IJMatrixCreate(comm, iLower_, iUpper_, jLower_, jUpper_, &mat_);
  HYPRE_IJMatrixSetObjectType(mat_, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &rhs_);
  HYPRE_IJVectorSetObjectType(rhs_, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(rhs_);
  HYPRE_IJVectorGetObject(rhs_, (void**)&(solver->parRhs_));

  HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &sln_);
  HYPRE_IJVectorSetObjectType(sln_, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(sln_);
  HYPRE_IJVectorGetObject(sln_, (void**)&(solver->parSln_));

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
}


void HypreLinearSystem::fill_entity_to_row_mapping()
{
  const stk::mesh::BulkData& bulk = realm_.bulk_data();
  stk::mesh::Selector selector = bulk.mesh_meta_data().universal_part() & !(realm_.get_inactive_selector());
  entityToLID_ = EntityToHypreIntTypeView("entityToRowLID",bulk.get_size_of_entity_index_space());

  const stk::mesh::BucketVector& nodeBuckets = realm_.get_buckets(stk::topology::NODE_RANK, selector);
  for(const stk::mesh::Bucket* bptr : nodeBuckets) {
    const stk::mesh::Bucket& b = *bptr;
    for(size_t i=0; i<b.size(); ++i) {
      stk::mesh::Entity node = b[i];
      const auto naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);
      const auto mnode = bulk.get_entity(stk::topology::NODE_RANK, naluId);
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, mnode);
      entityToLID_[node.local_offset()] = hid;
    }
  }
}

void
HypreLinearSystem::loadComplete()
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

  std::vector<void *> rhs(numDof_);
  for (unsigned i=0; i<numDof_; ++i) rhs[i] = (void*)(&rhs_);

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
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
      rows_.push_back(lid);
      cols_.push_back(lid);
      vals_.push_back(setval);
      rhs_rows_[0].push_back(lid);
      rhs_vals_[0].push_back(0.0);
#endif
    }
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
  
  sprintf(fname,"%s_rhsRowIndices%d_0.bin",name_.c_str(),numAssembles_);
  std::ofstream hrhsfile(fname, std::ios::out | std::ios::binary);
  hrhsfile.write((char*)(rhs_rows_[0].data()), rhs_rows_[0].size() * sizeof(HypreIntType));
  hrhsfile.close();

  sprintf(fname,"%s_rhsValues%d_0.bin",name_.c_str(),numAssembles_);
  std::ofstream hrhsvfile(fname, std::ios::out | std::ios::binary);
  hrhsvfile.write((char*)(rhs_vals_[0].data()), rhs_vals_[0].size() * sizeof(double));
  hrhsvfile.close();
  
  
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
HypreLinearSystem::dumpHypreMatrix()
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP    
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag((HYPRE_ParCSRMatrix) (solver->parMat_));
  HYPRE_Int * hr = (HYPRE_Int *) hypre_CSRMatrixI(diag);
  HYPRE_Int * hc = (HYPRE_Int *) hypre_CSRMatrixJ(diag);
  double * hd = (double *) hypre_CSRMatrixData(diag);
  HYPRE_Int num_rows = diag->num_rows;
  HYPRE_Int num_nonzeros = diag->num_nonzeros;
    
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

  sprintf(fname,"%s_HypreRows%s%d.bin",name_.c_str(),extension.c_str(),numAssembles_);
  std::ofstream hrfile(fname, std::ios::out | std::ios::binary);
  std::vector<HypreIntType> tmp(num_rows+1);
  for (int i=0; i<num_rows+1; ++i) { tmp[i] = (HypreIntType)hr[i]; }
  hrfile.write((char*)&tmp[0], (num_rows+1) * sizeof(HypreIntType));
  hrfile.close();
  
  sprintf(fname,"%s_HypreCols%s%d.bin",name_.c_str(),extension.c_str(),numAssembles_);
  std::ofstream hcfile(fname, std::ios::out | std::ios::binary);
  tmp.resize(num_nonzeros);
  for (int i=0; i<num_nonzeros; ++i) { tmp[i] = (HypreIntType)hc[i]; }
  hcfile.write((char*)&tmp[0], num_nonzeros * sizeof(HypreIntType));
  hcfile.close();
  
  sprintf(fname,"%s_HypreData%s%d.bin",name_.c_str(),extension.c_str(),numAssembles_);
  std::ofstream hdfile(fname, std::ios::out | std::ios::binary);
  hdfile.write((char*)&hd[0], num_nonzeros * sizeof(double));
  hdfile.close();
  
  std::vector<HypreIntType> hmetaData(0);
  hmetaData.push_back((HypreIntType)num_rows);
  hmetaData.push_back((HypreIntType)num_nonzeros);
  sprintf(fname,"%s_HypreMetaData%s%d.bin",name_.c_str(),extension.c_str(),numAssembles_);
  std::ofstream hmdfile(fname, std::ios::out | std::ios::binary);
  long pos = hmdfile.tellp();
  int size = sizeof(HypreIntType);
  hmdfile.write((char *)&size, 4);
  hmdfile.seekp(pos+4);
  hmdfile.write((char*)&hmetaData[0], hmetaData.size() * sizeof(HypreIntType));
  hmdfile.close();

#endif

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
}


void
HypreLinearSystem::dumpHypreRhs()
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP    
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag((HYPRE_ParCSRMatrix) (solver->parMat_));
  HYPRE_Int num_rows = diag->num_rows;

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
    
  double * local_data = hypre_VectorData(hypre_ParVectorLocalVector(solver->parRhs_));
  sprintf(fname,"%s_HypreRHSData%s%d_0.bin",name_.c_str(),extension.c_str(),numAssembles_);
  std::ofstream hrhsfile(fname, std::ios::out | std::ios::binary);
  hrhsfile.write((char*)local_data, num_rows * sizeof(double));
  hrhsfile.close();
#endif

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
}

void
HypreLinearSystem::loadCompleteSolver()
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  HYPRE_IJMatrixAssemble(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  HYPRE_IJVectorAssemble(rhs_);
  HYPRE_IJVectorGetObject(rhs_, (void**)&(solver->parRhs_));

  HYPRE_IJVectorAssemble(sln_);
  HYPRE_IJVectorGetObject(sln_, (void**)&(solver->parSln_));

  solver->comm_ = realm_.bulk_data().parallel();

  /* dump the matrix */
  dumpHypreMatrix();
  dumpHypreRhs();

#ifdef KOKKOS_ENABLE_CUDA

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_LOAD_FROM_CPU

  hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag((HYPRE_ParCSRMatrix) (solver->parMat_));
  HYPRE_Int num_rows = diag->num_rows;
  HYPRE_Int num_nonzeros = diag->num_nonzeros;

  HYPRE_IJMatrixDestroy(mat_);
  HYPRE_IJVectorDestroy(rhs_);

  HYPRE_IJMatrixCreate(solver->comm_, iLower_, iUpper_, jLower_, jUpper_, &mat_);
  HYPRE_IJMatrixSetObjectType(mat_, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  HYPRE_IJVectorCreate(solver->comm_, iLower_, iUpper_, &rhs_);
  HYPRE_IJVectorSetObjectType(rhs_, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(rhs_);
  HYPRE_IJVectorGetObject(rhs_, (void**)&(solver->parRhs_));

  HYPRE_IJMatrixSetConstantValues(mat_, 0.0);
  HYPRE_ParVectorSetConstantValues(solver->parRhs_, 0.0);

  printf("%s %s %d : name=%s, num_rows=%lld, num_nonzeros=%lld\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),num_rows,num_nonzeros);

  std::string dir="/scratch/pmullown/nalu-wind/build_cpu_new/reg_tests/test_files/ablNeutralEdgeHypre/";

  std::vector<HypreIntType> rows(num_rows+1);
  std::vector<HypreIntType> cols(num_nonzeros);
  std::vector<double> data(num_nonzeros);
  std::vector<double> rhs(num_rows);

  char fname[1000];
  sprintf(fname,"%s%s_HypreRows%d.bin",dir.c_str(),name_.c_str(),numAssembles_);
  std::ifstream file1(fname, std::ios::in | std::ios::binary);
  file1.read((char*)rows.data(), (num_rows+1) * sizeof(HypreIntType));
  file1.close();
  
  sprintf(fname,"%s%s_HypreCols%d.bin",dir.c_str(),name_.c_str(),numAssembles_);
  std::ifstream file2(fname, std::ios::in | std::ios::binary);
  file2.read((char*)cols.data(), (num_nonzeros) * sizeof(HypreIntType));
  file2.close();
  
  sprintf(fname,"%s%s_HypreData%d.bin",dir.c_str(),name_.c_str(),numAssembles_);
  std::ifstream file3(fname, std::ios::in | std::ios::binary);
  file3.read((char*)data.data(), (num_nonzeros) * sizeof(double));
  file3.close();
  
  sprintf(fname,"%s%s_HypreRHSData%d_0.bin",dir.c_str(),name_.c_str(),numAssembles_);
  std::ifstream file4(fname, std::ios::in | std::ios::binary);
  file4.read((char*)rhs.data(), (num_rows) * sizeof(double));
  file4.close();

  std::vector<HypreIntType> row_indices(num_rows);
  std::vector<HypreIntType> row_counts(num_rows);
  for (int i=0; i<num_rows; ++i) {
    row_indices[i] = (HypreIntType)i;
    row_counts[i] = (HypreIntType)(rows[i+1]-rows[i]);
  }
  HYPRE_IJMatrixSetValues(mat_, num_rows, row_counts.data(), row_indices.data(), cols.data(), data.data());  
  HYPRE_IJVectorSetValues(rhs_, num_rows, row_indices.data(), rhs.data());

  HYPRE_IJMatrixAssemble(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  HYPRE_IJVectorAssemble(rhs_);
  HYPRE_IJVectorGetObject(rhs_, (void**)&(solver->parRhs_));

#endif

#endif

  // Set flag to indicate zeroSystem that the matrix must be reinitialized
  // during the next invocation.
  matrixAssembled_ = true;

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
}

void
HypreLinearSystem::zeroSystem()
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("\n\nZero System\n%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  // It is unsafe to call IJMatrixInitialize multiple times without intervening
  // call to IJMatrixAssemble. This occurs during the first outer iteration (of
  // first timestep in static application and every timestep in moving mesh
  // applications) when the data structures have been created but never used and
  // zeroSystem is called for a reset. Include a check to ensure we only
  // initialize if it was previously assembled.
  
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
  rows_.clear();
  cols_.clear();
  vals_.clear();
  rhs_rows_.clear();
  rhs_vals_.clear();
  rhs_rows_.resize(1);
  rhs_vals_.resize(1);
  rhs_rows_[0].resize(0);
  rhs_vals_[0].resize(0);
#endif

  if (matrixAssembled_) {
    HYPRE_IJMatrixInitialize(mat_);
    HYPRE_IJVectorInitialize(rhs_);
    HYPRE_IJVectorInitialize(sln_);

    // Set flag to false until next invocation of IJMatrixAssemble in loadComplete
    matrixAssembled_ = false;
  }

  HYPRE_IJMatrixSetConstantValues(mat_, 0.0);
  HYPRE_ParVectorSetConstantValues(solver->parRhs_, 0.0);
  HYPRE_ParVectorSetConstantValues(solver->parSln_, 0.0);

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

sierra::nalu::CoeffApplier* HypreLinearSystem::get_coeff_applier()
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

    HypreIntTypeView partitionCountView = HypreIntTypeView("partitionCount",numPartitions);

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
    Kokkos::deep_copy(partition_node_start, partition_node_start_host);

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
    HypreIntType maxRowID = realm_.hypreNumNodes_ * numDof_ - 1;
    
    hostCoeffApplier.reset(new HypreLinSysCoeffApplier(numDof_, numPartitions, maxRowID,
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

  return LinearSystem::get_coeff_applier();

#endif
}

/********************************************************************************************************/
/*                     Beginning of HypreLinSysCoeffApplier implementations                             */
/********************************************************************************************************/
HypreLinearSystem::HypreLinSysCoeffApplier::HypreLinSysCoeffApplier(unsigned numDof, 
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
  : numDof_(numDof), numPartitions_(numPartitions), maxRowID_(maxRowID),
    iLower_(iLower), iUpper_(iUpper), jLower_(jLower), jUpper_(jUpper),
    mat_partition_start_(mat_partition_start), mat_count_(mat_count), numMatPtsToAssembleTotal_(numMatPtsToAssembleTotal),
    rhs_partition_start_(rhs_partition_start), rhs_count_(rhs_count), numRhsPtsToAssembleTotal_(numRhsPtsToAssembleTotal),
    partition_node_start_(partition_node_start), entityToLID_(entityToLID), skippedRowsMap_(skippedRowsMap),
    devicePointer_(nullptr)
{
  /* The total number of rows handled by this MPI rank for Hypre */
  numRows_ = (iUpper_ - iLower_ + 1);
  
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : numDof_=%d\n",__FILE__,__FUNCTION__,__LINE__,numDof_);
#endif

  /* This 2D array gets atomically incremented each time we read the hid of the first
     node in each group of entities */
  partition_node_count_ = HypreIntTypeView2D("partition_node_count_", numRows_, numPartitions_);
  Kokkos::deep_copy(partition_node_count_, 0);

  /* get host copies of these */
  mat_partition_start_host = Kokkos::create_mirror_view(mat_partition_start_);
  rhs_partition_start_host = Kokkos::create_mirror_view(rhs_partition_start_);
  mat_count_host = Kokkos::create_mirror_view(mat_count_);
  rhs_count_host = Kokkos::create_mirror_view(rhs_count_);
  Kokkos::deep_copy(mat_partition_start_host, mat_partition_start_);
  Kokkos::deep_copy(rhs_partition_start_host, rhs_partition_start_);
  Kokkos::deep_copy(mat_count_host, mat_count_);
  Kokkos::deep_copy(rhs_count_host, rhs_count_);

  /* storage for the matrix lists */
  rows_ = HypreIntTypeView("rows_",numMatPtsToAssembleTotal_ + numRows_);
  cols_ = HypreIntTypeView("cols_",numMatPtsToAssembleTotal_ + numRows_);
  vals_ = DoubleView("vals_",numMatPtsToAssembleTotal_ + numRows_);
  Kokkos::deep_copy(rows_, -1);
  Kokkos::deep_copy(cols_, -1);
  Kokkos::deep_copy(vals_, 0);

  /* create mirrors */
  rows_host = Kokkos::create_mirror_view(rows_);
  cols_host = Kokkos::create_mirror_view(cols_);
  vals_host = Kokkos::create_mirror_view(vals_);

  /* storage for the rhs lists */
  rhs_rows_ = HypreIntTypeView2D("rhs_rows_", numRhsPtsToAssembleTotal_ + numRows_, numDof_);
  rhs_vals_ = DoubleView2D("rhs_vals_", numRhsPtsToAssembleTotal_ + numRows_, numDof_);
  Kokkos::deep_copy(rhs_rows_, -1);
  Kokkos::deep_copy(rhs_vals_, 0);

  /* create mirrors */
  rhs_rows_host = Kokkos::create_mirror_view(rhs_rows_);
  rhs_vals_host = Kokkos::create_mirror_view(rhs_vals_);

  /* initialize the row filled status vector */
  row_filled_ = Kokkos::View<RowFillStatus*>("row_filled_",numRows_);  
  Kokkos::deep_copy(row_filled_, RS_UNFILLED);
  row_filled_host = Kokkos::create_mirror_view(row_filled_);

  /* This is for the final cleanup step */
  mat_partition_total_ = HypreIntTypeViewScalar("mat_partition_total_");
  Kokkos::deep_copy(mat_partition_total_, numMatPtsToAssembleTotal_);

  /* This is for the final cleanup step */
  rhs_partition_total_ = HypreIntTypeViewScalar("rhs_partition_total_");
  Kokkos::deep_copy(rhs_partition_total_, numRhsPtsToAssembleTotal_);

  /* define the partition index */
  partition_index_ = HypreIntTypeViewScalar("partition_index_");
  Kokkos::deep_copy(partition_index_, -1);
  partition_index_host = Kokkos::create_mirror_view(partition_index_);

  /* check skipped rows */
  checkSkippedRows_ = HypreIntTypeViewScalar("checkSkippedRows_");
  checkSkippedRows_() = skippedRowsMap_.size()>0 ? 1 : 0;
  
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : numDof_=%d\n",__FILE__,__FUNCTION__,__LINE__,numDof_);
#endif
}

KOKKOS_FUNCTION
void
HypreLinearSystem::HypreLinSysCoeffApplier::sum_into(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  unsigned numDof, HypreIntType iLower, HypreIntType iUpper,
  HypreIntType partitionIndex) {
  
  HypreIntType hid0 = entityToLID_[entities[0].local_offset()];
  HypreIntType counter = Kokkos::atomic_fetch_add(&partition_node_count_(hid0, partitionIndex), 1); 
  HypreIntType nodeStart = partition_node_start_(hid0, partitionIndex);
  HypreIntType matIndex = mat_partition_start_(partitionIndex) + (nodeStart + counter)*mat_count_(partitionIndex);
  HypreIntType rhsIndex = rhs_partition_start_(partitionIndex) + (nodeStart + counter)*rhs_count_(partitionIndex);

  HypreIntType numRows = numEntities * numDof;
  for(unsigned i=0; i<numEntities; i++) {
    HypreIntType hid = entityToLID_[entities[i].local_offset()];
    for(unsigned d=0; d<numDof; ++d) {
      unsigned lid = i*numDof + d;
      localIds[lid] = hid*numDof + d;
    }
  }

  for (unsigned i=0; i<numEntities; i++) {
    int ix = i * numDof;
    HypreIntType hid = localIds[ix];
    if (checkSkippedRows_()) {
      if (skippedRowsMap_.exists(hid)) continue;
    }
    for (unsigned d=0; d<numDof; d++) {
      unsigned ir = ix + d;
      HypreIntType lid = localIds[ir];
      const double* cur_lhs = &lhs(ir, 0);

      /* fill the matrix values */
      for (unsigned k=0; k<numRows; k++) {
	rows_(matIndex+i*numRows+k) = lid;
	cols_(matIndex+i*numRows+k) = localIds[k];
	vals_(matIndex+i*numRows+k) = cur_lhs[k];
      }
      /* fill the right hand side values */
      rhs_rows_(rhsIndex+i,0) = lid;
      rhs_vals_(rhsIndex+i,0) = rhs[ir];

      if ((lid >= iLower) && (lid <= iUpper))
        row_filled_(lid - iLower) = RS_FILLED;
    }
  }
}


KOKKOS_FUNCTION
void
HypreLinearSystem::HypreLinSysCoeffApplier::operator()(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<int*, DeviceShmem>&,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const char* /*trace_tag*/)
{
  sum_into(numEntities,entities,localIds,rhs,lhs,numDof_,iLower_,iUpper_,partition_index_());
}


void HypreLinearSystem::HypreLinSysCoeffApplier::dumpData(std::string name, const int di) {
  int matCount = (int) numMatPtsToAssembleTotal_ + numRows_;
  int rhsCount = (int) numRhsPtsToAssembleTotal_ + numRows_;
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("matCount=%d, rhsCount=%d\n",matCount,rhsCount);
#endif

  Kokkos::deep_copy(rows_host, rows_);
  Kokkos::deep_copy(cols_host, cols_);
  Kokkos::deep_copy(vals_host, vals_);
  Kokkos::deep_copy(rhs_rows_host, rhs_rows_);
  Kokkos::deep_copy(rhs_vals_host, rhs_vals_);

  char fname[50];
  sprintf(fname,"%s_CoeffApplier_rowIndices%d.bin",name.c_str(),di);
  std::ofstream rfile(fname, std::ios::out | std::ios::binary);
  rfile.write((char*)&rows_host(0), matCount * sizeof(HypreIntType));
  rfile.close();
    
  sprintf(fname,"%s_CoeffApplier_colIndices%d.bin",name.c_str(),di);
  std::ofstream cfile(fname, std::ios::out | std::ios::binary);
  cfile.write((char*)&cols_host(0), matCount * sizeof(HypreIntType));
  cfile.close();
    
  sprintf(fname,"%s_CoeffApplier_values%d.bin",name.c_str(),di);
  std::ofstream vfile(fname, std::ios::out | std::ios::binary);
  vfile.write((char*)&vals_host(0), matCount * sizeof(double));
  vfile.close();

  for (unsigned j=0; j<numDof_; ++j) {
    sprintf(fname,"%s_CoeffApplier_rhsRowIndices%d_%u.bin",name.c_str(),di,j);
    std::ofstream rrfile(fname, std::ios::out | std::ios::binary);
    rrfile.write((char*)&rhs_rows_host(0,j), rhsCount * sizeof(HypreIntType));
    rrfile.close();
    
    sprintf(fname,"%s_CoeffApplier_rhsValues%d_%u.bin",name.c_str(),di,j);
    std::ofstream vvfile(fname, std::ios::out | std::ios::binary);
    vvfile.write((char*)&rhs_vals_host(0,j), rhsCount * sizeof(double));
    vvfile.close();
  }

  std::vector<HypreIntType> metaData(0);
  metaData.push_back((HypreIntType)iLower_);
  metaData.push_back((HypreIntType)iUpper_);
  metaData.push_back((HypreIntType)jLower_);
  metaData.push_back((HypreIntType)jUpper_);
  metaData.push_back((HypreIntType)matCount);
  metaData.push_back((HypreIntType)rhsCount);
  sprintf(fname,"%s_CoeffApplier_metaData%d.bin",name.c_str(),di);
  std::ofstream mdfile(fname, std::ios::out | std::ios::binary);
  long pos = mdfile.tellp();
  int size = sizeof(HypreIntType);
  mdfile.write((char *)&size, 4);
  mdfile.seekp(pos+4);
  mdfile.write((char*)&metaData[0], metaData.size() * sizeof(HypreIntType));
  mdfile.close();
}

void
HypreLinearSystem::HypreLinSysCoeffApplier::applyDirichletBCs(Realm & realm, 
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
  std::vector<HypreIntType> trhsRows(0);
  std::vector<double> trhsVals(0);

  NGPDoubleFieldType ngpSolutionField = realm.ngp_field_manager().get_field<double>(solutionField->mesh_meta_data_ordinal());
  NGPDoubleFieldType ngpBCValuesField = realm.ngp_field_manager().get_field<double>(bcValuesField->mesh_meta_data_ordinal());

  ngpSolutionField.sync_to_host();
  ngpBCValuesField.sync_to_host();

  int count = 0;
  for (auto b: bkts) {
    const double* solution = (double*)stk::mesh::field_data(
      *solutionField, *b);
    const double* bcValues = (double*)stk::mesh::field_data(
      *bcValuesField, *b);

    for (unsigned in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm.hypreGlobalId_, node);

      for (unsigned d=0; d<numDof_; d++) {
        HypreIntType lid = hid * numDof_ + d;
        double bcval = bcValues[in*numDof_ + d] - solution[in*numDof_ + d];
	
	if (row_filled_host(lid - iLower_) == RS_FILLED) count++;

	/* fill the mirrored version */
        row_filled_host(lid - iLower_) = RS_FILLED;

	/* fill these temp values */
	tRows.push_back(lid);
	tCols.push_back(lid);
	tVals.push_back(diag_value);
	trhsRows.push_back(lid);
	trhsVals.push_back(bcval);
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

  HypreIntTypeView rr("rr",trhsRows.size());
  HypreIntTypeViewHost rrh  = Kokkos::create_mirror_view(rr);

  DoubleView rv("rv",trhsVals.size());
  DoubleViewHost rvh  = Kokkos::create_mirror_view(rv);

  /* Step 4 : next copy the std::vectors into the host mirrors */
  for (unsigned int i=0; i<tRows.size(); ++i) {
    rh(i) = tRows[i];
    ch(i) = tCols[i];
    vh(i) = tVals[i];
    rrh(i) = trhsRows[i];
    rvh(i) = trhsVals[i];
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
  kokkos_parallel_for("bcHack", tRows.size(), [&] (const unsigned& i) {
      HypreIntType matIndex = mat_partition_start_(partition_index_())+i;
      HypreIntType rhsIndex = rhs_partition_start_(partition_index_())+i;
      rows_(matIndex)=r(i);
      cols_(matIndex)=c(i);
      vals_(matIndex)=v(i);
      rhs_rows_(rhsIndex,0) = rr(i);
      rhs_vals_(rhsIndex,0) = rv(i);
    });

#else

  stk::mesh::MetaData & metaData = realm.meta_data();

  const stk::mesh::Selector selector = (
    metaData.locally_owned_part() &
    stk::mesh::selectUnion(parts) &
    stk::mesh::selectField(*solutionField) &
    !(realm.get_inactive_selector()));

  using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;

  stk::mesh::NgpMesh ngpMesh = realm.ngp_mesh();
  NGPDoubleFieldType ngpSolutionField = realm.ngp_field_manager().get_field<double>(solutionField->mesh_meta_data_ordinal());
  NGPDoubleFieldType ngpBCValuesField = realm.ngp_field_manager().get_field<double>(bcValuesField->mesh_meta_data_ordinal());

  ngpSolutionField.sync_to_device();
  ngpBCValuesField.sync_to_device();

  nalu_ngp::run_entity_algorithm(
    "HypreLinSysCoeffApplier::applyDirichletBCs", ngpMesh, stk::topology::NODE_RANK, selector,
    KOKKOS_LAMBDA(const MeshIndex& meshIdx)
    {
      stk::mesh::Entity entity = (*meshIdx.bucket)[meshIdx.bucketOrd];
      HypreIntType hid = entityToLID_[entity.local_offset()];

      /* This stuff doesn't work any more */
      HypreIntType counter = Kokkos::atomic_fetch_add(&partition_node_count_(hid, partition_index_()), 1); 
      HypreIntType nodeStart = partition_node_start_(hid, partition_index_());
      HypreIntType matIndex = mat_partition_start_(partition_index_()) + (nodeStart + counter)*mat_count_(partition_index_());
      HypreIntType rhsIndex = rhs_partition_start_(partition_index_()) + (nodeStart + counter)*rhs_count_(partition_index_());
      
      double diag_value = 1.0;
      
      for (unsigned d=0; d<numDof_; d++) {
        HypreIntType lid = hid * numDof_ + d;
        const double bc_residual = ngpBCValuesField.get(meshIdx, d) - ngpSolutionField.get(meshIdx, d);
	rows_(matIndex+d) = lid;
	cols_(matIndex+d) = lid;
	vals_(matIndex+d) = diag_value;
	rhs_rows_(rhsIndex,d) = lid;
	rhs_vals_(rhsIndex,d) = bc_residual;
	row_filled_(lid - iLower_) = RS_FILLED;
	
      }
    }
  );

#endif

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d\n",__FILE__,__FUNCTION__,__LINE__);
#endif
}

void
HypreLinearSystem::HypreLinSysCoeffApplier::finishAssembly(void * mat, std::vector<void *> rhs, const int di, std::string name) {

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d\n",__FILE__,__FUNCTION__,__LINE__);
#endif

  /* for some reason, Kokkos::parallel_for with a LAMBDA function does not compile. */
  kokkos_parallel_for("unfilledRows", numRows_, [&] (const unsigned& i) {
      if (row_filled_(i)==RS_UNFILLED) {
	HypreIntType lid = iLower_ + i;
	rows_(mat_partition_total_()+i) = lid;
	cols_(mat_partition_total_()+i) = lid;
	vals_(mat_partition_total_()+i) = 1.0;
	for (unsigned d=0; d<numDof_; ++d) {
	  rhs_rows_(rhs_partition_total_()+i,d) = lid;
	  rhs_vals_(rhs_partition_total_()+i,d) = 0.0;
	}
      }
    }); 

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
  /* dump the data before anyything else */
  dumpData(name, di);
#endif

#ifdef KOKKOS_ENABLE_CUDA

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_LOAD_FROM_CPU_LISTS

  char fname[1000];
  std::string dir="/scratch/pmullown/nalu-wind/build_cpu_new/reg_tests/test_files/ablNeutralEdgeHypre/";

  sprintf(fname,"%s/%s_metaData%d.bin",dir.c_str(),name.c_str(),di);
  std::ifstream file(fname, std::ios::in | std::ios::binary);
  long pos = file.tellg();
  int intTypeSize=4;
  std::vector<HypreIntType> metaData(6);
  file.read((char*)(&intTypeSize), sizeof(int));
  file.seekg(pos+4);
  file.read((char*)metaData.data(), 6 * sizeof(HypreIntType));
  file.close();

  int r0 = metaData[0];
  int c0 = metaData[2];
  int num_rows = metaData[1]+1-r0;
  int num_cols = metaData[3]+1-c0;
  HypreIntType nDataPtsToAssemble = metaData[4];
  HypreIntType nvDataPtsToAssemble = metaData[5];

  HypreIntType * rows = (HypreIntType *)malloc(nDataPtsToAssemble*sizeof(HypreIntType));
  HypreIntType * cols = (HypreIntType *)malloc(nDataPtsToAssemble*sizeof(HypreIntType));
  double * data = (double *)malloc(nDataPtsToAssemble*sizeof(double));
  HypreIntType * rhsRows = (HypreIntType *)malloc(nvDataPtsToAssemble*sizeof(HypreIntType));
  double * rhsData = (double *)malloc(nvDataPtsToAssemble*sizeof(double));

  sprintf(fname,"%s/%s_rowIndices%d.bin",dir.c_str(),name.c_str(),di);
  std::ifstream file1(fname, std::ios::in | std::ios::binary);
  file1.read((char*)rows, nDataPtsToAssemble * sizeof(HypreIntType));
  file1.close();

  sprintf(fname,"%s/%s_colIndices%d.bin",dir.c_str(),name.c_str(),di);
  std::ifstream file2(fname, std::ios::in | std::ios::binary);
  file2.read((char*)cols, nDataPtsToAssemble * sizeof(HypreIntType));
  file2.close();

  sprintf(fname,"%s/%s_values%d.bin",dir.c_str(),name.c_str(),di);
  std::ifstream file3(fname, std::ios::in | std::ios::binary);
  file3.read((char*)data, nDataPtsToAssemble * sizeof(double));
  file3.close();

  sprintf(fname,"%s/%s_rhsRowIndices%d_0.bin",dir.c_str(),name.c_str(),di);
  std::ifstream file4(fname, std::ios::in | std::ios::binary);
  file4.read((char*)rhsRows, nvDataPtsToAssemble * sizeof(HypreIntType));
  file4.close();

  sprintf(fname,"%s/%s_rhsValues%d_0.bin",dir.c_str(),name.c_str(),di);
  std::ifstream file5(fname, std::ios::in | std::ios::binary);
  file5.read((char*)rhsData, nvDataPtsToAssemble * sizeof(double));
  file5.close();

  /**********/
  /* Matrix */
  /**********/

  /* Build the assembler objects */
  if (!MatAssembler_)
    MatAssembler_ = new MatrixAssembler<HypreIntType>(name,true,iLower_,jLower_,numRows_,numRows_,nDataPtsToAssemble);

  MatAssembler_->copySrcDataToDevice(rows, cols, data);
  MatAssembler_->assemble();
  MatAssembler_->copyAssembledCSRMatrixToHost();  
  int * row_offsets = MatAssembler_->getHostRowOffsetsPtr();
  HypreIntType * col_indices = MatAssembler_->getHostColIndicesPtr();
  double * values = MatAssembler_->getHostValuesPtr();

  std::vector<HypreIntType> row_indices(numRows_);
  std::vector<HypreIntType> row_counts(numRows_);
  for (int i=0; i<numRows_; ++i) {
    row_indices[i] = (HypreIntType)i;
    row_counts[i] = (HypreIntType)(row_offsets[i+1]-row_offsets[i]);
  }

  /* Cast these to their types ... ugly */
  HYPRE_IJMatrix hmat = *((HYPRE_IJMatrix *)mat);
  HYPRE_IJMatrixSetValues(hmat, numRows_, row_counts.data(), row_indices.data(), col_indices, values);  

  /********/
  /* Rhs */
  /********/

  /* Build the assembler objects */
  if (!RhsAssembler_)
    RhsAssembler_ = new RhsAssembler<HypreIntType>(name,true,iLower_,numRows_,nvDataPtsToAssemble);

  for (unsigned i=0; i<numDof_; ++i) {
    /* get the src data from the kokkos views */
    RhsAssembler_->copySrcDataToDevice(rhsRows, rhsData);
    RhsAssembler_->assemble();
    RhsAssembler_->copyAssembledRhsVectorToHost();  
    double * values = RhsAssembler_->getHostRhsPtr();
    
    /* Cast these to their types ... ugly */
    HYPRE_IJVector hrhs = *((HYPRE_IJVector *)rhs[i]);
    HYPRE_IJVectorSetValues(hrhs, numRows_, row_indices.data(), values);
  }

  free(rows);
  free(cols);
  free(data);
  free(rhsRows);
  free(rhsData);

#else

  /**********/
  /* Matrix */
  /**********/

  /* Build the assembler objects */
  if (!MatAssembler_)
    MatAssembler_ = new MatrixAssembler<HypreIntType>(name,true,iLower_,jLower_,numRows_,numRows_,numMatPtsToAssembleTotal_+numRows_);

  MatAssembler_->copySrcDataFromKokkos(rows_.data(), cols_.data(), vals_.data());
  MatAssembler_->assemble();
  //MatAssembler_->reorderDLU();
  MatAssembler_->copyAssembledCSRMatrixToHost();  
  int * row_offsets = MatAssembler_->getHostRowOffsetsPtr();
  HypreIntType * col_indices = MatAssembler_->getHostColIndicesPtr();
  double * values = MatAssembler_->getHostValuesPtr();

  std::vector<HypreIntType> row_indices(numRows_);
  std::vector<HypreIntType> row_counts(numRows_);
  for (int i=0; i<numRows_; ++i) {
    row_indices[i] = (HypreIntType)i;
    row_counts[i] = (HypreIntType)(row_offsets[i+1]-row_offsets[i]);
  }

  /* Cast these to their types ... ugly */
  HYPRE_IJMatrix hmat = *((HYPRE_IJMatrix *)mat);
  HYPRE_IJMatrixSetValues(hmat, numRows_, row_counts.data(), row_indices.data(), col_indices, values);  

  /********/
  /* Rhs */
  /********/

  /* Build the assembler objects */
  if (!RhsAssembler_)
    RhsAssembler_ = new RhsAssembler<HypreIntType>(name,true,iLower_,numRows_,numRhsPtsToAssembleTotal_+numRows_);

  for (unsigned i=0; i<numDof_; ++i) {
    /* get the src data from the kokkos views */
    RhsAssembler_->copySrcDataFromKokkos(&rhs_rows_(0,i), &rhs_vals_(0,i));
    RhsAssembler_->assemble();
    RhsAssembler_->copyAssembledRhsVectorToHost();  
    double * values = RhsAssembler_->getHostRhsPtr();
    
    /* Cast these to their types ... ugly */
    HYPRE_IJVector hrhs = *((HYPRE_IJVector *)rhs[i]);
    HYPRE_IJVectorSetValues(hrhs, numRows_, row_indices.data(), values);
  }

#endif //HYPRE_LINEAR_SYSTEM_DEBUG_LOAD_FROM_CPU_LISTS

#endif //KOKKOS_ENABLE_CUDA

  /* Reset after assembly */
  Kokkos::deep_copy(partition_index_, -1);
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d\n",__FILE__,__FUNCTION__,__LINE__);
#endif
}


void
HypreLinearSystem::HypreLinSysCoeffApplier::resetInternalData() {

  if (numPartitions_>0) {
    Kokkos::deep_copy(partition_index_host, partition_index_);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
    printf("%s %s %d : partition_index_=%d\n",__FILE__,__FUNCTION__,__LINE__,(int)partition_index_host());
#endif

    partition_index_host()++;
    partition_index_host() = (partition_index_host()%numPartitions_);
    Kokkos::deep_copy(partition_index_, partition_index_host);

    Kokkos::deep_copy(checkSkippedRows_, 1);
    if (partition_index_host()==0) {
      Kokkos::deep_copy(partition_node_count_, 0);
      Kokkos::deep_copy(row_filled_, RS_UNFILLED);
      Kokkos::deep_copy(rows_, -1);
      Kokkos::deep_copy(cols_, -1);
      Kokkos::deep_copy(vals_, 0.);
      Kokkos::deep_copy(rhs_rows_, -1);
      Kokkos::deep_copy(rhs_vals_, 0.);      
    }

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
    printf("Done %s %s %d : partition_index_=%d\n",__FILE__,__FUNCTION__,__LINE__,(int)partition_index_host());
#endif
  }
}

void HypreLinearSystem::HypreLinSysCoeffApplier::free_device_pointer()
{
#ifdef KOKKOS_ENABLE_CUDA
  if (this != devicePointer_) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
#endif
}

sierra::nalu::CoeffApplier* HypreLinearSystem::HypreLinSysCoeffApplier::device_pointer()
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

/********************************************************************************************************/
/*                           End of HypreLinSysCoeffApplier implementations                             */
/********************************************************************************************************/


void
HypreLinearSystem::sumInto(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const SharedMemView<int*, DeviceShmem>&,
  const SharedMemView<int*, DeviceShmem>&,
  const char*  /* trace_tag */)
{
#ifndef KOKKOS_ENABLE_CUDA
  const size_t n_obj = numEntities;
  HypreIntType numRows = n_obj * numDof_;
  const HypreIntType bufSize = idBuffer_.size();

  ThrowAssertMsg(lhs.span_is_contiguous(), "LHS assumed contiguous");
  ThrowAssertMsg(rhs.span_is_contiguous(), "RHS assumed contiguous");
  if (bufSize < numRows) idBuffer_.resize(numRows);

  for (size_t in=0; in < n_obj; in++) {
    HypreIntType hid = get_entity_hypre_id(entities[in]);
    HypreIntType localOffset = hid * numDof_;
    for (size_t d=0; d < numDof_; d++) {
      size_t lid = in * numDof_ + d;
      idBuffer_[lid] = localOffset + d;
    }
  }

  for (size_t in=0; in < n_obj; in++) {
    int ix = in * numDof_;
    HypreIntType hid = idBuffer_[ix];

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) {
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
	for (size_t d=0; d < numDof_; d++) {
	  for (unsigned j=0; j<numRows; ++j) {
	    rows_.push_back(-1);
	    cols_.push_back(-1);
	    vals_.push_back(0.0);
	  }
	  rhs_rows_[0].push_back(-1);
	  rhs_vals_[0].push_back(0.0);
	}
#endif
	continue;
      }
    }

    for (size_t d=0; d < numDof_; d++) {
      int ir = ix + d;
      HypreIntType lid = idBuffer_[ir];

      const double* cur_lhs = &lhs(ir, 0);
      HYPRE_IJMatrixAddToValues(mat_, 1, &numRows, &lid,
                                &idBuffer_[0], cur_lhs);
      HYPRE_IJVectorAddToValues(rhs_, 1, &lid, &rhs[ir]);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
      for (unsigned j=0; j<numRows; ++j) {
	rows_.push_back(lid);
	cols_.push_back(idBuffer_[j]);
	vals_.push_back(cur_lhs[j]);
      }
      rhs_rows_[0].push_back(lid);
      rhs_vals_[0].push_back(rhs[ir]);
#endif
      
      if ((lid >= iLower_) && (lid <= iUpper_))
        rowFilled_[lid - iLower_] = RS_FILLED;
    }
  }
#endif
}


void
HypreLinearSystem::sumInto(
  const std::vector<stk::mesh::Entity>& entities,
  std::vector<int>&  /* scratchIds */,
  std::vector<double>& scratchVals,
  const std::vector<double>& rhs,
  const std::vector<double>& lhs,
  const char*  /* trace_tag */)
{
  const size_t n_obj = entities.size();
  HypreIntType numRows = n_obj * numDof_;
  const HypreIntType bufSize = idBuffer_.size();

  ThrowAssert(numRows == static_cast<HypreIntType>(rhs.size()));
  ThrowAssert(numRows*numRows == static_cast<HypreIntType>(lhs.size()));

  if (bufSize < numRows) idBuffer_.resize(numRows);

  for (size_t in=0; in < n_obj; in++) {
    HypreIntType hid = get_entity_hypre_id(entities[in]);
    HypreIntType localOffset = hid * numDof_;
    for (size_t d=0; d < numDof_; d++) {
      size_t lid = in * numDof_ + d;
      idBuffer_[lid] = localOffset + d;
    }
  }

  for (size_t in=0; in < n_obj; in++) {
    int ix = in * numDof_;
    HypreIntType hid = idBuffer_[ix];

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) {
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
	for (size_t d=0; d < numDof_; d++) {
	  for (unsigned j=0; j<numRows; ++j) {
	    rows_.push_back(-1);
	    cols_.push_back(-1);
	    vals_.push_back(0.0);
	  }
	  rhs_rows_[0].push_back(-1);
	  rhs_vals_[0].push_back(0.0);
	}
#endif
	continue;
      }
    }

    for (size_t d=0; d < numDof_; d++) {
      int ir = ix + d;
      HypreIntType lid = idBuffer_[ir];

      for (int c=0; c < numRows; c++)
        scratchVals[c] = lhs[ir * numRows + c];

      HYPRE_IJMatrixAddToValues(mat_, 1, &numRows, &lid,
                                &idBuffer_[0], &scratchVals[0]);
      HYPRE_IJVectorAddToValues(rhs_, 1, &lid, &rhs[ir]);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
      for (unsigned j=0; j<numRows; ++j) {
	rows_.push_back(lid);
	cols_.push_back(idBuffer_[j]);
	vals_.push_back(scratchVals[j]);
      }
      rhs_rows_[0].push_back(lid);
      rhs_vals_[0].push_back(rhs[ir]);
#endif

      if ((lid >= iLower_) && (lid <= iUpper_))
        rowFilled_[lid - iLower_] = RS_FILLED;
    }
  }
}

void
HypreLinearSystem::applyDirichletBCs(
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

      for (size_t d=0; d<numDof_; d++) {
        HypreIntType lid = hid * numDof_ + d;
        double bcval = bcValues[in*numDof_ + d] - solution[in*numDof_ + d];

        HYPRE_IJMatrixSetValues(mat_, 1, &ncols, &lid, &lid, &diag_value);
        HYPRE_IJVectorSetValues(rhs_, 1, &lid, &bcval);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
	rows_.push_back(lid);
	cols_.push_back(lid);
	vals_.push_back(diag_value);
	rhs_rows_[0].push_back(lid);
	rhs_vals_[0].push_back(bcval);
#endif

        rowFilled_[lid - iLower_] = RS_FILLED;
      }
    }
  }
#endif
  adbc_time += NaluEnv::self().nalu_time();

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
}

HypreIntType
HypreLinearSystem::get_entity_hypre_id(const stk::mesh::Entity& node)
{
  auto& bulk = realm_.bulk_data();
  const auto naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);
  const auto mnode = bulk.get_entity(stk::topology::NODE_RANK, naluId);
#ifndef NDEBUG
  if (!bulk.is_valid(node))
    throw std::runtime_error("BAD STK NODE");
#endif
  HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, mnode);

#ifndef NDEBUG
  HypreIntType chk = ((hid+1) * numDof_ - 1);
  if ((hid < 0) || (chk > maxRowID_)) {
    std::cerr << bulk.parallel_rank() << "\t"
              << hid << "\t" << iLower_ << "\t" << iUpper_ << std::endl;
    throw std::runtime_error("BAD STK to hypre conversion");
  }
#endif

  return hid;
}

int
HypreLinearSystem::solve(stk::mesh::FieldBase* linearSolutionField)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif

  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(
    linearSolver_);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
    const std::string matFile = eqSysName_ + ".IJM." + writeCounter + ".mat";
    const std::string rhsFile = eqSysName_ + ".IJV." + writeCounter + ".rhs";
    HYPRE_IJMatrixPrint(mat_, matFile.c_str());
    HYPRE_IJVectorPrint(rhs_, rhsFile.c_str());
  }

  int iters = 0;
  double finalResidNorm = 0.0;

  // Call solve
  int status = 0;

  status = solver->solve(iters, finalResidNorm, realm_.isFinalOuterIter_);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
    const std::string slnFile = eqSysName_ + ".IJV." + writeCounter + ".sln";
    HYPRE_IJVectorPrint(sln_, slnFile.c_str());
    ++eqSys_->linsysWriteCounter_;
  }

  double norm2 = copy_hypre_to_stk(linearSolutionField);
  sync_field(linearSolutionField);

  linearSolveIterations_ = iters;
  // Hypre provides relative residuals not the final residual, so multiply by
  // the non-linear residual to obtain a final residual that is comparable to
  // what is reported by TpetraLinearSystem. Note that this assumes the initial
  // solution vector is set to 0 at the start of linear iterations.
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("norm2 = %1.16g\n",norm2);
#endif
  linearResidual_ = finalResidNorm * norm2;
  nonLinearResidual_ = realm_.l2Scaling_ * norm2;

  if (eqSys_->firstTimeStepSolve_)
    firstNonLinearResidual_ = nonLinearResidual_;

  scaledNonLinearResidual_ =
    nonLinearResidual_ /
    std::max(std::numeric_limits<double>::epsilon(), firstNonLinearResidual_);

  if (provideOutput_) {
    const int nameOffset = eqSysName_.length() + 8;
    NaluEnv::self().naluOutputP0()
      << std::setw(nameOffset) << std::right << eqSysName_
      << std::setw(32 - nameOffset) << std::right << iters << std::setw(18)
      << std::right << linearResidual_ << std::setw(15) << std::right
      << nonLinearResidual_ << std::setw(14) << std::right
      << scaledNonLinearResidual_ << std::endl;
  }

  eqSys_->firstTimeStepSolve_ = false;
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("Done %s %s %d : name=%s\n\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
#endif
  return status;
}

double
HypreLinearSystem::copy_hypre_to_stk(
  stk::mesh::FieldBase* stkField)
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();
  const auto sel = stk::mesh::selectField(*stkField)
    & meta.locally_owned_part()
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  const auto& bkts = bulk.get_buckets(
    stk::topology::NODE_RANK, sel);

  double lclnorm2 = 0.0;
  double rhsVal = 0.0;

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  int c=0, printCount=0;  
  bool badRhs = false;
  bool badSln = false;
#endif

  for (auto b: bkts) {
    double* field = (double*) stk::mesh::field_data(*stkField, *b);
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = get_entity_hypre_id(node);

      for (size_t d=0; d < numDof_; d++) {
        HypreIntType lid = hid * numDof_ + d;
        int sid = in * numDof_ + d;
        HYPRE_IJVectorGetValues(sln_, 1, &lid, &field[sid]);
        HYPRE_IJVectorGetValues(rhs_, 1, &lid, &rhsVal);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
	if (!std::isfinite(field[sid]) && badSln==false) badSln=true;
	if (badSln && printCount<20) { printf("%d : lid=%lld, sln=%1.16lf, rhs=%1.16lf\n",c,lid,field[sid],rhsVal); printCount++; }
	c++;

	if (!std::isfinite(rhsVal) && badRhs==false) badRhs=true;
#endif
        lclnorm2 += rhsVal * rhsVal;
      }
    }
  }
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  printf("%s %s %d : %s badRhs=%d, badSln=%d\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str(),(int)badRhs,(int)badSln);
#endif


#ifdef HYPRE_LINEAR_SYSTEM_DEBUG_DUMP
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);
  double * local_data = hypre_VectorData(hypre_ParVectorLocalVector(solver->parSln_));

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
  sprintf(fname,"%s_HypreSolution%s%d.bin",name_.c_str(),extension.c_str(),numAssembles_);

  std::ofstream slnfile(fname, std::ios::out | std::ios::binary);
  slnfile.write((char*)local_data, numRows_ * sizeof(double));
  slnfile.close();
#endif

  NGPDoubleFieldType ngpField = realm_.ngp_field_manager().get_field<double>(stkField->mesh_meta_data_ordinal());  
  ngpField.modify_on_host();
  ngpField.sync_to_device();

  double gblnorm2 = 0.0;
  stk::all_reduce_sum(bulk.parallel(), &lclnorm2, &gblnorm2, 1);

  return std::sqrt(gblnorm2);
}

}  // nalu
}  // sierra
