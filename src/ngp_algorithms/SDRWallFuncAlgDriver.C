/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "ngp_algorithms/SDRWallFuncAlgDriver.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "PeriodicManager.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/FieldBLAS.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_ngp/NgpFieldParallel.hpp"

namespace sierra {
namespace nalu {

SDRWallFuncAlgDriver::SDRWallFuncAlgDriver(
  Realm& realm
) : NgpAlgDriver(realm)
{}

void SDRWallFuncAlgDriver::pre_work()
{
  const auto& ngpMesh = realm_.ngp_mesh();
  auto& bcsdr = nalu_ngp::get_ngp_field(realm_.mesh_info(), "wall_model_sdr_bc");
  auto& wallArea = nalu_ngp::get_ngp_field(realm_.mesh_info(), "assembled_wall_area_sdr");

  bcsdr.set_all(ngpMesh, 0.0);
  wallArea.set_all(ngpMesh, 0.0);
}

void SDRWallFuncAlgDriver::post_work()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<ngp::Mesh>::MeshIndex;
  const auto& ngpMesh = realm_.ngp_mesh();

  auto& bcsdr = nalu_ngp::get_ngp_field(realm_.mesh_info(), "wall_model_sdr_bc");
  auto& wallArea = nalu_ngp::get_ngp_field(realm_.mesh_info(), "assembled_wall_area_sdr");
  auto& sdr = nalu_ngp::get_ngp_field(realm_.mesh_info(), "specific_dissipation_rate");
  auto& sdrWallBC = nalu_ngp::get_ngp_field(realm_.mesh_info(), "sdr_bc");

  bcsdr.modify_on_device();
  wallArea.modify_on_device();

  // Parallel synchronization
  const std::vector<NGPDoubleFieldType*> fields {&bcsdr, &wallArea};
  const bool doFinalSyncToDevice = true;
  ngp::parallel_sum(realm_.bulk_data(), fields, doFinalSyncToDevice);

  auto* bcsdrF = realm_.meta_data().get_field(
    stk::topology::NODE_RANK, "wall_model_sdr_bc");
  if (realm_.hasPeriodic_) {
    // Periodic synchronization
    const unsigned nComponents = 1;
    const bool bypassFieldCheck = false;
    const bool addMirrorValues = true;
    const bool setMirrorValues = true;

    auto* wallAreaF = realm_.meta_data().get_field(
      stk::topology::NODE_RANK, "assembled_wall_area_sdr");

    auto* periodicMgr = realm_.periodicManager_;
    periodicMgr->ngp_apply_constraints(
      bcsdrF, nComponents, bypassFieldCheck, addMirrorValues, setMirrorValues);
    periodicMgr->ngp_apply_constraints(
      wallAreaF, nComponents, bypassFieldCheck, addMirrorValues, setMirrorValues);
  }

  // Normalize the computed BC SDR
  const stk::mesh::Selector sel = (
    realm_.meta_data().locally_owned_part() |
    realm_.meta_data().globally_shared_part()) &
    stk::mesh::selectField(*bcsdrF);

  nalu_ngp::run_entity_algorithm(
    ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double warea = wallArea.get(mi, 0);
      const double sdrVal = bcsdr.get(mi, 0) / warea;

      bcsdr.get(mi, 0) = sdrVal;
      sdrWallBC.get(mi, 0) = sdrVal;
      sdr.get(mi, 0) = sdrVal;
    });

  wallArea.modify_on_device();
  bcsdr.modify_on_device();
  sdr.modify_on_device();
}

}  // nalu
}  // sierra