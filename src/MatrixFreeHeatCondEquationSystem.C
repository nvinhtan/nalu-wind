// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "MatrixFreeHeatCondEquationSystem.h"

#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/ConductionUpdate.h"

#include "NaluEnv.h"
#include "NaluParsing.h"
#include "PeriodicManager.h"
#include "Realm.h"
#include "TimeIntegrator.h"

#include "stk_mesh/base/Selector.hpp"
#include <stk_mesh/base/NgpForEachEntity.hpp>
#include <stk_mesh/base/Types.hpp>

namespace sierra {
namespace nalu {

MatrixFreeHeatCondEquationSystem::MatrixFreeHeatCondEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "HeatCondEQS", "temperature"),
    polynomial_order_(realm_.polynomial_order()),
    meta_(realm_.meta_data())
{

  realm_.push_equation_to_systems(this);
  ThrowRequireMsg(
    realm_.spatialDimension_ == 3u,
    "Only 3D supported for matrix free heat conduction");
  ThrowRequireMsg(realm_.matrixFree_, "Only matrix free supported");
}

MatrixFreeHeatCondEquationSystem::~MatrixFreeHeatCondEquationSystem() = default;

namespace {
template <typename T = double>
stk::mesh::NgpField<T>&
get_node_field(
  const stk::mesh::MetaData& meta,
  std::string name,
  stk::mesh::FieldState state = stk::mesh::StateNP1)
{
  ThrowAssert(meta.get_field(stk::topology::NODE_RANK, name));
  ThrowAssert(
    meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));
  return stk::mesh::get_updated_ngp_field<T>(
    *meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));
}

void
register_scalar_nodal_field_on_part(
  stk::mesh::MetaData& meta,
  std::string name,
  const stk::mesh::Selector& selector,
  int num_states,
  double ic = 0)
{

  auto& field = meta.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, name, num_states);
  stk::mesh::put_field_on_mesh(field, selector, 1, &ic);
}

} // namespace

void
MatrixFreeHeatCondEquationSystem::register_nodal_fields(stk::mesh::Part* part)
{
  constexpr int three_states = 3;
  constexpr int one_state = 1;
  register_scalar_nodal_field_on_part(
    meta_, names::temperature, *part, three_states);
  register_scalar_nodal_field_on_part(meta_, names::delta, *part, one_state);
  register_scalar_nodal_field_on_part(
    meta_, names::volume_weight, *part, one_state);
  register_scalar_nodal_field_on_part(meta_, names::density, *part, one_state);
  register_scalar_nodal_field_on_part(
    meta_, names::specific_heat, *part, one_state);
  register_scalar_nodal_field_on_part(
    meta_, names::thermal_conductivity, *part, one_state);

  realm_.augment_restart_variable_list(names::temperature);
  realm_.augment_property_map(
    DENSITY_ID, meta_.get_field<stk::mesh::Field<double>>(
                  stk::topology::NODE_RANK, names::density));
  realm_.augment_property_map(
    SPEC_HEAT_ID, meta_.get_field<stk::mesh::Field<double>>(
                    stk::topology::NODE_RANK, names::specific_heat));
  realm_.augment_property_map(
    THERMAL_COND_ID, meta_.get_field<stk::mesh::Field<double>>(
                       stk::topology::NODE_RANK, names::thermal_conductivity));
}

void
MatrixFreeHeatCondEquationSystem::register_interior_algorithm(
  stk::mesh::Part* part)
{
  ThrowRequireMsg(
    matrix_free::part_is_valid_for_matrix_free(polynomial_order_, *part),
    "part " + part->name() + " has invalid topology " +
      part->topology().name() + ". Only hex8/hex27 supported");
  interior_selector_ |= *part;
}

void
MatrixFreeHeatCondEquationSystem::register_wall_bc(
  stk::mesh::Part* part,
  const stk::topology&,
  const WallBoundaryConditionData& wallBCData)
{
  ThrowRequireMsg(
    matrix_free::part_is_valid_for_matrix_free(polynomial_order_, *part),
    "part " + part->name() + " has invalid topology " +
      part->topology().name() + ". Only Quad4 and Quad9 supported");

  WallUserData userData = wallBCData.userData_;
  ThrowRequireMsg(!userData.irradSpec_, "Irradiation BC not implemented");
  ThrowRequireMsg(!userData.robinParameterSpec_, "Robin BC not implemented");

  if (userData.tempSpec_) {
    const auto temperature_data = wallBCData.userData_.temperature_;
    constexpr int one_state = 1;
    register_scalar_nodal_field_on_part(
      meta_, names::qbc, *part, one_state, temperature_data.temperature_);
    dirichlet_selector_ |= *part;
  } else if (userData.heatFluxSpec_) {
    const auto flux_data = wallBCData.userData_.q_;

    constexpr int one_state = 1;
    register_scalar_nodal_field_on_part(
      meta_, names::flux, *part, one_state, flux_data.qn_);
    flux_selector_ |= *part;
  }
}

void
MatrixFreeHeatCondEquationSystem::initialize()
{
  stk::mesh::ProfilingBlock pf("MatrixFreeHeatCondEquationSystem::initialize");

  stk::mesh::Selector replica_selector{};
  if (realm_.periodicManager_ != nullptr) {
    replica_selector =
      stk::mesh::selectUnion(realm_.periodicManager_->get_slave_part_vector());
  }
  auto& bulk = realm_.bulk_data();
  {
    stk::mesh::ProfilingBlock pf_inner("fill_tpetra_id_field");
    matrix_free::populate_global_id_field(
      realm_.ngp_mesh(), interior_selector_ - replica_selector,
      get_node_field<typename Tpetra::Map<>::global_ordinal_type>(
        meta_, names::tpetra_gid));
  }

  {
    stk::mesh::ProfilingBlock pf_inner("make_equation_update");
    update_ = matrix_free::make_equation_update<matrix_free::ConductionUpdate>(
      polynomial_order_, bulk, realm_.solver_parameters("temperature"),
      interior_selector_, dirichlet_selector_, flux_selector_,
      replica_selector);
  }
}

void
MatrixFreeHeatCondEquationSystem::reinitialize_linear_system()
{
  initialized_ = false;
  initialize();
}

namespace {

void
field_hadamard(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& selector,
  stk::mesh::NgpField<double>& xy,
  const stk::mesh::NgpConstField<double>& x,
  const stk::mesh::NgpConstField<double>& y)
{
  stk::mesh::for_each_entity_run(
    mesh, stk::topology::NODE_RANK, selector,
    KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
      xy.get(mi, 0) = x.get(mi, 0) * y.get(mi, 0);
    });
  xy.modify_on_device();
}

} // namespace

void
MatrixFreeHeatCondEquationSystem::predict_state()
{
  stk::mesh::ProfilingBlock("MatrixFreeHeatCondEquationSystem::predict_state");

  const auto& current_state =
    get_node_field(meta_, names::temperature, stk::mesh::StateN);
  auto& predicted_state =
    get_node_field(meta_, names::temperature, stk::mesh::StateNP1);
  stk::mesh::for_each_entity_run(
    realm_.ngp_mesh(), stk::topology::NODE_RANK, interior_selector_,
    KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
      predicted_state.get(mi, 0) = current_state.get(mi, 0);
    });
  predicted_state.modify_on_device();
}

void
MatrixFreeHeatCondEquationSystem::compute_volumetric_heat_capacity() const
{
  stk::mesh::ProfilingBlock pf_inner("compute_volumetric_heat_capacity");
  const auto& rho = get_node_field(meta_, names::density);
  const auto& cp = get_node_field(meta_, names::specific_heat);
  auto& alpha = get_node_field(meta_, names::volume_weight);
  field_hadamard(realm_.ngp_mesh(), interior_selector_, alpha, rho, cp);
}

double
MatrixFreeHeatCondEquationSystem::provide_norm() const
{
  return update_->provide_norm();
}

double
MatrixFreeHeatCondEquationSystem::provide_scaled_norm() const
{
  return update_->provide_scaled_norm();
}

void
MatrixFreeHeatCondEquationSystem::sync_delta_on_periodic_nodes() const
{
  stk::mesh::ProfilingBlock pf("sync_periodic nodes");
  if (realm_.hasPeriodic_) {
    realm_.periodic_delta_solution_update(
      meta_.get_field(stk::topology::NODE_RANK, names::delta), 1);
  }
}

namespace {

Kokkos::Array<double, 3>
compute_scaled_gammas(const TimeIntegrator& ti)
{
  return Kokkos::Array<double, 3>{{ti.get_gamma1() / ti.get_time_step(),
                                   ti.get_gamma2() / ti.get_time_step(),
                                   ti.get_gamma3() / ti.get_time_step()}};
}

void
nonlinear_iteration_banner(
  int k, int max_k, std::string name, std::ostream& stream)
{
  stream << " " << k + 1 << "/" << max_k << std::setw(15) << std::right << name
         << std::endl;
}

} // namespace

void
MatrixFreeHeatCondEquationSystem::initialize_solve_and_update()
{
  stk::mesh::ProfilingBlock pf("initialize");
  if (initialized_) {
    return;
  }
  initialized_ = true;

  compute_volumetric_heat_capacity();
  update_->initialize();
}

void
MatrixFreeHeatCondEquationSystem::solve_and_update()
{
  const auto time_start_initialize = NaluEnv::self().nalu_time();
  initialize_solve_and_update();
  const auto time_end_initialize = NaluEnv::self().nalu_time();
  timerInit_ += time_end_initialize - time_start_initialize;

  const auto time_start_update_states = NaluEnv::self().nalu_time();
  update_->swap_states();
  update_->update_solution_fields();
  const auto time_end_update_states = NaluEnv::self().nalu_time();
  timerAssemble_ += time_end_update_states - time_start_update_states;

  const auto time_start_preconditioner = NaluEnv::self().nalu_time();
  update_->compute_preconditioner(
    realm_.timeIntegrator_->get_gamma1() /
    realm_.timeIntegrator_->get_time_step());
  const auto time_end_preconditioner = NaluEnv::self().nalu_time();
  timerPrecond_ += time_end_preconditioner - time_start_preconditioner;

  for (int k = 0; k < maxIterations_; ++k) {
    nonlinear_iteration_banner(
      k, maxIterations_, userSuppliedName_, NaluEnv::self().naluOutputP0());

    const auto time_start_solve = NaluEnv::self().nalu_time();
    update_->compute_update(
      compute_scaled_gammas(*realm_.timeIntegrator_),
      get_node_field(meta_, names::delta));
    const auto time_end_solve = NaluEnv::self().nalu_time();
    timerSolve_ += time_end_solve - time_start_solve;

    const auto time_start_assemble = NaluEnv::self().nalu_time();
    sync_delta_on_periodic_nodes();

    solution_update(
      1.0,
      *meta_.get_field<ScalarFieldType>(stk::topology::NODE_RANK, names::delta),
      1.0,
      meta_
        .get_field<ScalarFieldType>(
          stk::topology::NODE_RANK, names::temperature)
        ->field_of_state(stk::mesh::StateNP1));

    update_->update_solution_fields();
    const auto time_end_assemble = NaluEnv::self().nalu_time();
    timerAssemble_ += time_end_assemble - time_start_assemble;

    const auto time_start_banner = NaluEnv::self().nalu_time();
    update_->banner(name_, NaluEnv::self().naluOutputP0());
    const auto time_end_banner = NaluEnv::self().nalu_time();
    timerMisc_ += time_end_banner - time_start_banner;
  }
}

} // namespace nalu
} // namespace sierra
