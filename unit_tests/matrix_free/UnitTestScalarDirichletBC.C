// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ScalarDirichletBC.h"
#include "matrix_free/StkEntityToRowMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkSimdNodeConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/ConductionFields.h"

#include "StkConductionFixture.h"
#include "gtest/gtest.h"

#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#include <Kokkos_Array.hpp>
#include <Kokkos_CopyViews.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_Export.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <algorithm>
#include <stk_simd/Simd.hpp>
#include <type_traits>

#include "mpi.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

class DirichletFixture : public ConductionFixture
{
protected:
  DirichletFixture()
    : ConductionFixture(nx, scale),
      owned_map(make_owned_row_map(mesh, meta.universal_part())),
      owned_and_shared_map(make_owned_and_shared_row_map(
        mesh, meta.universal_part(), gid_field_ngp)),
      exporter(
        Teuchos::rcpFromRef(owned_and_shared_map),
        Teuchos::rcpFromRef(owned_map)),
      owned_lhs(Teuchos::rcpFromRef(owned_map), 1),
      owned_rhs(Teuchos::rcpFromRef(owned_map), 1),
      owned_and_shared_lhs(Teuchos::rcpFromRef(owned_and_shared_map), 1),
      owned_and_shared_rhs(Teuchos::rcpFromRef(owned_and_shared_map), 1),
      elid(make_stk_lid_to_tpetra_lid_map(
        mesh,
        meta.universal_part(),
        gid_field_ngp,
        owned_and_shared_map.getLocalMap())),
      dirichlet_nodes(simd_node_map(
        mesh, meta.get_topology_root_part(stk::topology::QUAD_4))),
      dirichlet_offsets(simd_node_offsets(
        mesh, meta.get_topology_root_part(stk::topology::QUAD_4), elid))
  {
    owned_lhs.putScalar(0.);
    owned_rhs.putScalar(0.);

    for (const auto* ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateNP1), node) = 5.0;
        *stk::mesh::field_data(qbc_field, node) = -2.3;
      }
    }
  }

  static constexpr int nx = 4;
  static constexpr double scale = M_PI;

  const Tpetra::Map<> owned_map;
  const Tpetra::Map<> owned_and_shared_map;
  const Tpetra::Export<> exporter;
  Tpetra::MultiVector<> owned_lhs;
  Tpetra::MultiVector<> owned_rhs;
  Tpetra::MultiVector<> owned_and_shared_lhs;
  Tpetra::MultiVector<> owned_and_shared_rhs;

  const const_entity_row_view_type elid;
  const const_node_mesh_index_view dirichlet_nodes;
  const const_node_offset_view dirichlet_offsets;
};

TEST_F(DirichletFixture, bc_residual)
{
  auto qp1 = node_scalar_view("qp1_at_bc", dirichlet_nodes.extent_int(0));
  stk_simd_scalar_node_gather(
    dirichlet_nodes, get_ngp_field<double>(meta, conduction_info::q_name), qp1);

  auto qbc =
    node_scalar_view("qspecified_at_bc", dirichlet_nodes.extent_int(0));
  stk_simd_scalar_node_gather(
    dirichlet_nodes, get_ngp_field<double>(meta, conduction_info::qbc_name),
    qbc);

  owned_and_shared_rhs.putScalar(0.);
  scalar_dirichlet_residual(
    dirichlet_offsets, qp1, qbc, owned_rhs.getLocalLength(),
    owned_and_shared_rhs.getLocalViewDevice());
  owned_and_shared_rhs.modify_device();
  owned_rhs.putScalar(0.);
  owned_rhs.doExport(owned_and_shared_rhs, exporter, Tpetra::ADD);

  owned_rhs.sync_host();
  auto view_h = owned_rhs.getLocalViewHost();

  double maxval = -1;
  for (size_t k = 0u; k < owned_rhs.getLocalLength(); ++k) {
    maxval = std::max(maxval, std::abs(view_h(k, 0)));
  }
  ASSERT_DOUBLE_EQ(maxval, 7.3);
}

TEST_F(DirichletFixture, linearized_bc_residual)
{
  constexpr double some_val = 85432.2;
  owned_lhs.putScalar(some_val);

  owned_and_shared_lhs.doImport(owned_lhs, exporter, Tpetra::INSERT);

  scalar_dirichlet_linearized(
    dirichlet_offsets, owned_lhs.getLocalLength(),
    owned_and_shared_lhs.getLocalViewDevice(),
    owned_and_shared_rhs.getLocalViewDevice());

  owned_and_shared_rhs.modify_device();
  owned_rhs.putScalar(0.);
  owned_rhs.doExport(owned_and_shared_rhs, exporter, Tpetra::ADD);

  owned_rhs.sync_host();
  auto view_h = owned_rhs.getLocalViewHost();

  constexpr double tol = 1.0e-14;
  double maxval = -1;
  for (size_t k = 0u; k < owned_rhs.getLocalLength(); ++k) {
    const bool zero_or_val = std::abs(view_h(k, 0) - 0) < tol ||
                             std::abs(view_h(k, 0) - some_val) < tol;
    ASSERT_TRUE(zero_or_val);
    maxval = std::max(maxval, view_h(k, 0));
  }
  ASSERT_DOUBLE_EQ(maxval, some_val);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
