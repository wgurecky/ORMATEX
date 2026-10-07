// Rising-bubble-plume tank: 0.4 m square box with a centered 0.01 m
// square hole acting as the bubble injector (like the cylinder obstacle).
// Outer walls are no-slip, the top is a free surface, and the bottom edge
// of the inner hole is a velocity/void inlet.
//
// Mesh grading (background Distance + Threshold fields):
//   h_fine   - element size at the injector walls (resolves jet shear
//              layer and inner corners).
//   h_coarse - element size at the outer walls.
//   dist_min - distance from injector below which size = h_fine.
//   dist_max - distance from injector above which size = h_coarse;
//              linear ramp in between for smooth coarsening.
// Tune h_fine / h_coarse / dist_min / dist_max below to trade accuracy
// (injector shear layer) against total quad count for the p=2 SEM solve.
Mesh.MshFileVersion = 2.2;
Mesh.Binary = 1;

// Size control parameters (meters).
h_fine = 0.001;
h_coarse = 0.025;
dist_min = 0.005;
dist_max = 0.15;

L = 0.4;
S = 0.01;
CX = 0.2;
CY = 0.2;

XI0 = CX - S / 2;
XI1 = CX + S / 2;
YI0 = CY - S / 2;
YI1 = CY + S / 2;

// Explicit built-in geometry with deterministic curve ids so the
// Distance background field can reference the injector hole boundary
// directly (CurvesList = {5, 6, 7, 8}). The Physical definitions below
// keep their original names/numbers and BoundingBox selectors, which the
// example and tests/drift_flux_free_surface.rs depend on.
Point(1) = {0, 0, 0};
Point(2) = {L, 0, 0};
Point(3) = {L, L, 0};
Point(4) = {0, L, 0};
Point(5) = {XI0, YI0, 0};
Point(6) = {XI1, YI0, 0};
Point(7) = {XI1, YI1, 0};
Point(8) = {XI0, YI1, 0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(1) = {1, 2};

// All-quad Frontal-Delaunay with blossom recombination. Subdivision
// algorithm forces any residual triangles to split so the mesh stays
// QUAD-ONLY for the gmsh_quad_data loader.
Mesh.Algorithm = 8;
Mesh.RecombineAll = 1;
Mesh.RecombinationAlgorithm = 3;
Mesh.SubdivisionAlgorithm = 1;
Mesh.Smoothing = 10;

// Background-field-only sizing: ignore per-point sizes, curvature
// refinement, and boundary extension so the Distance/Threshold fields
// fully control the grading.
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeMin = h_fine;
Mesh.MeshSizeMax = h_coarse;

// Distance from the injector hole boundary curves.
Field[1] = Distance;
Field[1].CurvesList = {5, 6, 7, 8};
Field[1].Sampling = 100;

// Linear ramp: h_fine close to the hole, h_coarse far away.
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_fine;
Field[2].SizeMax = h_coarse;
Field[2].DistMin = dist_min;
Field[2].DistMax = dist_max;
Background Field = 2;

Physical Curve("left", 1) = Curve In BoundingBox {-0.0001, -0.0001, -0.0001, 0.0001, L + 0.0001, 0.0001};
Physical Curve("right", 2) = Curve In BoundingBox {L - 0.0001, -0.0001, -0.0001, L + 0.0001, L + 0.0001, 0.0001};
Physical Curve("bottom", 3) = Curve In BoundingBox {-0.0001, -0.0001, -0.0001, L + 0.0001, 0.0001, 0.0001};
Physical Curve("freesurface", 4) = Curve In BoundingBox {-0.0001, L - 0.0001, -0.0001, L + 0.0001, L + 0.0001, 0.0001};
Physical Curve("injector_inlet", 5) = Curve In BoundingBox {XI0 - 0.0001, YI0 - 0.0001, -0.0001, XI1 + 0.0001, YI0 + 0.0001, 0.0001};
Physical Curve("injector_wall_left", 6) = Curve In BoundingBox {XI0 - 0.0001, YI0 - 0.0001, -0.0001, XI0 + 0.0001, YI1 + 0.0001, 0.0001};
Physical Curve("injector_wall_right", 7) = Curve In BoundingBox {XI1 - 0.0001, YI0 - 0.0001, -0.0001, XI1 + 0.0001, YI1 + 0.0001, 0.0001};
Physical Curve("injector_wall_top", 8) = Curve In BoundingBox {XI0 - 0.0001, YI1 - 0.0001, -0.0001, XI1 + 0.0001, YI1 + 0.0001, 0.0001};
Physical Surface("domain", 10) = {1};
