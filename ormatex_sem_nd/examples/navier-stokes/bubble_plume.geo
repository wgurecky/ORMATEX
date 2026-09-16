// Rising-bubble-plume tank: 0.1 m square box with a centered 0.01 m
// square hole acting as the bubble injector (like the cylinder obstacle).
// Outer walls are no-slip, the top is a free surface, and the bottom edge
// of the inner hole is a velocity/void inlet.
Mesh.MshFileVersion = 2.2;
Mesh.Binary = 1;
SetFactory("OpenCASCADE");

L = 0.1;
S = 0.01;
CX = 0.05;
CY = 0.05;

Rectangle(1) = {0, 0, 0, L, L};
Rectangle(2) = {CX - S / 2, CY - S / 2, 0, S, S};
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; }

// All-quad recombination plus a refined box around the injector so the
// jet shear layer and the inner corners stay well shaped.
Mesh.Algorithm = 8;
Mesh.RecombinationAlgorithm = 2;
Mesh.RecombineAll = 1;
Mesh.Smoothing = 10;
Mesh.CharacteristicLengthMin = 0.0006;
Mesh.CharacteristicLengthMax = 0.003;

Field[1] = Box;
Field[1].VIn = 0.0006;
Field[1].VOut = 0.003;
Field[1].XMin = CX - 0.02;
Field[1].XMax = CX + 0.02;
Field[1].YMin = CY - 0.02;
Field[1].YMax = CY + 0.02;
Field[1].Thickness = 0.01;
Background Field[1] = 1;

XI0 = CX - S / 2;
XI1 = CX + S / 2;
YI0 = CY - S / 2;
YI1 = CY + S / 2;

Physical Curve("left", 1) = Curve In BoundingBox {-0.0001, -0.0001, -0.0001, 0.0001, L + 0.0001, 0.0001};
Physical Curve("right", 2) = Curve In BoundingBox {L - 0.0001, -0.0001, -0.0001, L + 0.0001, L + 0.0001, 0.0001};
Physical Curve("bottom", 3) = Curve In BoundingBox {-0.0001, -0.0001, -0.0001, L + 0.0001, 0.0001, 0.0001};
Physical Curve("freesurface", 4) = Curve In BoundingBox {-0.0001, L - 0.0001, -0.0001, L + 0.0001, L + 0.0001, 0.0001};
Physical Curve("injector_inlet", 5) = Curve In BoundingBox {XI0 - 0.0001, YI0 - 0.0001, -0.0001, XI1 + 0.0001, YI0 + 0.0001, 0.0001};
Physical Curve("injector_wall_left", 6) = Curve In BoundingBox {XI0 - 0.0001, YI0 - 0.0001, -0.0001, XI0 + 0.0001, YI1 + 0.0001, 0.0001};
Physical Curve("injector_wall_right", 7) = Curve In BoundingBox {XI1 - 0.0001, YI0 - 0.0001, -0.0001, XI1 + 0.0001, YI1 + 0.0001, 0.0001};
Physical Curve("injector_wall_top", 8) = Curve In BoundingBox {XI0 - 0.0001, YI1 - 0.0001, -0.0001, XI1 + 0.0001, YI1 + 0.0001, 0.0001};
Physical Surface("domain", 10) = {1};
