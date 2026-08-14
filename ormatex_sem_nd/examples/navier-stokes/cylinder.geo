// All-quad channel with a circular obstacle.
Mesh.MshFileVersion = 2.2;
SetFactory("OpenCASCADE");

L = 15;
H = 4;
D = 1;

Rectangle(1) = {-5, -H / 2, 0, L, H};
Disk(2) = {0, 0, 0, D, D};
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; }

// Blossom recombination is used rather than accepting triangles from a
// generic triangular mesh. The checked-in MSH2 file is verified by the loader.
Mesh.Algorithm = 8;
Mesh.RecombinationAlgorithm = 2;
Mesh.RecombineAll = 1;
Mesh.Smoothing = 10;
Mesh.CharacteristicLengthMin = 0.08;
Mesh.CharacteristicLengthMax = 0.24;

Physical Curve("inlet", 1) = Curve In BoundingBox {-5.0001, -2.0001, -0.0001, -4.9999, 2.0001, 0.0001};
Physical Curve("outlet", 2) = Curve In BoundingBox {9.9999, -2.0001, -0.0001, 10.0001, 2.0001, 0.0001};
Physical Curve("bottom", 3) = Curve In BoundingBox {-5.0001, -2.0001, -0.0001, 10.0001, -1.9999, 0.0001};
Physical Curve("top", 4) = Curve In BoundingBox {-5.0001, 1.9999, -0.0001, 10.0001, 2.0001, 0.0001};
Physical Curve("cylinder", 5) = Curve In BoundingBox {-1.0001, -1.0001, -0.0001, 1.0001, 1.0001, 0.0001};
Physical Surface("domain", 10) = {1};
