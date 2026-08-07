// Structured unit-square quadrilateral mesh for the tagged-boundary example.
Mesh.MshFileVersion = 2.2;

nx = 32;
ny = 2;

Point(1) = {0, 0, 0, 1};
Point(2) = {1, 0, 0, 1};
Point(3) = {1, 1, 0, 1};
Point(4) = {0, 1, 0, 1};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Transfinite Curve {1, 3} = nx + 1;
Transfinite Curve {2, 4} = ny + 1;
Transfinite Surface {1};
Recombine Surface {1};

Physical Curve("left", 1) = {4};
Physical Curve("right", 2) = {2};
Physical Curve("bottom", 3) = {1};
Physical Curve("top", 4) = {3};
Physical Surface("domain", 10) = {1};
