// Three-block transfinite/recombined backward-facing-step channel.
Mesh.MshFileVersion = 2.2;

Point(1) = {0, 0, 0, 1};
Point(2) = {1, 0, 0, 1};
Point(3) = {1, 1, 0, 1};
Point(4) = {0, 1, 0, 1};
Point(5) = {1, -1, 0, 1};
Point(6) = {5, -1, 0, 1};
Point(7) = {5, 0, 0, 1};
Point(8) = {5, 1, 0, 1};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 2};
Line(8) = {2, 5};
Line(9) = {7, 8};
Line(10) = {8, 3};

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Line Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};
Line Loop(3) = {-7, 9, 10, -2};
Plane Surface(3) = {3};

Transfinite Curve {1, 2, 3, 4, 6, 8, 9} = 30;
Transfinite Curve {5, 7, 10} = 50;
Transfinite Surface {1, 2, 3};
Recombine Surface {1, 2, 3};

Physical Curve("inlet", 1) = {4};
Physical Curve("outlet", 2) = {6, 9};
Physical Curve("wall", 3) = {1, 3, 5, 8, 10};
Physical Surface("domain", 10) = {1, 2, 3};
