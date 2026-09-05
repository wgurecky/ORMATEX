SetFactory("Built-in");

lc = 1.0 / 20.0;
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};

Line(1) = {1, 2}; // bottom
Line(2) = {2, 3}; // cold right wall
Line(3) = {3, 4}; // top
Line(4) = {4, 1}; // hot left wall

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Transfinite Line {1, 2, 3, 4} = 21;
Transfinite Surface {1};
Recombine Surface {1};

Physical Line("adiabatic_bottom") = {1};
Physical Line("cold") = {2};
Physical Line("adiabatic_top") = {3};
Physical Line("hot") = {4};
Physical Surface("domain") = {1};
