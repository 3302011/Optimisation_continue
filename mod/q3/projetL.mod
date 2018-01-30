#Partie 1 Déclaration de variable

param n;
param m;

set I={1..m};
set J={1..n};

param MM{i in I};
param AA{j in J};
param a{i in I,j in J};
param c{i in I,j in J};
param mm_{i in I,j in J};

var x{i in I,j in J}>=0;
var y{i in I,j in J}>=0 binary;
var z{i in I,j in J,k in J}>=0;

#Fonction objective

minimize cout: sum{i in I,j in J}(y[i,j]*a[i,j]+c[i,j]*x[i,j]);

#Contraintes

subject to c1{j in J}:sum{i in I}x[i,j]>=AA[j];
subject to c2{i in I,k in J}:sum{j in J}z[i,j,k]<=MM[i]*y[i,k];
subject to c3{i in I,k in J}:sum{j in J}(x[i,j]-z[i,j,k])<=MM[i]-MM[i]*y[i,k];
subject to c4{i in I,j in J}:x[i,j]>=mm_[i,j]*y[i,j];
subject to c5{i in I,j in J}:x[i,j]<=MM[i]*y[i,j];
subject to c6{i in I,j in J,k in J}:z[i,j,k]<=x[i,j];