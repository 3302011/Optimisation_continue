param n; 
param m; 

set I={1..m}; 
set J={1..n}; 

param c{I,J}; 
param a{I,J} ;
param mm_{I,J} ; 

param AA{J};
param MM{I}; 


var y{I,J}>=0 binary;
var x{I,J}>=0 ; # variable de la fonction objectif

# défintion de la fonction objectif
minimize z: sum{i in I , j in J}(y[i,j]*a[i,j]+x[i,j]*c[i,j]);
	

s.t. ConstraintA{j in J}:
	sum{i in I} x[i,j]>=AA[j];
	
	ConstraintM{i in I, k in J}:
	sum{j in J}x[i,j]<=MM[i];
	
	ConstraintMY1{i in I, j in J}:
	x[i,j]<=MM[i]*y[i,j];
	
	ConstraintMY2{i in I, j in J}:
	x[i,j]>=mm_[i,j]*y[i,j];