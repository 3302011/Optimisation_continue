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
var X{I,J}>=0 ; # variable de la fonction objectif

# défintion de la fonction objectif
minimize z: sum{i in I , j in J}(y[i,j]*a[i,j]+X[i,j]*c[i,j]);
	

s.t. ConstraintA{j in J}:
	sum{i in I} X[i,j]>=AA[j];
	
	ConstraintMY{i in I, k in J}:
	sum{j in J}X[i,j]*y[i,k]<=MM[i]*y[i,k];
	
	ConstraintMY2{i in I, k in J}:
	sum{j in J }X[i,j]*(1-y[i,k])<=MM[i]*(1-y[i,k]);
	
	Constraint4{i in I, j in J}:
	X[i,j]<=MM[i]*y[i,j];
	Constraint5{i in I, j in J}:
	X[i,j]>=mm_[i,j]*y[i,j];

