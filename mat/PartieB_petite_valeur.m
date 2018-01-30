function PartieB()

% question 5 

%Lance les variables du fichier 1.m 
n=  4;
m=   4;
MM=[  112,  109,  131,  318 ];
AA=[    6,    8,    8,   30];
a=[
[  251,  144,  116,  260]
[  253,  112,  234,  182]
[  144,  147,  220,  252]
[  105,  239,  191,  118]
];
c=[
[ 1, 1, 4, 4]
[10,10, 4,10]
[ 1,10, 7, 1]
[ 1, 2, 8, 1]
];
mm_=[
[    0,    0,    0,    0]
[    0,    0,    0,    0]
[    0,    0,    0,    0]
[    0,    0,    0,    0]
];

%rng(3);
%run('1.m') %c'est pas la bonne commande mais je m'en souviens plus 

%p.s: les fichiers en seront pas dans le même dossier 

% Le problème (P)

% La fonction objective 
object= @(x) objective(x,m,n,a,c); %a et c donnés par les scripts 

% Les contraintes
[A,b]=contraintes(AA, MM, mm_, m, n);
%Les variables 
ub = ones(2*m*n,1)';
ub(1:m*n)=Inf; %en suivant les instructions sur la page info de ga
lb = zeros(2*m*n,1)';
%intervalle = m*n+1:2*m*n;
x0 =[    6,    0,    0,    0,    0,    10,    0,    0,    0,    0,    8,    0    0,    0,    0,    30,    1,    0,    0,    0,    0,    1,    0,    0,    0,    0,    1,    0    0,    0,    0,    1];
%x0 = rand(1,2*m*n).*10;
opti = gaoptimset('InitialPopulation',x0, 'TimeLimit',60, 'PopulationSize', 3200,'PopulationType','doubleVector');
[X,fval,exitFlag,Output] = ga(object, 2*m*n, A,b,[],[],lb,ub,[],m*n+1:2*m*n,opti);


fprintf('The number of generations was : %d\n', Output.generations);
fprintf('The number of function evaluations was : %d\n', Output.funccount);
fprintf('The best function value found was : %g\n', fval);

%X = [6 8 8 30 6 32 2 2 7 2 3 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 0 0 0 0 0 0 0 0 0 0 0 0 0 11 0 0 0 0 0 0 0 0 11 0 15 11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12 0 0 0 11 0 0 15 0 0 0 0 0 0 15 0 0 0 0 0 0 0 0 0 0 0 0 0 8 10 8 0 0 0 0 0 0 0 0 0 8 0 0 0 0 0 0 0 8 0 0 8 0 6 0 6 0 7 0 9 0 0 0 9 0 5 0 7 0 0 3 0 0 5 0 0 17 7 0 0 0 0 0 0 0 0 0 0 0 0 37 18 0 0 0 21 4 0 3 4 1 24 44 3 0 3 0 3 0 0 2 19 0 1 4 0 1 0 4 0 2 0 3 0 22 31 2 0 0 0 4 0 0 33 14 0 3 0 4 4 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
fid=fopen('data.txt','w');
   fprintf(fid,'matrice X trouvé avec ga \n');
   fprintf(fid,'%d ',X);
   fprintf(fid, '\nresultat optimal : \n');
   fprintf(fid, '%d\n\n', fval);

%patternsearch 
X = patternsearch(object,x0,A,b,[],[],lb,ub);
   fprintf(fid,'matrice X trouvé avec patternsearch \n');
   fprintf(fid,'%d ',X);
   fprintf(fid, '\nresultat optimal : \n');
   fprintf(fid, '%d\n\n', object(X));

fclose(fid);
 
edit data.txt

% question 6 

% question 7

end 


function object=objective(X,m,n,a,c) % X est celui que l'on cherche 

x = zeros(m,n); % x de la fonction objective 
y = zeros(m,n); % y deinteger_values = m*n+1:1:2*m*n; la fonction objective 
object=0;

for i=1:m 
    for j=1:n
        x(i,j) = X((i-1)*n+j);
        y(i,j) = X(m*n+(i-1)*n+j);
    end
end

for i=1:m
    o=0;
    for j=1:n
        o=o + y(i,j)*a(i,j)+ c(i,j)*x(i,j);
    end 
    object=object+o;
end 
end 

function [A,b]=contraintes( AA, MM, mm_, m, n)

A=zeros(2*m*n+n+m,2*m*n);
b=zeros(2*m*n+n+m,1);

%contrainte 46

for j=1:n %colonnes
    for i=1:m %lignes     
        A(j,j+((i-1)*n)) = -1; %somme sur la premiere colonne donc on saute le reste
        %on laisse le reste à 0
    end
    b(j) = -AA(j); %on multiplie par (-1) car ga est en inf 
end

%contrainte 47
iterateur=n; %on a remplit les n lignes avant donc on commence à n


for i=1:m
    for j=1:n 
        A(iterateur+i,(i-1)*n+j)=1; 
    end 
    b(iterateur+i)=MM(i);
end 

%contrainte 48

iterateur=iterateur+m; % m+n

% a gauche 
for i=1:m
    for j=1:n 
        A(iterateur+(i-1)*n+j,(i-1)*n+j)=1;
        A(iterateur+(i-1)*n+j,n*m+(i-1)*n+j)=-MM(i); %n*m pour passer aux y
        b(iterateur+(i-1)*n+j)=0;
    end     
end

iterateur=iterateur+m*n; % m+n+m*n
%a droite

for i=1:m
    for j=1:n 
        A(iterateur+(i-1)*n+j,(i-1)*n+j)=-1;
        A(iterateur+(i-1)*n+j,n*m+(i-1)*n+j)=mm_(i,j); 
        b(iterateur+(i-1)*n+j)=0;
    end     
end


% taille de la matrice A ligne : m+n+2*m*n
% colonne : 2*m*n

%les bornes sur x et y sont inutiles dans A car elles sont déjà dans ub et
%lb !!!!


%contrainte 49
%{
iterateur=iterateur+m*n; % m+n+2*m*n

for i=1:m
    for j=1:n 
        A(iterateur+(i-1)*n+j,(i-1)*n+j)=-1;
        b(iterateur+(i-1)*n+j)=0;
    end     
end

iterateur=iterateur+m*n; % m+n+3*m*n

for i=1:m
    for j=1:n 
        A(iterateur+(i-1)*n+j,(i-1)*n+j)=-1; % condition superieur
        b(iterateur+(i-1)*n+j)=0;
    end     
end

iterateur=iterateur+m*n; % m+n+4*m*n

for i=1:m
    for j=1:n 
        A(iterateur+(i-1)*n+j,(i-1)*n+j)=1; %condition inferieur
        b(iterateur+(i-1)*n+j)=1;
    end     
end
%}


end