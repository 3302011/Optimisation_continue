function PartieB()

% question 5 

%Lance les variables du fichier 1.m 
n=  64;
m=   4;
MM=[  112,  109,  131,  318 ];
AA=[    6,    8,    8,   30,   14,   32,    2,    2,    7,    2,    3,   43,   18,    8,   10,    8,   21,    4,   11,    3,    4,    1,   24,   44,    3,    8,    3,   11,    3,   15,   11,    2,   19,    8,    1,    4,    8,    1,    6,    4,    6,    2,    7,    3,    9,   22,   31,    2,    9,   12,    5,    4,    7,   11,   33,   17,   15,    3,    5,    4,    4,   17,    7,   15 ];
a=[
[  251,  144,  116,  260,  167,  130,  187,  251,  142,  180,  139,  231,  190,  252,   89,   65,  260,  111,  247,  256,  252,  193,  256,   97,  234,  187,  232,  196,  202,  190,   90,   90,   94,  106,  235,  134,  147,  200,  169,  196,  235,   71,  186,  143,  124,  256,  123,  134,  232,   72,   87,  110,  250,   98,   83,  254,  101,  163,  130,  260,  106,  163,  169,  164 ],
[  253,  112,  234,  182,  194,   89,  100,  158,  201,  198,  255,  213,  100,  260,  250,  235,  139,  128,  248,  235,  157,   93,  263,  125,  206,   86,  207,  226,  254,  129,  177,   66,   80,  216,  192,   89,   66,  122,  129,  169,  209,  196,  252,   67,  177,  225,  160,  157,  161,   70,  111,  186,  262,  259,   93,  192,  218,  141,  127,  144,  215,  177,  241,   99 ],
[  144,  147,  220,  252,  254,  119,  187,  162,  149,  126,  158,  182,  248,  232,  115,  235,  104,  159,  217,  226,   68,  207,   88,  113,  264,  261,  172,  153,  128,  115,  177,  162,  174,   92,  230,  150,  122,   85,  188,  227,  244,  200,  259,  196,  176,  185,  180,  173,   71,  262,  177,  246,  106,  120,  136,  161,  165,  161,  132,  116,  236,   90,  129,  216 ],
[  105,  239,  191,  118,  109,  102,  218,  155,  158,  113,  145,  127,   91,  235,  101,  185,  103,  154,  236,  165,   68,  196,   92,  145,   73,  178,  224,  202,  150,  261,   89,  139,  191,  225,  155,  122,  226,   86,   89,   76,  134,  174,  256,  130,  253,  101,   79,  173,  102,  188,  192,  121,  122,  176,  166,  144,  200,   97,  252,  192,  176,  187,  112,  160 ]
];
c=[
[ 1, 1, 4, 4, 7, 5, 4, 1, 4, 7, 7, 6, 7,10,10, 1, 4, 6, 5, 8, 1, 9, 8, 8, 4, 3, 7, 2, 7, 7, 2, 2, 5, 5, 5, 9,10, 5, 9, 9, 5, 6, 2, 2, 6, 7,10, 9, 6, 6, 8, 6, 3, 6, 4, 8, 8, 7, 5, 5, 5, 7,10, 6 ],
[10,10, 4,10,10,10,10, 1, 4, 1, 8, 4,10, 4, 3, 3, 7, 3, 3, 3, 2, 2, 2, 7, 9, 7, 9, 2, 9, 5, 5, 1, 2, 9, 6, 9, 2, 9, 5, 9, 3, 9, 1, 2, 5, 1, 9, 2, 8, 5, 6, 2, 2, 5, 5, 6,10, 5, 7, 2, 9, 6,10, 6 ],
[ 1,10, 7, 1, 9, 2, 4, 7, 9, 7, 1,10, 4, 8, 3, 3, 2,10,10, 2, 3, 9,10, 9, 1, 3, 2, 4, 3, 3, 6, 7, 3, 1,10, 7, 9, 6, 5, 9, 6, 5, 5, 2, 5, 1, 7, 2, 6, 3, 5, 5, 5, 3, 7, 8, 6, 8, 3, 4, 5, 2, 5,10 ],
[ 1, 2, 8, 1, 7, 2, 6, 3,10, 9, 1, 8, 3, 3, 8, 3, 3, 4, 5, 6, 3, 2, 2, 1, 8, 6, 5, 8, 8, 4, 9, 7, 6, 1, 4, 9, 2, 7,10,10, 9, 3, 8, 1,10, 9, 1, 2, 3, 3, 6, 3, 9, 5, 7, 2, 4, 8,10, 6, 5, 3,10, 9 ]
];
mm_=[
[    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ]
[    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ]
[    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ]
[    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ]
];


%run('1.m') %c'est pas la bonne commande mais je m'en souviens plus 

%p.s: les fichiers en seront pas dans le même dossier 

% Le problème (P)

% La fonction objective 
object= @(x) objective(x,m,n,a,c); %a et c donnés par les scripts 

% Les contraintes
[A,b]=contraintesPproblem(AA, MM, mm_, m, n);

%Les variables 
ub = ones(2*m*n,1)';
ub(1:m*n)=Inf; %en suivant les instructions sur la page info de ga
lb = zeros(2*m*n,1)';
Aeq = [];
Beq = [];
Idiscrete = m*n+1:2*m*n;

f = zeros(1,2*m*n); % assumes x0 is the initial point
x0 = linprog(f,A,b,Aeq,Beq,lb,ub);

%opti = gaoptimset('InitialPopulation',x0', 'TimeLimit',60, 'PopulationSize', m*n*2*10,'PopulationType','doubleVector');
opti = gaoptimset('TimeLimit',60, 'PopulationSize', m*n*2*10,'PopulationType','doubleVector');
[X,fval,exitFlag,Output] = ga(object, 2*m*n, A,b,Aeq,Beq,lb,ub,[],Idiscrete,opti);


fprintf('The number of generations was : %d\n', Output.generations);
fprintf('The number of function evaluations was : %d\n', Output.funccount);
fprintf('The best function value found was : %g\n', fval);

fid=fopen('dataB1.txt','w');
   fprintf(fid,'matrice X trouvé avec ga \n');
   fprintf(fid,'%d ',X);
   fprintf(fid, '\nresultat optimal : \n');
   fprintf(fid, '%d\n\n', fval);
fclose(fid);
%{
%patternsearch 
X = patternsearch(object,x0,A,b,[],[],lb,ub);
   fprintf(fid,'matrice X trouvé avec patternsearch \n');
   fprintf(fid,'%d ',X);
   fprintf(fid, '\nresultat optimal : \n');
   fprintf(fid, '%d\n\n', object(X));

f = [reshape(a,1,n*m) reshape(c,1,m*n)];
   [X,fval] = linprog(f,A,b,[],[],lb,ub);
   fprintf(fid,'matrice X trouvé avec linprog \n');
   fprintf(fid,'%d ',X);
   fprintf(fid, '\nresultat optimal : \n');
   fprintf(fid, '%d\n\n', fval);
   
fclose(fid);
%}
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

function [A,b]=contraintesPproblem( AA, MM, mm_, m, n)

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

iterateur=iterateur+m;

% a gauche 
for i=1:m
    for j=1:n 
        A(iterateur+(i-1)+j,(i-1)+j)=1;
        A(iterateur+(i-1)+j,n*m+(i-1)+j)=-MM(i); %n*m pour passer aux y
        b(iterateur+(i-1)+j)=0;
    end     
end

iterateur=iterateur+m*n;
%a droite

for i=1:m
    for j=1:n 
        A(iterateur+(i-1)+j,(i-1)+j)=-1;
        A(iterateur+(i-1)+j,n*m+(i-1)+j)=mm_(i,j); 
        b(iterateur+(i-1)+j)=0;
    end     
end

end 
