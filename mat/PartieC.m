function PartieC()

%On a un jeu de donnée de petite taille

n=  8;
m=   4;
MM=[  112,  109,  131,  318 ];
AA=[    6,    8,    8,   30,   14,   32,    2,    2];
a=[
[  251,  144,  116,  260,  167,  130,  187,  251],
[  253,  112,  234,  182,  194,   89,  100,  158],
[  144,  147,  220,  252,  254,  119,  187,  162],
[  105,  239,  191,  118,  109,  102,  218,  155]
];
c=[
[ 1, 1, 4, 4, 7, 5, 4, 1],
[10,10, 4,10,10,10,10, 1],
[ 1,10, 7, 1, 9, 2, 4, 7],
[ 1, 2, 8, 1, 7, 2, 6, 34]
];
mm_=[
[    0,    0,    0,    0,    0,    0,    0,    0]
[    0,    0,    0,    0,    0,    0,    0,    0]
[    0,    0,    0,    0,    0,    0,    0,    0]
[    0,    0,    0,    0,    0,    0,    0,    0]
];


% La fonction objective 
object= @(x) objective(x,m,n,a,c); %a et c donnés par les scripts 

% Les contraintes
[A,b]=contraintesL(AA, MM, mm_, m, n);

%Les variables 
ub = ones(2*m*n+m*n*n,1);
ub(1:m*n)=Inf;
ub(2*m*n+1:end)=Inf;
lb = zeros(2*m*n+m*n*n,1);

eps=10e-5;
rho=1;
pi0=ones(m*n,1);
iterLimit=1000;
DualNoChangTOL=100;
x=subgradient_method(eps,rho, pi0, iterLimit,DualNoChangTOL,object,ub,lb,a,c,AA, MM, mm_,m,n);
end

function x=subgradient_method(eps,rho, pi0, iterLimit,DualNoChangTOL,object,ub,lb,a,c, AA, MM, mm_,m,n)

%eps control parameter
%tau appartient à ]0, 2[ control parameter
%pi0 initial nonnegative vector (usually the nul vector)
%iterLimit number of maximum iterations
%DualNoChangTOL accepted number of iterations without change in the value of the dual function

%Ensure: A local optimum of theta and an approximate primal solution x
%Set k = 1 and t = 1
%Set betak = - inf and pik = pi0

k=1; 
t=1; 
beta=-inf; % infinie
pik=pi0; 
J = m*n;
gamma=zeros(J,1);

% critères d'arrêts

while ( 1 )

    % Solve the Lagrangian relaxation: theta(pi) = min L(x,pi) avec x
    % appartient à F
    % set x <- argmin{L(x,pik):x appartient à F}
    f0= vecteurLagr56(m,n,a,c,pik,MM);          %f0 est le vecteur correspondant à la fonction de lagrange en ne prenant en compte que la contrainte 56
    [A,b]=contraintesLF( AA, MM, mm_, m, n);    %A et b sont les contraites F (positivité de x et z + y entre 0 et 1)
    [x, theta] = linprog(f0, A, b, [], [], lb, ub);
	for i=m*n+1:2*m*n %Permet de discretiser le vecteur y
        if(x(i)~=0)
            x(i)=1;
        end
    end
    
   %On récupère dans un vecteur toutes les valeurs des contraintes X
   gamma = vecG(x,m,n,MM);
   
   % if theta(pi^k) > beta^k then set beta^k+1 = theta(pi^k)
   % else 
   
   if theta > beta
       beta=theta; 
   
   %else if t <= DualNoChangTOL then Set t = t + 1
   else
       if t <= DualNoChangTOL
           t=t+1; 
       else 
            rho=rho/2; 
            t=1; 
       end 
   end
   
   %if ?k =0 then Stop; The current solution is optimal
   
   if gamma==zeros(J,1)
       fprintf("gamma=0\n");
       break; % on arrête l'algorithme
       
   % else thetachapeau= applyMHeuristic (x^k,f,X?F)
   else 
      [A,b]=contraintesL(AA, MM, mm_, m, n);
      f2=objectivevec(m,n,a,c);
      [x1, theta_c] = linprog(f2,A,b,[],[],lb,ub); 
%       f=@(x) objective(x,m,n,a,c);
%       [x1, theta_c] = patternsearch(f,x,A,b,[],[],lb,ub);
      for i=m*n+1:2*m*n
          if(x1(i)~=0)
              x1(i)=1;
          end
      end
      theta_c=f2'*x1
      
      %for all j appartient J do ... end for 
      
      for j=1:J
        pik(j) = max(0,pik(j)-(rho*(theta-theta_c)/norm(gamma))*gamma(j));  
      end 
      
      %Set k = k + 1
      
      k= k+1;
      
      %end if 
   end 
   if( k > iterLimit)
       if(k > iterLimit)
           fprintf("limite atteinte\n");
       else
           fprintf("Valeur optimale\n");
       end
       break;
   end
end
end 


function g=vecG(X,m,n,MM)
    g=zeros(m*n,1);
    x = zeros(m,n); % x de la fonction objective 
    y = zeros(m,n); % y deinteger_values = m*n+1:1:2*m*n; la fonction objective 
    z = zeros(m,n,n);

    for i=1:m 
        for j=1:n
            x(i,j) = X((i-1)*n+j);
            y(i,j) = X(m*n+(i-1)*n+j);
            for k=1:n
                z(i,j,k) = X(2*m*n+(i-1)*n*n+(j-1)*n+k);
            end
        end
    end
    for i=1:m
        for k=1:n
            tmp=0;
            for j=1:n
                tmp=tmp+z(i,j,k);
            end
            tmp=tmp-MM(i)*y(i,k);
            g((i-1)*n+k)=tmp;
        end
    end
    
end         %On met sous forme de vecteur les contraintes 56

function object=objective(X,m,n,a,c) % Fonction objective que l'on souhaite optimiser

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

function vec= vecteurLagr56(m,n,a,c,pik,MM) %Réprésente le vecteur f de la fonction lagrangienne du problème avec la contrainte 56

vec=zeros(2*m*n+m*n*n,1);


for i=1:m
    for j=1:n
        vec((i-1)*n+j,1)=c(i,j);
    end 
end 
for i=1:m
    for j=1:n
        vec(m*n+(i-1)*n+j,1)=a(i,j)-pik((i-1)*n+j)*MM(i);
    end 
end
for i=1:m
    for k=1:n
        for j=1:n
            vec(2*m*n+(i-1)*n*n+(k-1)*n+j,1)=pik((i-1)*n+j);
        end
    end
end
end

function object=objectivevec(m,n,a,c)       %Réprésente la fonction objective sous forme de vecteur

object=zeros(2*m*n+m*n*n,1);


for i=1:m
    for j=1:n
        object((i-1)*n+j,1)=c(i,j);
    end 
end 
for i=1:m
    for j=1:n
        object(m*n+(i-1)*n+j,1)=a(i,j);
    end 
end
end

function [A,b]=contraintesLF( AA, MM, mm_, m, n)

A=zeros(4*m*n+n,2*m*n+m*n*n);
b=zeros(4*m*n+n,1);

%{
%contrainte 55

for j=1:n %colonnes
    for i=1:m %lignes     
        A(j,j+((i-1)*n)) = -1; %somme sur la premiere colonne donc on saute le reste
        %on laisse le reste à 0
    end
    b(j) = -AA(j); %on multiplie par (-1) car ga est en inf 
end

iterateur=n; %on a remplit les n lignes avant donc on commence à n

%contrainte 56
for i=1:m
    for k=1:n 
        for j=1:n
            A(iterateur+(i-1)*n+k,2*m*n+(i-1)*n*n+(j-1)*n+k)=1;
            A(iterateur+(i-1)*n+k,m*n+(i-1)*n+k)=-MM(i);
        end 
    b(iterateur+(i-1)*n+k)=0;
    end
end 
iterateur=iterateur+m*n;

%contrainte 57
for i=1:m
    for k=1:n 
        for j=1:n
            A(iterateur+(i-1)*n+k,2*m*n+(i-1)*n*n+(j-1)*n+k)=-1;
            A(iterateur+(i-1)*n+k,(i-1)*n+j)=1;
            A(iterateur+(i-1)*n+k,m*n+(i-1)*n+k)=MM(i);
        end 
    b(iterateur+(i-1)*n+k)=MM(i);
    end
end 
iterateur=iterateur+m*n;


%contrainte 58
% a gauche 
for i=1:m
    for j=1:n 
        A(iterateur+(i-1)*n+j,(i-1)+j)=1;
        A(iterateur+(i-1)*n+j,n*m+(i-1)+j)=-MM(i); %n*m pour passer aux y
        b(iterateur+(i-1)*n+j)=0;
    end     
end

iterateur=iterateur+m*n;
%a droite
for i=1:m
    for j=1:n 
        A(iterateur+(i-1)*n+j,(i-1)+j)=-1;
        A(iterateur+(i-1)*n+j,n*m+(i-1)+j)=mm_(i,j); 
        b(iterateur+(i-1)*n+j)=0;
    end     
end
iterateur = iterateur+m*n;
%}


%contrainte 59
iterateur=0;
for i=1:m
    for k=1:n 
        for j=1:n
            A(iterateur+(i-1)*n+(j-1)*n+k,2*m*n+(i-1)*n*n+(j-1)*n+k)=1;
            A(iterateur+(i-1)*n+(j-1)*n+k,(i-1)*n+j)=-1;
            b(iterateur+(i-1)*n+(j-1)*n+k)=0;
        end 
    end
end 
end     %Contraintes X du problème L

function [A,b]=contraintesL( AA, MM, mm_, m, n)

A=zeros(4*m*n+n,2*m*n+m*n*n);
b=zeros(4*m*n+n,1);

%contrainte 55

for j=1:n %colonnes
    for i=1:m %lignes     
        A(j,j+((i-1)*n)) = -1; %somme sur la premiere colonne donc on saute le reste
        %on laisse le reste à 0
    end
    b(j) = -AA(j); %on multiplie par (-1) car ga est en inf 
end

iterateur=n; %on a remplit les n lignes avant donc on commence à n


%contrainte 56
for i=1:m
    for k=1:n 
        for j=1:n
            A(iterateur+(i-1)*n+k,2*m*n+(i-1)*n*n+(j-1)*n+k)=1;
            A(iterateur+(i-1)*n+k,m*n+(i-1)*n+k)=-MM(i);
        end 
    b(iterateur+(i-1)*n+k)=0;
    end
end 
iterateur=iterateur+m*n;


%contrainte 57
for i=1:m
    for k=1:n 
        for j=1:n
            A(iterateur+(i-1)*n+k,2*m*n+(i-1)*n*n+(j-1)*n+k)=-1;
            A(iterateur+(i-1)*n+k,(i-1)*n+j)=1;
            A(iterateur+(i-1)*n+k,m*n+(i-1)*n+k)=MM(i);
        end 
    b(iterateur+(i-1)*n+k)=MM(i);
    end
end 
iterateur=iterateur+m*n;
%contrainte 58
% a gauche 
for i=1:m
    for j=1:n 
        A(iterateur+(i-1)*n+j,(i-1)+j)=1;
        A(iterateur+(i-1)*n+j,n*m+(i-1)*n+j)=-MM(i); %n*m pour passer aux y
        b(iterateur+(i-1)*n+j)=0;
    end     
end

iterateur=iterateur+m*n;
%a droite
for i=1:m
    for j=1:n 
        A(iterateur+(i-1)*n+j,(i-1)+j)=-1;
        A(iterateur+(i-1)*n+j,n*m+(i-1)*n+j)=mm_(i,j); 
        b(iterateur+(i-1)*n+j)=0;
    end     
end
iterateur = iterateur+m*n;


%contrainte 59
for i=1:m
    for k=1:n 
        for j=1:n
            A(iterateur+(i-1)*n*n+(j-1)*n+k,2*m*n+(i-1)*n*n+(j-1)*n+k)=1;
            A(iterateur+(i-1)*n*n+(j-1)*n+k,(i-1)*n+j)=-1;
            b(iterateur+(i-1)*n*n+(j-1)*n+k)=0;
        end 
    end
end 
end     %Contraintes X inter F du problème L