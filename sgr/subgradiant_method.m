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
    f0= vecteurLagr(m,n,a,c,pik,MM);          %f0 est le vecteur correspondant à la fonction de lagrange
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
%       [A,b]=contraintesL(AA, MM, mm_, m, n);
%       f2=objectivevec(m,n,a,c);
%       [x1, theta_c] = linprog(f2,A,b,[],[],lb,ub); 
      f2=@(x) objective(x,m,n,a,c);
      [x1, theta_c] = patternsearch(f2,x,A,b,[],[],lb,ub);
      for i=m*n+1:2*m*n
          if(x1(i)~=0)
              x1(i)=1;
          end
      end
      theta_c=f2(x1)
      
      %for all j appartient J do ... end for 
      
      for j=1:J
        pik(j) = max(0,pik(j)-(rho*(theta-theta_c)/norm(gamma))*gamma(j));  
      end 
      %Set k = k + 1
      
      k= k+1;
      
      %end if 
   end 
   if(abs( object(x) - theta )/theta <= eps || k > iterLimit)
       if(k > iterLimit)
           fprintf("limite atteinte\n");
       else
           fprintf("Valeur optimale\n");
       end
       break;
   end
end
fid=fopen('dataC.txt','w');
   fprintf(fid,'matrice X trouvée \n');
   fprintf(fid,'%d ',x);
   fprintf(fid, '\n multiplicateur de Lagrange : \n');
   fprintf(fid, '%d\n\n', pik);
   fprintf(fid, '\n valeure optimale : \n');
   fprintf(fid, '%d\n\n', object(x));
fclose(fid);
end 
