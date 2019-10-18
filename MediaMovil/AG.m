% Disculpa por todas las faltas de ortografia, Matlab no permite acentos. 

clear all;
close all;
clc;

tic;
%% funcion
% para n variables, se utiliza el formato x(:,i), donde i es la variable
% entre 1 y n. Algoritmo genetico busca maximizar. 

% funcion x1^2+x2^4+10 debera¡ escribirse como:
% 'x(:,1).^2 + x(:,2).^4 + 10'

a = 100000;

r1 = 'max(3*x(:,1) + 6*x(:,2) + 2*x(:,3) - 600, 0)';
r2 = '- min(x(:,1) + x(:,2) + x(:,3) - 240, 0)';
r3 = ' - min(x(:,1),0) - min(x(:,2)-1,0) - min(x(:,3),0) + max(x(:,3)-140,0)';
rest = 'eval(r1) + eval(r2) + eval(r3)';

func = '(4*x(:,1) + 12*x(:,2) + 2*x(:,3)) - a*eval(rest)';


%% Parametros
nv = 3; % Numero de variables
iteraciones = 10000;

%Se pueden tomar valores unitarios o como vectores de n dimensiones. 
x_min = [0 1 0]; % x min
x_max = [500 500 500]; % x max
tp = [.01 .01 .01]; % tamanio de paso


elmnts = (x_max-x_min)./tp+1; % Elementos 
nbits = ceil(log2(elmnts)); % Numero de bits

%% Generar la poblacion
np=32; % Numero de pobladores
mp = sum(nbits);% ancho matriz pobladores (suma de bits de cada variable)


xe = zeros(3/2*np,nv); % X perteneciente a los enteros positivos (2^n)
for i=1:nv
    xe(:,i) = randi([1,2^nbits(i)-1],3/2*np,1); % X enteros
end

x = xe.*tp + x_min; % X perteneciente a los reales

acum = cumsum([1 nbits]); % acumulados en binarios
xb = zeros(3/2*np,sum(nbits)); % x en binarios
hb = zeros(np,sum(nbits)); % hijos binarios
he = zeros(np,nv); % hijos enteros

hist = zeros(iteraciones,1);


%% Algoritmo Genetico

for k=1:iteraciones
    fx = eval(func); % evaluamos la funcion
    
    for i=1:nv
       xb(:,acum(i):acum(i+1)-1) = de2bi(xe(:,i),nbits(i)); % convertimos a binarios
    end
    
    %% Seleccion.
    
    [out, idx] = sort(fx);    
%     pb = xb(idx,:); % todos los padres en binario  
    p = x(idx(np+1:end),:); % Padres
    pe = xe(idx(np+1:end),:); % Padres enteros
    pb = xb(idx(np+1:end),:); % la mitad de los padres (los mejores)
   
    hist(k) = mean(out(np:end));
    %% Cruzamiento por mascara
    
    sel = randi([1,np/2],np,sum(nbits)); % Mascaras
    for i=1:sum(nbits)
        hb(:,i) = pb(sel(:,i),i); % hijos binarios cruzados
    end

    %% Mutacion
    
    pct = .075; % muta el pct porciento de los digitos de los hijos
    mut = rand(np,sum(nbits)); % Crea matriz de mutacion
    hb(mut<pct) = abs(hb(mut<pct)-1); % hijos binarios mutados
    
    for i=1:nv
        he(:,i) = bi2de(hb(:,acum(i):acum(i+1)-1)); % hijos a enteros
    end
    h = he.*tp + x_min; % hijos

    x = [h;p]; % Vector x a evaluar.
    xe = floor((x - x_min)./tp);
end   

disp(fx)   
plot(hist)
toc;
