%% Importar Bases de datos (csv)
stock = {'AMXL'; 'WALMEX'; 'TLEVISACPO'; 'GMEXICOB'; 'GFNORTEO'; 'CEMEXCPO'; 'ALFAA'; 'PENOLES'; 'GFINBURO'; 'ELEKTRA'; 'BIMBOA'; 'AC'; 'KIMBERA'; 'LABB'; 'LIVEPOL1'; 'ASURB'; 'GAPB'; 'ALPEKA'; 'GRUMAB'; 'ALSEA'; 'GCARSOA1';  'PINFRA'};

n = size(stock,1);

% for i=1:n
%     txt = sprintf('prices.%s =  readtable(''../Data/%s.MX.csv'',''ReadVariableNames'',true)',stock{i},stock{i});
%     eval(txt) % Importa base de datos csv a formato de tablas y los guarda todas en una estructura 'struct' 
% end
% 
% save prices.mat prices

load prices.mat

%% Calcular rendimientos de Cierres. 

rend = [];
for i=1:n
    x = eval(sprintf('prices.%s.Close',stock{i}));
    eval(sprintf('rend.%s = diff(x)./x(1:end-1,:)',stock{i}));
end





