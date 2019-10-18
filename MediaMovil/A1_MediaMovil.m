%% Descargar Bases de datos (csv)
% stock = {'AMXL'; 'WALMEX'; 'TLEVISACPO'; 'GMEXICOB'; 'GFNORTEO'; 'CEMEXCPO'; 'ALFAA'; 'PENOLES'; 'GFINBURO'; 'ELEKTRA'; 'BIMBOA'; 'AC'; 'KIMBERA'; 'LABB'; 'LIVEPOL1'; 'ASURB'; 'GAPB'; 'ALPEKA'; 'GRUMAB'; 'ALSEA'; 'GCARSOA1';  'PINFRA'};
stock = {'AC';'ALFAA';'ALPEKA';'ALSEA';'ELEKTRA';'IENOVA';'MEXCHEM';'PENOLES';'PINFRA';'WALMEX'};

n = size(stock,1);

% hoy = datestr(date,29);
% inicio = datestr(datenum('18-Dic-2013'),29);
% for i=1:n
%     txt = sprintf('writetable(Download(''%s.mx'',inicio,hoy,''d'',''history''),''../data/%s.MX.csv'')',lower(stock{i}),stock{i})
%     eval(txt) % Importa base de datos csv a formato de tablas y los guarda todas en una estructura 'struct' 
% end


%% Importar Bases de datos (csv)

% 
% 
% for i=1:n
%     txt = sprintf('prices.%s = readtable(''../data/%s.MX.csv'',''ReadVariableNames'',true)',stock{i},stock{i});
%     eval(txt) % Importa base de datos csv a formato de tablas y los guarda todas en una estructura 'struct' 
% end
% 
% save prices.mat prices

load prices.mat


%% Cambiar 'null' por el numero anterior
% m = size(prices.ALFAA.High,1);
% for j = 2:7 
% 
% 
%     for k = 2:m
%         if length(prices.ALFAA{k,j}{1})==4
%             prices.ALFAA{k,j}{1} = prices.ALFAA{k-1,j}{1};
%         end
%     end
% end
% prices.ALFAA.Open = str2num(cell2mat(prices.ALFAA{:,2}));
% prices.ALFAA.High = str2num(cell2mat(prices.ALFAA{:,3}));
% prices.ALFAA.Low = str2num(cell2mat(prices.ALFAA{:,4}));
% prices.ALFAA.Close = str2num(cell2mat(prices.ALFAA{:,5}));
% prices.ALFAA.AdjClose = str2num(cell2mat(prices.ALFAA{:,6}));
%prices.ALFAA.Volume = str2num(cell2mat(prices.ALFAA{:,7}));

%% Calcular rendimientos de Cierres. 

rend = [];
for i=1:n-1
    x = eval(sprintf('prices.%s.Close',stock{i}));
    eval(sprintf('rend.%s = diff(x)./x(1:end-1,:)',stock{i}));
end


rclose = struct2cell(rend);
mat_rclose = zeros(size(prices.AC,1)-1,n);

for i=1:n 
    mat_rclose(:,i) = rclose{i};
end
%% 
cov_rclose = cov(mat_rclose);
W = ones(n,1)/n;


Mean = mean(mean(mat_rclose))*252
Std = ((W'*(cov_rclose*W))*252)^0.5







