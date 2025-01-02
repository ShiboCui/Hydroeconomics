%%%%% Data Preparation %%%%%
%Input the flood losses data, named 'loss'
%In our research, the 'loss' matrix contains 30 rows and 29 columns
%The 30 rows represent 30 provinces
%The 29 columns represent 29 years

%%%%% Correlation coefficients Cal %%%%%
losscorr=corr(loss','type','Pearson');

%%%%% Individual premiums and risk reserves without bundling Cal %%%%%
rows = size(loss, 1); 
EL = zeros(rows, 1);
STL = zeros(rows, 1);
p = zeros(rows, 1);

for i = 1:rows
    row = loss(i, :);
    non_zero_elements = row(row ~= 0); 
    non_zero_count = length(non_zero_elements); 
    if non_zero_count > 0
        EL(i) = mean(non_zero_elements);
        STL(i) = std(non_zero_elements);
    else
        EL(i) = NaN;
        STL(i) = NaN;
    end
    p(i) = non_zero_count / length(row);
end

for j=1:1:rows
    NEL=log(EL(j))-0.5*log(1+STL(j)^2/EL(j)^2);
    NSTL=(log(1+STL(j)^2/EL(j)^2))^0.5;
    funlog=@(L) (p(j)*lognpdf(L,NEL,NSTL));
    premiums(j)=0.85*(mean(loss(j,:))+0.25*std(loss(j,:)));
    reserves(j)= fzero(@(z)(integral(funlog,0,z)-p(j)+0.005),0.05);
end


%%%%% Individual premiums with Shapley value Cal %%%%%
az=rows;
bz = az - 1;
num_sharpley = zeros(az, 1); 

num_cores = 16;  
if isempty(gcp('nocreate'))
    parpool(num_cores); 
end

parfor k = 1:az
    loss=loss0;
    loss([k, az], :) = loss([az, k], :);

    num = zeros(1, 2^bz); 
    sp = zeros(1, 2^bz);  

    for i = 1:(2^bz)
        vector = dec2bin(i-1, bz) - '0';  

        lossa = loss(1:bz, :) .* vector';  
        lossb = sum(lossa, 1);             
        stdb = 0.85 * (mean(lossb) + 0.25 * std(lossb));
        stdc = 0.85 * (mean(lossb + loss(az,:)) + 0.25 * std(lossb + loss(az,:)));
        
        num(i) = sum(vector);  
        sp(i) = stdc - stdb;   
    end


    numa = zeros(1, 2^bz);
    for i = 1:(2^bz)
        for j = 0:bz
            if num(i) == j
                numa(i) = factorial(j) * factorial(bz - j);
            end
        end
    end

    num_sharpley(k) = sum((numa .* sp)) / factorial(az);  
end

a = num_sharpley;

%%%%% Risk aversion and willingness Cal %%%%%
for j=1:30
    for i=1:500
        EL =mean(loss(j,:));  
        STL = std(loss(j,:));  
        A = max(loss(j,:));
        Pm = PM0(j);  
    
        NEL=log(EL)-0.5*log(1+STL^2/EL^2);
        NSTL=(log(1+STL^2/EL^2))^0.5;
        p_L=@(L) lognpdf(L,NEL,NSTL);
    
        b = 0.01*i;  
        U = @(x) x.^(1-b)/(1-b); 
      
        L_max = A-0.01;  
        L_min = 0.01; 
      
        UW_value = integral(@(L) U(A-L) .* p_L(L), L_min, L_max);  
        UY_value = integral(@(L) U(A-0.1*L-Pm) .* p_L(L), L_min, L_max);  
      
        pan(i)=UY_value-UW_value;
    end
        panbie(:,j)=pan';
end

    mu = 0;     
    sigma = 1;  
    willingness = 1 - normcdf(threshold, mu, sigma);
