include("4.1.R_Replicates.jl")



######################################################## TESTING: FUNCTIONS AND ESTIMATOR #################################################





######################################################## BOSTON HOUSING DATASET ########################################################
# DATA IMPORT AND TRANSFORMATION
# https://github.com/Jiarui-Du/ubmcqmc/blob/master/data_model.m

folder_path = raw"C:\Users\mussi\Documents\Manhattan\Leuven\UnMCQMC (He and Du)\Codes\Data\\"
df_boston = CSV.read(folder_path * "DB_Boston.txt", DataFrame)
M_boston = Matrix(df_boston)
r,c = size(M_boston)

Y = log.(M_boston[:, end])
X = [ones(r,1)  M_boston[:, 1:end-1]]           # incercept
X[:,6] = X[:,6].^2
X[:,7] = X[:,7].^2
X[:,9] = log.(X[:,9])
X[:,10] = log.(X[:,10])
X[:,14] = log.(X[:,14])

# Parameter initialisation -- pag 14 (He, and Du 2024) --
p = size(X, 2); b0= zeros(p); B0= Matrix{Float64}(I, p, p)*100; n0=5.0; s0=0.01
N = 20000
WCUD_seq = rand(p+1, N)
blm = UnMCQMC_linear(p, Y, X, b0, B0, n0, s0)
@time Mean_Posterior = F_k_m(N,CUD_seq, blm, 500, x -> x)

# 0.897079 seconds

# Term     Coef.
# x1    4.562779225948906
# x2   -0.01182486512073814
# x3    0.00005050747653669512
# x4    0.00006312596509550238
# x5    0.09141417758583727
# x6   -0.6337244878998968
# x7    0.0063009519259636035
# x8    0.00011615745984262865
# x9   -0.19030643719345725
# x10   0.09309843445569665
# x11  -0.00040631302110001935
# x12  -0.03127872719794009
# x13   0.0003645383979063723
# x14  -0.372841298193898
# Sigma   0.03314326957355439

# Mean Square Error 
mean((Y - X * Mean_Posterior[1:end-1]).^2)


# OLS 
@time ols = lm(X[:,1:end], Y)

#   1.021749 seconds
# -----------------------------------------------------------------
#   Term     Coef.       Std. Error     t       Pr(>|t|)   Lower 95%    Upper 95%
# -----------------------------------------------------------------
# x1    4.56578      0.154755      29.50    <1e-99   4.26172       4.86984
# x2   -0.011835     0.00124547    -9.50    <1e-19  -0.0142821    -0.00938789
# x3    5.17744e-5   0.000506807    0.10    0.9187  -0.000944004   0.00104755
# x4    4.30655e-5   0.00237506     0.02    0.9855  -0.00462346    0.00470959
# x5    0.0912068    0.0332114      2.75    0.0062   0.0259528     0.156461
# x6   -0.634756     0.113236      -5.61    <1e-07  -0.857242     -0.41227
# x7    0.00628201   0.00131369     4.78    <1e-05   0.00370087    0.00886316
# x8    0.000110762  0.00052691     0.21    0.8336  -0.000924515   0.00114604
# x9   -0.190859     0.033404      -5.71    <1e-07  -0.256491     -0.125226
# x10   0.0930878    0.0193767      4.80    <1e-05   0.0550163     0.131159
# x11  -0.000406041  0.000123814   -3.28    0.0011  -0.000649313  -0.00016277
# x12  -0.0313095    0.00501913    -6.24    <1e-09  -0.0411712    -0.0214479
# x13   0.00036292   0.000103146    3.52    0.0005   0.000160258   0.000565582
# x14  -0.372639     0.0250742    -14.86    <1e-40  -0.421905     -0.323373

# Mean Square Error
mean((Y- X*coef(ols)).^2)




########################################################## CALIFORNIA HOUSING DATASET ######################################################
# DATA IMPORT AND TRANSFORMATION
# https://github.com/Jiarui-Du/ubmcqmc/blob/master/data_model.m

folder_path = raw"C:\Users\mussi\Documents\Manhattan\Leuven\PhD Codes\UnMCQMC (He and Du)\Data\\"
df_california = CSV.read(folder_path * "DB_California_Housing.txt", DataFrame)

#=
        Data = load('./data/california_housing_data.txt');
        data = Data;
        households = round(Data(:,5)./Data(:,6)); % households
        data(:,3) = round(Data(:,3).*households); % total_rooms
        data(:,4) = round(Data(:,4).*households); % total_bedrooms
        colNames = ["median_income","housing_median_age","total_rooms","total_bedrooms",...
            "population","avePopulation","latitude","longitude"];
        Y = log(data(:,end).*100000);
        y = log(data(:,end));
        m = length(y);
        X(:,1) = ones(m,1);
        X(:,2:4) = [data(:,1),data(:,1).^2,data(:,1).^3];% Income
        X(:,5) = log(data(:,2));
        X(:,6) = log(data(:,3)./data(:,5));
        X(:,7) = log(data(:,4)./data(:,5));
        X(:,8) = log(data(:,6));
        X(:,9) = log(households);
        p = size(X,2);
        d = p+1;
        b0 = zeros(p,1);
        B0 = 100*eye(p);
        n0 = 5;
        s0 = 0.01;
=# 