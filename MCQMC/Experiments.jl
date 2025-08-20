










##################################### Probit Regression  #############################################
## DATA IMPORT AND TRANSFORMATION
folder_path = raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Data&Libraries\\"
df_vaso = CSV.read(folder_path * "DB_Vaso.csv", DataFrame)
rename!(df_vaso, [:x1, :x2, :y])

scatter(
    df_vaso.x1, df_vaso.x2,
    group = df_vaso.y,
    markershape = :circle,
    markersize = 6,
    palette = ["black", "red"],   # y=0 nero, y=1 rosso
    xlabel = "Volume of air inspired",
    ylabel = "Rate of air inspired",
    title = "Vasoconstriction data",
    legend = false
)

Y = df_vaso[:,end]; X = Matrix(df_vaso[:, 1:2])

##  GLM: Probit Model 
@time probit = glm(@formula(y ~ x1+ x2), df_vaso, Binomial(), ProbitLink())

# 3.135882 seconds (2.95 M allocations: 148.579 MiB, 1.62% gc time, 99.20% compilation time)

# y ~ 1 + x1 + x2

# Coefficients:    coeff > 0, positive effect on p(Y=1|X)
# ────────────────────────────────────────────────────────────────────────
#                 Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
# ────────────────────────────────────────────────────────────────────────
# (Intercept)  -5.19454    1.59339   -3.26    0.0011  -8.31752    -2.07156
# x1            2.11804    0.717416   2.95    0.0032   0.711934    3.52415
# x2            1.47643    0.46176    3.20    0.0014   0.571401    2.38147
# ────────────────────────────────────────────────────────────────────────

aic(probit); bic(probit)