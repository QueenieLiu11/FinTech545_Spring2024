using DataFrames
using Plots
using Distributions
using CSV
using Dates
using LoopVectorization
using LinearAlgebra
using StatsBase
using JuMP
using Ipopt
using QuadGK
using Random
using StatsPlots

include("../../library/return_calculate.jl")
include("../../library/fitted_model.jl")
include("../../library/simulate.jl")
include("../../library/RiskStats.jl")
include("../../library/ewCov.jl")


# Problem 2
problem1 = CSV.read("Project/problem1.csv",DataFrame).x
m = mean(problem1)
problem1 .-= m

s = sqrt(ewCovar([problem1 problem1],0.97)[1,1])
n = Normal(0.0,s)
var_n = VaR(n)
es_n = ES(n)

tm = fit_general_t(problem1)
var_t = VaR(tm.errorModel)
es_t = ES(tm.errorModel)

var_h = VaR(problem1)
es_h = ES(problem1)

density(problem1,label="Historical",color=:black)
x = [i for i in extrema(problem1)[1]:.001:extrema(problem1)[2]]
plot!(x,pdf.(n,x),label="Normal", color=:red)
plot!(x,pdf.(tm.errorModel,x),label="T Distribution",color=:blue)

vline!([-var_n],color=:red,style=:dash,label="")
vline!([-var_t],color=:blue,style=:dash,label="")
vline!([-var_h],color=:black,style=:dash,label="")

vline!([-es_n],color=:red,style=:dot,label="")
vline!([-es_t],color=:blue,style=:dot,label="")
vline!([-es_h],color=:black,style=:dot,label="")

toPrint = DataFrame(
    :Model=>["Normal","T","Historical"],
    :VaR=>[var_n,var_t,var_h],
    :ES=>[es_n,es_t,es_h]
)
println(kurtosis(problem1))
# Row │ Model       VaR        ES       
# │ String      Float64    Float64
# ─────┼─────────────────────────────────
# 1 │ Normal      0.0902895  0.113227
# 2 │ T           0.0755959  0.112338
# 3 │ Historical  0.0773653  0.115897

# 2.414872528860026

#Data is non-normal.  The T distribution fits well base on the graph.
# Normal VaR is larger than the T VaR as expected given the 
# excess kurtosis.  ES values are similar, likely an artifact
# of the data.

#problem 3
prices = CSV.read("Project/DailyPrices.csv",DataFrame)
returns = return_calculate(prices,dateColumn="Date")
returns = select!(returns,Not([:Date]))
rnames = names(returns)

currentPrice = prices[size(prices,1),:]

portfolio = CSV.read("Project/portfolio.csv",DataFrame)

stocks = portfolio.Stock

tStocks = filter(r->r.Portfolio in ["A","B"],portfolio)[!,:Stock]
nStocks = filter(r->r.Portfolio in ["C"],portfolio)[!,:Stock]

#remove the mean from all returns:
for nm in stocks
    v = returns[!,nm]
    returns[!,nm] = v .- mean(v)
end

fittedModels = Dict{String,FittedModel}()

for s in tStocks
    fittedModels[s] = fit_general_t(returns[!,s])
end
for s in nStocks
    fittedModels[s] = fit_normal(returns[!,s])
end

U = DataFrame()
for nm in stocks
    U[!,nm] = fittedModels[nm].u
end
R = corspearman(Matrix(U))

#what's the rank of R
evals = eigvals(R)
if min(evals...) > -1e-8
    println("Matrix is PSD")
else
    println("Matrix is not PSD")
end

#simulation
NSim = 50000
simU = DataFrame(
            #convert standard normals to U
            cdf(Normal(),
                simulate_pca(R,NSim)  #simulation the standard normals
            )   
            , stocks
        )

simulatedReturns = DataFrame()
Threads.@threads for stock in stocks
    simulatedReturns[!,stock] = fittedModels[stock].eval(simU[!,stock])
end

#Protfolio Valuation
function calcPortfolioRisk(simulatedReturns,NSim)
    iteration = [i for i in 1:NSim]
    values = crossjoin(portfolio, DataFrame(:iteration=>iteration))

    nVals = size(values,1)
    currentValue = Vector{Float64}(undef,nVals)
    simulatedValue = Vector{Float64}(undef,nVals)
    pnl = Vector{Float64}(undef,nVals)
    Threads.@threads for i in 1:nVals
        price = currentPrice[values.Stock[i]]
        currentValue[i] = values.Holding[i] * price
        simulatedValue[i] = values.Holding[i] * price*(1.0+simulatedReturns[values.iteration[i],values.Stock[i]])
        pnl[i] = simulatedValue[i] - currentValue[i]
    end
    values[!,:currentValue] = currentValue
    values[!,:simulatedValue] = simulatedValue
    values[!,:pnl] = pnl

    values[!,:Portfolio] = String.(values.Portfolio)
    aggRisk(values,[:Portfolio])[:,[:Portfolio,:VaR95, :ES95]]
end
risk = calcPortfolioRisk(simulatedReturns,NSim)
# 4×3 DataFrame
#  Row │ Portfolio  VaR95     ES95     
#      │ String     Float64   Float64
# ─────┼───────────────────────────────
#    1 │ A           8074.85  10589.4
#    2 │ B           6791.12   8892.39
#    3 │ C           5865.21   7317.02
#    4 │ Total      20468.2   26436.8

covar = ewCovar(Matrix(returns),.94)
simulatedReturns = DataFrame(simulate_pca(covar,NSim),rnames)
risk_n  = calcPortfolioRisk(simulatedReturns,NSim)
rename!(risk_n,[:VaR95=>:Normal_VaR, :ES95=>:Normal_ES])

leftjoin!(risk,risk_n,on=:Portfolio)
# 4×5 DataFrame
#  Row │ Portfolio  VaR95     ES95      Normal_VaR  Normal_ES 
#      │ String     Float64   Float64   Float64?    Float64?
# ─────┼──────────────────────────────────────────────────────
#    1 │ A           8074.85  10589.4      5694.88    7142.21
#    2 │ B           6791.12   8892.39     4514.78    5654.81
#    3 │ C           5865.21   7317.02     3819.37    4759.42
#    4 │ Total      20468.2   26436.8     13628.1    17086.8

# Compared to the same method as last week, the VaR and ES of the 
# portfolio signiantly increases.  Curiously, the normal portfolio C
# is significantly larger.  This could be the lambda or the 
# spearman correlation.  Run the portfolio with an unweighted covar

covar = cov(Matrix(returns))
simulatedReturns = DataFrame(simulate_pca(covar,NSim),rnames)
risk_n2  = calcPortfolioRisk(simulatedReturns,NSim)
rename!(risk_n2,[:VaR95=>:Normal2_VaR, :ES95=>:Normal2_ES])
leftjoin!(risk,risk_n2,on=:Portfolio)

# 4×7 DataFrame
#  Row │ Portfolio  VaR95     ES95      Normal_VaR  Normal_ES  Normal2_VaR  Normal2_ES 
#      │ String     Float64   Float64   Float64?    Float64?   Float64?     Float64?
# ─────┼───────────────────────────────────────────────────────────────────────────────
#    1 │ A           8074.85  10589.4      5694.88    7142.21      7962.09     9960.41
#    2 │ B           6791.12   8892.39     4514.78    5654.81      6727.92     8409.26
#    3 │ C           5865.21   7317.02     3819.37    4759.42      5639.28     7075.68
#    4 │ Total      20468.2   26436.8     13628.1    17086.8      20069.8     25100.5

# The VaR using the unweighted covariance is roughly the same as the new methodology
# but the ES is larger with the new methodology, as expected