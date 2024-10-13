using SpeciesInteractionNetworks
using DataFrames
import CSV
using GLMakie
using StatsBase

# Load virionette
vir = DataFrame(CSV.File("virionette.csv"; types=String))

# Make a network
nodes = Bipartite(unique(vir.virus_genus), unique(vir.host_species))
B = SpeciesInteractionNetwork(nodes, Binary(zeros(Bool, size(nodes))))

# Loop through the dataframe
for row in eachrow(vir)
    B[row.virus_genus, row.host_species] = true
end

function impute!(output, observations, template; kwargs...)
    for virus in species(observations, 1)
        for host in species(observations, 2)
            impute!(output, observations, template, virus, host; kwargs...)
        end
    end
    return output
end

function impute!(output, observations, template, virus, host; rank=4)
    v0 = observations[virus, host]
    observations[virus, host] = template[virus, host]
    for _ in 1:20
        X = rdpg(observations, rank)
        if X[virus, host] == observations[virus, host]
            break
        end
        observations[virus, host] = X[virus, host]
    end
    output[virus, host] = observations[virus, host]
    observations[virus, host] = v0
    return output
end

# Training hosts and viruses
training_hosts = unique(vir.host_species[findall(!isequal("Chiroptera"), vir.host_order)])
training_viruses = unique(vir.virus_genus[findall(!isequal("Betacoronavirus"), vir.virus_genus)])

# Network we can subsample for training
T = subgraph(B, training_viruses, training_hosts)

tv = sample(training_viruses, 40; replace=false)
th = sample(training_hosts, 40; replace=false)
S = simplify(subgraph(T, tv, th))
O = render(Quantitative{Float32}, copy(S))
P = linearfilter(S)

impute!(O, render(Quantitative{Float32}, S), P; rank=10)

thr = LinRange(extrema(Array(O))..., 100)
TP = [sum((Array(O) .>= t) .& Array(S)) for t in thr]
FP = [sum((Array(O) .>= t) .& .!Array(S)) for t in thr]
TN = [sum((Array(O) .< t) .& .!Array(S)) for t in thr]
FN = [sum((Array(O) .< t) .& Array(S)) for t in thr]

TPR = TP ./ (TP .+ FN)
TNR = TN ./ (TN .+ FP)
Y = TPR .+ TNR .- 1.0

lines(thr, Y)

# If we want to do actual out-of-bag prediction, we can! We just need to do the training
# without Betacoronaviruses and Chiroptera - this is not really difficult at all. Maybe the
# proper validation scheme is actually to draw random samples of 10x10 where the viruses
# cannot be βcov, and the host cannot be a bat, and then get freaky with it
