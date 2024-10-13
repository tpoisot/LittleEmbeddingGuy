using SpeciesInteractionNetworks
using DataFrames
import CSV
using GLMakie
using StatsBase
using SDeMo

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

outputs = DataFrame(out=Float64[], in=Float64[], co=Float64[], rank=Int[], mcc=Float64[], thr=Float64[])

for dout in 0:10
    @info "out: $(dout)"
    for din in 0:(10-dout)
        @info "\tinL $(din)"
        for dco in 0:(10-dout+din)
            @info "\t\tco: $(dco)"
            for r in 1:10
                @info "\t\t\trnk: $(r)"
                for rep in 1:5
                    try
                        tv = sample(training_viruses, 40; replace=false)
                        th = sample(training_hosts, 40; replace=false)
                        S = simplify(subgraph(T, tv, th))
                        O = render(Quantitative{Float32}, copy(S))
                        P = linearfilter(S; α=[0.0, dout, din, dco])

                        impute!(O, render(Quantitative{Float32}, S), P; rank=r)
                        thr = LinRange(extrema(Array(O))..., 100)
                        C = [ConfusionMatrix(vec(Array(O)), vec(Array(S)), t) for t in thr]
                        bmcc, idx = findmax(mcc.(C))
                        push!(outputs, [dout, din, dco, r, bmcc, thr[idx]])
                    catch error
                        continue
                    end
                end
            end
        end
    end
end


# Plot the rank v. mcc tuning curve
scatter(outputs.r, outputs.mcc)
