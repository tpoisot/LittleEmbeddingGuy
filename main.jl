using SpeciesInteractionNetworks
using DataFrames
import CSV

# Load virionette
vir = DataFrame(CSV.File("virionette.csv"; types=String))

# Make a network
nodes = Bipartite(unique(vir.virus_genus), unique(vir.host_species))
B = SpeciesInteractionNetwork(nodes, Binary(zeros(Bool, size(nodes))))

# Loop through the dataframe
for row in eachrow(vir)
    B[row.virus_genus, row.host_species] = true
end

P = linearfilter(B)

N = render(Quantitative{Float32}, B)

# Impute
function impute(N, P, virus, host; rank=4)
    v0 = N[virus, host]
    N[virus, host] = P[virus, host]
    for _ in 1:20
        X = rdpg(N, rank)
        if X[virus, host] == N[virus, host]
            break
        end
        N[virus, host] = X[virus, host]
    end
    pred = (virus, host, v0, N[virus, host])
    N[virus, host] = v0
    return pred
end

impute(N, P, "Vesiculovirus", "Peromyscus leucopus"; rank=2)
