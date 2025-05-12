module FRG

# Include your package code here.
include("SD.jl")
include("Plotter.jl")
include("FRGsolverMultiD.jl")
include("FRGerrorMultiD.jl")
include("FRGplotterMultiD.jl")

# Include structs for export here
export AbstractSemidiscretization
export AbstractSemidiscretizationCell
export AbstractSemidiscretizationBasis
export AbstractSemidiscretizationQuadrature
export AbstractSemidiscretizationElliptic

export AbstractTransportSemidiscretization
export AbstractTransportSemidiscretizationCell
export AbstractTransportSemidiscretizationBasis
export AbstractTransportSemidiscretizationQuadrature

export FRGSemidiscretizationDGMultiD
export FRTSemidiscretizationDGMultiD
export FRGSemidiscretizationMultiDCell
export FRGSemidiscretizationMultiDBasis
export FRGSemidiscretizationMultiDQuadrature

end # module FRG
