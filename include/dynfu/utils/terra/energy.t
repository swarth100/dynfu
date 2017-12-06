local D,N = Dim("D",0), Dim("N",1)

local rotation = Unknown("rotation", opt_float3,{D},0)
local translation = Unknown("translation", opt_float3,{D}, 1)

local canonicalVertices = Array("canonicalVertices", opt_float3,{N}, 2)
local canonicalNormals = Array("canonicalNormals", opt_float3,{N}, 3)

local liveVertices = Array("liveVertices", opt_float3,{N},4)
local liveNormals = Array("liveNormals", opt_float3,{N},5)

local transformationWeights = Array("weights", opt_float8, {N}, 6)

local G = Graph("dataGraph", 7,
                    "v", {N}, 8,
                    "n0", {D}, 9,
                    "n1", {D}, 10,
                    "n2", {D}, 11,
                    "n3", {D}, 12,
                    "n4", {D}, 13,
                    "n5", {D}, 14,
                    "n6", {D}, 15,
                    "n7", {D}, 16)

local totalTranslation = 0

nodes = {0,1,2,3,4,5,6,7}

for _,i in ipairs(nodes) do
    totalTranslation = totalTranslation + transformationWeights(G.v)(i) * translation(G["n"..i])
end

Energy(liveVertices(G.v) - canonicalVertices(G.v) - totalTranslation)
