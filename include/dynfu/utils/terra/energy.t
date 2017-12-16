-- function definitions

function huberPenalty(a, delta) -- delta = 0.00001
    if lesseq(abs(a), delta) then
        return a * a / 2
    else
        return delta * abs(a) - delta *  delta / 2
    end
end

function tukeyPenalty(x, c) -- c = 0.01
    if lesseq(abs(x), c) then
        return x * pow(1.0 - (x * x) / (c * c), 2)
    else
        return 0
    end
end

-- energy specifciation

local D,N = Dim("D",0), Dim("N",1)

local rotation = Unknown("rotation",opt_float3,{D},0)
local translation = Unknown("translation",opt_float3,{D},1)

local canonicalVertices = Array("canonicalVertices",opt_float3,{N},2)
local canonicalNormals = Array("canonicalNormals",opt_float3,{N},3)

local liveVertices = Array("liveVertices",opt_float3,{N},4)
local liveNormals = Array("liveNormals",opt_float3,{N},5)

local transformationWeights = Array("transformationWeights",opt_float8,{N},6)

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
local totalRotation = 0

nodes = {0,1,2,3,4,5,6,7}

for _,i in ipairs(nodes) do
    totalTranslation = totalTranslation + translation(G["n"..i])
    -- totalTranslation = totalTranslation + transformationWeights(G.v)(i) * translation(G["n"..i]) -- FIXME (dig15): use the transformation weights
    -- totalRotation = totalRotation + transformationWeights(G.v)(i) * rotation(G["n"..i]) -- FIXME (dig15): use rotations
end

Energy(liveVertices(G.v) - canonicalVertices(G.v) - totalTranslation)
-- Energy(tukeyPenalty(liveVertices(G.v) - canonicalVertices(G.v) - totalTranslation, 0.01)) -- FIXME (dig15): works for real data but will cause tests to fail
