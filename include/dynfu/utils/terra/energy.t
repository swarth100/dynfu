-- TODO (dig15): rotate canonical vertex node by node
-- TODO (dig15): add alpha, lambda, and huber weights to regularisation graph

-- package.cpath = package.cpath .. ";/homes/dig15/df/dynfu/include/dynfu/utils/terra/?.so"
-- require 'luadq'

----- FUNCTIONS -----

-- calculates the norm of a 3-D vector
function norm(v)
  return sqrt(v:dot(v))
end

-- calculates the transformation weight given the coordinates of a vertex and a node, and the radial basis weight of the node
function calcTransformationWeight(v, dg_v, dg_w)
  return exp(-pow(norm(v - dg_v), 2) / (2 * pow(dg_w, 2)))
end

----- DATA TERM -----

local D,N = Dim("D",0), Dim("N",1)

local dg_v = Array("dg_v",opt_float3,{D},0)
local translations = Unknown("translations",opt_float3,{D},1)
local rotations = Unknown("rotations",opt_float3,{D},2)
local dg_w = Array("dg_w",opt_float,{D},3)

local canonicalVertices = Array("canonicalVertices",opt_float3,{N},4)
local canonicalNormals = Array("canonicalNormals",opt_float3,{N},5)

local liveVertices = Array("liveVertices",opt_float3,{N},6)
local liveNormals = Array("liveNormals",opt_float3,{N},7)

local dataG = Graph("dataGraph", 8,
                    "v", {N}, 9,
                    "n0", {D}, 10,
                    "n1", {D}, 11,
                    "n2", {D}, 12,
                    "n3", {D}, 13,
                    "n4", {D}, 14,
                    "n5", {D}, 15,
                    "n6", {D}, 16,
                    "n7", {D}, 17)

local tukeyBiweights = Array("tukeyBiweights",opt_float,{N},18)

local nodes = { 0, 1, 2, 3, 4, 5, 6, 7 }
local totalTranslation = 0

for _,i in ipairs(nodes) do
    local transformationWeight = calcTransformationWeight(canonicalVertices(dataG.v), dg_v(dataG["n"..i]), dg_w(dataG["n"..i]))
    totalTranslation = totalTranslation + transformationWeight * translations(dataG["n"..i])
end

Energy(sqrt(tukeyBiweights(dataG.v)) * (liveVertices(dataG.v) - canonicalVertices(dataG.v) - totalTranslation))

--- REGULARISATION TERM -----

local regG = Graph("regGraph", 19,
                   "n", {D}, 20,
                   "v0", {D}, 21,
                   "v1", {D}, 22,
                   "v2", {D}, 23,
                   "v3", {D}, 24,
                   "v4", {D}, 25,
                   "v5", {D}, 26,
                   "v6", {D}, 27,
                   "v7", {D}, 28)

local huberWeights = Array("huberWeights", opt_float, {D}, 29)

local neighbours = { 0, 1, 2, 3, 4, 5, 6, 7 }

for _,i in ipairs(neighbours) do
    Energy(((dg_v(regG["v"..i]) - translations(regG.n)) - (dg_v(regG["v"..i]) - translations(regG["v"..i]))))
end
