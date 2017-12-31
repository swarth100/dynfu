-- function definitions

-- calculates the norm of a 3-D vector
function norm(v)
  return pow(pow(v(0), 2) + pow(v(1), 2) + pow(v(2), 2), 0.5)
end

-- calculates the dot product of 3-D vectors
function dotProduct(a, b)
  local dot = a(0) * b(0) + a(1) * b(1) + a(2) * b(2)
  return dot
end

-- calculates the transformation weight given the coordinates of a vertex and a node and the radial basis weight of the node
function calculateTransformationWeight(vertexCoordinates, nodeCoordinates, radialBasisWeight)
  return exp(-pow(norm(vertexCoordinates - nodeCoordinates), 2) / (2 * pow(radialBasisWeight, 2)))
end

-- function to calculate the huber penalty
function huberPenalty(a, delta) -- the value of delta?
  if lesseq(abs(a), delta) then
    return a * a / 2
  else
    return delta * abs(a) - delta *  delta / 2
  end
end

-- function to calculate the tukey penalty
function tukeyPenalty(x, c) -- the value of c is highly dependent on the quality of data; for alessandro recorded with realsense we recommend c = 4
  if lesseq(abs(x), c) then
    return x * pow(1.0 - (x * x) / (c * c), 2)
  else
    return 0
  end
end

-- energy specifciation

local D,N = Dim("D",0), Dim("N",1)

local nodeCoordinates = Array("nodeCoordinates",opt_float3,{D},0) -- used to calculate transformation weights from the radial basis weights

local rotation = Unknown("rotation",opt_float3,{D},1)
local translation = Unknown("translation",opt_float3,{D},2)
local radialBasisWeights = Unknown("radialBasisWeights",opt_float,{D},3)

local canonicalVertices = Array("canonicalVertices",opt_float3,{N},4)
local canonicalNormals = Array("canonicalNormals",opt_float3,{N},5)

local liveVertices = Array("liveVertices",opt_float3,{N},6)
local liveNormals = Array("liveNormals",opt_float3,{N},7)

local G = Graph("dataGraph", 8,
                    "v", {N}, 9,
                    "n0", {D}, 10,
                    "n1", {D}, 11,
                    "n2", {D}, 12,
                    "n3", {D}, 13,
                    "n4", {D}, 14,
                    "n5", {D}, 15,
                    "n6", {D}, 16,
                    "n7", {D}, 17)

local totalTranslation = 0
local totalRotation = 0

nodes = {0,1,2,3,4,5,6,7}

for _,i in ipairs(nodes) do
    local transformationWeight = calculateTransformationWeight(canonicalVertices(G.v), nodeCoordinates(G["n"..i]), radialBasisWeights(G["n"..i]))
    totalTranslation = totalTranslation + transformationWeight * translation(G["n"..i])
    -- totalRotation = totalRotation + rotation(G["n"..i]) -- FIXME (dig15): use rotations
end

local c = 40
Energy(tukeyPenalty(liveVertices(G.v) - canonicalVertices(G.v) - totalTranslation, c))
-- Energy(tukeyPenalty(dotProduct(canonicalNormals(G.v), (liveVertices(G.v) - canonicalVertices(G.v) - totalTranslation)), c)) -- FIXME (dig15): understand how to use the normals
