package.cpath = package.cpath .. ";/homes/dig15/df/dynfu/include/dynfu/utils/terra/?.so"
require 'luadq'

-- energy specifciation

local D,N = Dim("D",0), Dim("N",1)

local transformation = Unknown("transformation",opt_float3,{D},0)
local rotation = Unknown("rotation",opt_float3,{D},1)
local transformationWeights = Array("transformationWeights",opt_float8,{N},2)

local canonicalVertices = Array("canonicalVertices",opt_float3,{N},3)
local canonicalNormals = Array("canonicalNormals",opt_float3,{N},4)

local liveVertices = Array("liveVertices",opt_float3,{N},5)
local liveNormals = Array("liveNormals",opt_float3,{N},6)

local tukeyBiweights = Array("tukeyBiweights",opt_float,{N},7)

local G = Graph("dataGraph", 8,
                    "v", {N}, 9,
                    "w", {N}, 10, -- transformation weights
                    "n0", {D}, 11,
                    "n1", {D}, 12,
                    "n2", {D}, 13,
                    "n3", {D}, 14,
                    "n4", {D}, 15,
                    "n5", {D}, 16,
                    "n6", {D}, 17,
                    "n7", {D}, 18)

local nodes = { 0, 1, 2, 3, 4, 5, 6, 7 }
-- local totalTransformation = 0
-- local totalRotation = 0

-- energy specifciation

local D,N = Dim("D",0), Dim("N",1)

local transformation = Unknown("transformation",opt_float3,{D},0)
local rotation = Unknown("rotation",opt_float3,{D},1)
local transformationWeights = Array("transformationWeights",opt_float8,{N},2)

local canonicalVertices = Array("canonicalVertices",opt_float3,{N},3)
local canonicalNormals = Array("canonicalNormals",opt_float3,{N},4)

local liveVertices = Array("liveVertices",opt_float3,{N},5)
local liveNormals = Array("liveNormals",opt_float3,{N},6)

local tukeyBiweights = Array("tukeyBiweights",opt_float,{N},7)

local G = Graph("dataGraph", 8,
                    "v", {N}, 9,
                    "w", {N}, 10, -- transformation weights
                    "n0", {D}, 11,
                    "n1", {D}, 12,
                    "n2", {D}, 13,
                    "n3", {D}, 14,
                    "n4", {D}, 15,
                    "n5", {D}, 16,
                    "n6", {D}, 17,
                    "n7", {D}, 18)

local nodes = { 0, 1, 2, 3, 4, 5, 6, 7 }
local totalTransformation = 0
-- local totalRotation = 0

for _,i in ipairs(nodes) do
  local c1, c2, c3 = cos(rotation(G["n"..i])(0) / 2), cos(rotation(G["n"..i])(1) / 2), cos(rotation(G["n"..i])(2) / 2)
  local s1, s2, s3 = sin(rotation(G["n"..i])(0) / 2), sin(rotation(G["n"..i])(1) / 2), sin(rotation(G["n"..i])(2) / 2)

  local angle = 2 * acos(c1 * c2 * c3 - s1 * s2 * s3)
  local axis = { s1 * s2 * c3 + c1 * c2 * s3, s1 * c2 * c3 + c1 * s2 * s3, c1 * s2 * c3 - s1 * c2 * s3 }

  totalTransformation = totalTransformation + transformationWeights(G.w)(i) * transformation(G["n"..i])
end

Energy(sqrt(tukeyBiweights(G.v)) * (liveVertices(G.v) - canonicalVertices(G.v) - totalTransformation))
