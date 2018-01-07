-- function definitions

-- -- multiply two 3x3 matrices
-- function MatrixMul(m1, m2)
--   local mtx = {}
--
-- 	mtx[0] = m1(0) * m2(0) + m1(1) * m2(3) + m1(2) * m2(6)
--   mtx[1] = m1(0) * m2(1) + m1(1) * m2(4) + m1(2) * m2(7)
--   mtx[2] = m1(0) * m2(2) + m1(1) * m2(5) + m1(2) * m2(8)
--
--   mtx[3] = m1(3) * m2(0) + m1(4) * m2(3) + m1(5) * m2(6)
--   mtx[4] = m1(3) * m2(1) + m1(4) * m2(4) + m1(5) * m2(7)
--   mtx[5] = m1(3) * m2(2) + m1(4) * m2(5) + m1(5) * m2(8)
--
--   mtx[6] = m1(6) * m2(0) + m1(7) * m2(3) + m1(8) * m2(6)
--   mtx[7] = m1(6) * m2(1) + m1(7) * m2(4) + m1(8) * m2(7)
--   mtx[8] = m1(6) * m2(2) + m1(7) * m2(5) + m1(8) * m2(8)
--
--   return ad.Vector(mtx[0], mtx[1], mtx[2],
--                    mtx[3], mtx[4], mtx[5],
--                    mtx[6], mtx[7], mtx[8])
-- end
--
-- -- convert Euler angles to a rotation matrix
-- function eulerToMatrix(r)
--     local alpha, beta, gamma = r(0), r(1), r(2)
--     local CosAlpha, CosBeta, CosGamma, SinAlpha, SinBeta, SinGamma = ad.cos(alpha), ad.cos(beta), ad.cos(gamma), ad.sin(alpha), ad.sin(beta), ad.sin(gamma)
--
--     local matrix = ad.Vector(
--           CosGamma*CosBeta,
--           -SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha,
--           SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha,
--
--           SinGamma*CosBeta,
--           CosGamma*CosAlpha + SinGamma*SinBeta*SinAlpha,
--           -CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha,
--
--           -SinBeta,
--           CosBeta*SinAlpha,
--           CosBeta*CosAlpha)
--
--     return matrix
-- end

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
    totalTransformation = totalTransformation + transformationWeights(G.w)(i) * transformation(G["n"..i])
end

Energy(sqrt(transformationWeights(G.v)) * (liveVertices(G.v) - canonicalVertices(G.v) - totalTransformation))
