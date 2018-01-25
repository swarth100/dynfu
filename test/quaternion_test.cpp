#include <gtest/gtest.h>
#include <dynfu/utils/dual_quaternion.hpp>

#include <cmath>

/* The fixture for testing class Quaternion. */
class DualQuaternionTest : public ::testing::Test {
protected:
    /* You can remove any or all of the following functions if its body
     * is empty. */

    /* You can do set-up work for each test here. */
    DualQuaternionTest() = default;

    /* You can do clean-up work that doesn't throw exceptions here. */
    ~DualQuaternionTest() override = default;

    /* If the constructor and destructor are not enough for setting up
     * and cleaning up each test, you can define the following methods: */

    /* Code here will be called immediately after the constructor (right
     * before each test). */
    void SetUp() override {}

    /* Code here will be called immediately after each test (right
     * before the destructor). */
    void TearDown() override {}

    /* Objects declared here can be used by all tests in the test case for Foo. */
    float RAD180 = M_PI;
    float RAD150 = RAD120 + RAD30;
    float RAD120 = RAD30 + RAD90;

    float RAD90 = M_PI / 2;
    float RAD60 = M_PI / 3;
    float RAD45 = M_PI / 4;
    float RAD30 = M_PI / 6;
    float RAD15 = M_PI / 12;

    float MAXERROR = 0.0001;

    DualQuaternion<float> dq90    = DualQuaternion<float>(RAD90, RAD90, RAD90, 0.f, 0.f, 0.f);
    DualQuaternion<float> dq60    = DualQuaternion<float>(RAD60, RAD60, RAD60, 0.f, 0.f, 0.f);
    DualQuaternion<float> dq45    = DualQuaternion<float>(RAD45, RAD45, RAD45, 0.f, 0.f, 0.f);
    DualQuaternion<float> dq30Rot = DualQuaternion<float>(RAD30, RAD30, RAD30, 0.f, 0.f, 0.f);
    DualQuaternion<float> dq0     = DualQuaternion<float>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);

    DualQuaternion<float> dq30 = DualQuaternion<float>(0.0f, RAD30, 0.0f, 0.0f, 0.0f, 100.0f);
};

/* The following calculator has been used for the tests:
 * http://www.andre-gaschler.com/rotationconverter/
 */

/* This test checks that the Real part of Dual Quaternions is computed correctly
 * We check the results against an online calculator */
TEST_F(DualQuaternionTest, TestReal) {
    /* Real should be:
     *     (0.8446, 0.1913, 0.4619, 0.1913)
     */
    ASSERT_NEAR(dq45.getReal().R_component_1(), 0.8446231020115715, MAXERROR);
    ASSERT_NEAR(dq45.getReal().R_component_2(), 0.19134170284356308, MAXERROR);
    ASSERT_NEAR(dq45.getReal().R_component_3(), 0.4619399539487806, MAXERROR);
    ASSERT_NEAR(dq45.getReal().R_component_4(), 0.19134170284356303, MAXERROR);
}

/* This test checks that the Dual part of Dual Quaternions is computed correctly
 * The following test can be found at:
 * https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/other/dualQuaternion/example/index.htm */
TEST_F(DualQuaternionTest, TestDual) {
    /* First the Quaternion should evaluate the Real Part.
     * Secondly, given the (normal of the) Real Part, the Quaternion multiplies
     * it by the Translation value to hold the Dual Part. */

    /* Real should be:
     *     (0.9659, 0, 0.2588, 0)
     */
    ASSERT_NEAR(dq30.getReal().R_component_1(), 0.9659, MAXERROR);
    ASSERT_NEAR(dq30.getReal().R_component_2(), 0.0f, MAXERROR);
    ASSERT_NEAR(dq30.getReal().R_component_3(), 0.2588, MAXERROR);
    ASSERT_NEAR(dq30.getReal().R_component_4(), 0.0f, MAXERROR);

    /* Dual should be:
     *     [0.5 * (0, 0, 0, 100)](0.9659, 0, 0.2588, 0) =
     *     (0, -12.94, 0, 48.295)
     */
    ASSERT_NEAR(dq30.getDual().R_component_1(), 0.0f, MAXERROR);
    ASSERT_NEAR(dq30.getDual().R_component_2(), -12.9409, MAXERROR);
    ASSERT_NEAR(dq30.getDual().R_component_3(), 0.0f, MAXERROR);
    ASSERT_NEAR(dq30.getDual().R_component_4(), 48.2962, MAXERROR);
}

TEST_F(DualQuaternionTest, TestFromRodrigues) {
    auto dq30New = DualQuaternion<float>(0.f, RAD30, 0.f, 0.f, 0.f, 0.f);

    cv::Vec3f translation(0.f, 0.f, 0.f);

    cv::Vec3f rodrigues30(0.f, 0.267949192431123, 0.f);
    cv::Vec3f rodrigues45(0.226540919660986, 0.546918160678027, 0.226540919660986);
    cv::Vec3f rodrigues90(0.f, 1.f, 0.f);

    auto dq30FromRodrigues = DualQuaternion<float>(rodrigues30, translation);
    auto dq45FromRodrigues = DualQuaternion<float>(rodrigues45, translation);
    auto dq90FromRodrigues = DualQuaternion<float>(rodrigues90, translation);

    ASSERT_NEAR(dq30FromRodrigues.getReal().R_component_1(), dq30New.getReal().R_component_1(), MAXERROR);
    ASSERT_NEAR(dq30FromRodrigues.getReal().R_component_2(), dq30New.getReal().R_component_2(), MAXERROR);
    ASSERT_NEAR(dq30FromRodrigues.getReal().R_component_3(), dq30New.getReal().R_component_3(), MAXERROR);
    ASSERT_NEAR(dq30FromRodrigues.getReal().R_component_4(), dq30New.getReal().R_component_4(), MAXERROR);

    ASSERT_NEAR(dq45FromRodrigues.getReal().R_component_1(), dq45.getReal().R_component_1(), MAXERROR);
    ASSERT_NEAR(dq45FromRodrigues.getReal().R_component_2(), dq45.getReal().R_component_2(), MAXERROR);
    ASSERT_NEAR(dq45FromRodrigues.getReal().R_component_3(), dq45.getReal().R_component_3(), MAXERROR);
    ASSERT_NEAR(dq45FromRodrigues.getReal().R_component_4(), dq45.getReal().R_component_4(), MAXERROR);

    ASSERT_NEAR(dq90FromRodrigues.getReal().R_component_1(), dq90.getReal().R_component_1(), MAXERROR);
    ASSERT_NEAR(dq90FromRodrigues.getReal().R_component_2(), dq90.getReal().R_component_2(), MAXERROR);
    ASSERT_NEAR(dq90FromRodrigues.getReal().R_component_3(), dq90.getReal().R_component_3(), MAXERROR);
    ASSERT_NEAR(dq90FromRodrigues.getReal().R_component_4(), dq90.getReal().R_component_4(), MAXERROR);
}

/* Test that the sum + between two DualQuaternion is computed correctly */
TEST_F(DualQuaternionTest, TestSum) {
    DualQuaternion<float> dqSum = dq45 + dq30;

    /* Sum should be:
     *     Real:
     *     (1.8105, 0.1913, 0.7208, 0.1913)
     *     Dual:
     *     (0, -12.94, 0, 48.295)
     */
    ASSERT_NEAR(dqSum.getReal().R_component_1(), 1.8105, MAXERROR);
    ASSERT_NEAR(dqSum.getReal().R_component_2(), 0.1913, MAXERROR);
    ASSERT_NEAR(dqSum.getReal().R_component_3(), 0.7208, MAXERROR);
    ASSERT_NEAR(dqSum.getReal().R_component_4(), 0.1913, MAXERROR);

    ASSERT_NEAR(dqSum.getDual().R_component_1(), 0.0f, MAXERROR);
    ASSERT_NEAR(dqSum.getDual().R_component_2(), -12.9410, MAXERROR);
    ASSERT_NEAR(dqSum.getDual().R_component_3(), 0.0f, MAXERROR);
    ASSERT_NEAR(dqSum.getDual().R_component_4(), 48.2963, MAXERROR);
}

/* test that rotations compose correctly */
TEST_F(DualQuaternionTest, TestComposeRotations) {
    pcl::PointXYZ vertex(0, 0, 1);

    pcl::PointXYZ vertexTransformed1Rot  = dq90.transformVertex(vertex);
    pcl::PointXYZ vertexTransformed2Rots = dq90.transformVertex(vertexTransformed1Rot);

    DualQuaternion<float> dqComposition        = dq90 * dq90;
    pcl::PointXYZ vertexTransformedComposition = dqComposition.transformVertex(vertex);

    ASSERT_NEAR(vertexTransformed2Rots.x, vertexTransformedComposition.x, MAXERROR);
    ASSERT_NEAR(vertexTransformed2Rots.y, vertexTransformedComposition.y, MAXERROR);
    ASSERT_NEAR(vertexTransformed2Rots.z, vertexTransformedComposition.z, MAXERROR);
}

/* Test that the sum and assign += between two DualQuaternion is computed correctly */
TEST_F(DualQuaternionTest, TestSumAssign) {
    DualQuaternion<float> dqSum = DualQuaternion<float>(RAD30, RAD45, RAD30, 30, 20, 10);

    dqSum += dq30;

    /* Sum should be:
     *     Real:
     *     (1.8536, 0.1353, 0.6778, 0.1353)
     *     Dual:
     *     (-6.8953, -0.3683, 7.5233, 57.6655)
     */
    ASSERT_NEAR(dqSum.getReal().R_component_1(), 1.8536, MAXERROR);
    ASSERT_NEAR(dqSum.getReal().R_component_2(), 0.1353, MAXERROR);
    ASSERT_NEAR(dqSum.getReal().R_component_3(), 0.6778, MAXERROR);
    ASSERT_NEAR(dqSum.getReal().R_component_4(), 0.1353, MAXERROR);

    ASSERT_NEAR(dqSum.getDual().R_component_1(), -6.8953, MAXERROR);
    ASSERT_NEAR(dqSum.getDual().R_component_2(), -0.3683, MAXERROR);
    ASSERT_NEAR(dqSum.getDual().R_component_3(), 7.5233, MAXERROR);
    ASSERT_NEAR(dqSum.getDual().R_component_4(), 57.6655, MAXERROR);
}

/* Test that the diff - between two DualQuaternion is computed correctly */
TEST_F(DualQuaternionTest, TestDiff) {
    DualQuaternion<float> dqDiff = dq45 - dq30;

    /* Diff should be:
     *     Real:
     *     (-0.1213, 0.1913, 0.2031, 0.1913)
     *     Dual:
     *     (0, 12.9410, 0, 48.2962)
     */
    ASSERT_NEAR(dqDiff.getReal().R_component_1(), -0.1213, MAXERROR);
    ASSERT_NEAR(dqDiff.getReal().R_component_2(), 0.1913, MAXERROR);
    ASSERT_NEAR(dqDiff.getReal().R_component_3(), 0.2031, MAXERROR);
    ASSERT_NEAR(dqDiff.getReal().R_component_4(), 0.1913, MAXERROR);

    ASSERT_NEAR(dqDiff.getDual().R_component_1(), 0.0f, MAXERROR);
    ASSERT_NEAR(dqDiff.getDual().R_component_2(), 12.9410, MAXERROR);
    ASSERT_NEAR(dqDiff.getDual().R_component_3(), 0.0f, MAXERROR);
    ASSERT_NEAR(dqDiff.getDual().R_component_4(), -48.2963, MAXERROR);
}

/* Test that the diff and assign -= between two DualQuaternion is computed correctly */
TEST_F(DualQuaternionTest, TestDiffAssign) {
    DualQuaternion<float> dqDiff = DualQuaternion<float>(RAD30, RAD45, RAD30, 30, 20, 10);

    dqDiff -= dq30;

    /* Diff should be:
     *     Real:
     *     (-0.0783, 0.1353, 0.1601, 0.1353)
     *     Dual:
     *     (-6.8953, 25.5137, 7.5233, -38.9271)
     */
    ASSERT_NEAR(dqDiff.getReal().R_component_1(), -0.0783, MAXERROR);
    ASSERT_NEAR(dqDiff.getReal().R_component_2(), 0.1353, MAXERROR);
    ASSERT_NEAR(dqDiff.getReal().R_component_3(), 0.1601, MAXERROR);
    ASSERT_NEAR(dqDiff.getReal().R_component_4(), 0.1353, MAXERROR);

    ASSERT_NEAR(dqDiff.getDual().R_component_1(), -6.8953, MAXERROR);
    ASSERT_NEAR(dqDiff.getDual().R_component_2(), 25.5137, MAXERROR);
    ASSERT_NEAR(dqDiff.getDual().R_component_3(), 7.5233, MAXERROR);
    ASSERT_NEAR(dqDiff.getDual().R_component_4(), -38.9271, MAXERROR);
}

/* Test that the scaling between a DualQuaternion and a scalar is computed correctly */
TEST_F(DualQuaternionTest, TestScale) {
    float scale                   = 0.30;
    DualQuaternion<float> dqScale = dq30 * scale;

    /* Scaled dq should be:
     *     Dual:
     *     (0, -3.8823, 0, 14.4889)
     */
    ASSERT_NEAR(dqScale.getReal().R_component_1(), dqScale.getReal().R_component_1(), MAXERROR);
    ASSERT_NEAR(dqScale.getReal().R_component_2(), dqScale.getReal().R_component_2(), MAXERROR);
    ASSERT_NEAR(dqScale.getReal().R_component_3(), dqScale.getReal().R_component_3(), MAXERROR);
    ASSERT_NEAR(dqScale.getReal().R_component_4(), dqScale.getReal().R_component_4(), MAXERROR);

    ASSERT_NEAR(dqScale.getDual().R_component_1(), 0.0f, MAXERROR);
    ASSERT_NEAR(dqScale.getDual().R_component_2(), -3.8823, MAXERROR);
    ASSERT_NEAR(dqScale.getDual().R_component_3(), 0.0f, MAXERROR);
    ASSERT_NEAR(dqScale.getDual().R_component_4(), 14.4889, MAXERROR);
}

/* Test that the scale and assign *= between a DualQuaternion and a scalar is computed correctly */
TEST_F(DualQuaternionTest, TestScaleAssign) {
    float scale                   = 0.30;
    DualQuaternion<float> dqScale = DualQuaternion<float>(RAD30, RAD45, RAD30, 30, 20, 10);

    dqScale *= scale;

    /* Scaled dq should be:
     *     Dual:
     *     (-2.0686, 3.7718, 2.2570, 2.8108)
     */
    ASSERT_NEAR(dqScale.getReal().R_component_1(), dqScale.getReal().R_component_1(), MAXERROR);
    ASSERT_NEAR(dqScale.getReal().R_component_2(), dqScale.getReal().R_component_2(), MAXERROR);
    ASSERT_NEAR(dqScale.getReal().R_component_3(), dqScale.getReal().R_component_3(), MAXERROR);
    ASSERT_NEAR(dqScale.getReal().R_component_4(), dqScale.getReal().R_component_4(), MAXERROR);

    ASSERT_NEAR(dqScale.getDual().R_component_1(), -2.0686, MAXERROR);
    ASSERT_NEAR(dqScale.getDual().R_component_2(), 3.7718, MAXERROR);
    ASSERT_NEAR(dqScale.getDual().R_component_3(), 2.2570, MAXERROR);
    ASSERT_NEAR(dqScale.getDual().R_component_4(), 2.8108, MAXERROR);
}

/* Test that the multiplication between two DualQuaternions is computed correctly */
TEST_F(DualQuaternionTest, TestMul) {
    DualQuaternion<float> dqMul = dq30 * dq45;

    /* Multiplied dq should be:
     *     Real:
     *     (0.6963, 0.2343, 0.6648, 0.1353)
     *     Dual:
     *     (-6.7650, -33.2402, 11.7172, 34.8142)
     */
    ASSERT_NEAR(dqMul.getReal().R_component_1(), 0.6963, MAXERROR);
    ASSERT_NEAR(dqMul.getReal().R_component_2(), 0.2343, MAXERROR);
    ASSERT_NEAR(dqMul.getReal().R_component_3(), 0.6648, MAXERROR);
    ASSERT_NEAR(dqMul.getReal().R_component_4(), 0.1353, MAXERROR);

    ASSERT_NEAR(dqMul.getDual().R_component_1(), -6.7650, MAXERROR);
    ASSERT_NEAR(dqMul.getDual().R_component_2(), -33.2402, MAXERROR);
    ASSERT_NEAR(dqMul.getDual().R_component_3(), 11.7172, MAXERROR);
    ASSERT_NEAR(dqMul.getDual().R_component_4(), 34.8142, MAXERROR);
}

/* Test that the multiplication and assign *= between two DualQuaternions is computed correctly */
TEST_F(DualQuaternionTest, TestMulAssign) {
    DualQuaternion<float> dqMul = DualQuaternion<float>(RAD30, RAD45, RAD30, 30, 20, 10);
    dqMul *= dq30;

    /* Multiplied dq should be:
     *     Real:
     *     (0.7490, 0.0957, 0.6344, 0.1657)
     *     Dual:
     *     (-13.3911, 18.4657, -2.8031, 60.5945)
     */
    ASSERT_NEAR(dqMul.getReal().R_component_1(), 0.7490, MAXERROR);
    ASSERT_NEAR(dqMul.getReal().R_component_2(), 0.0957, MAXERROR);
    ASSERT_NEAR(dqMul.getReal().R_component_3(), 0.6344, MAXERROR);
    ASSERT_NEAR(dqMul.getReal().R_component_4(), 0.1657, MAXERROR);

    ASSERT_NEAR(dqMul.getDual().R_component_1(), -13.3911, MAXERROR);
    ASSERT_NEAR(dqMul.getDual().R_component_2(), 18.4657, MAXERROR);
    ASSERT_NEAR(dqMul.getDual().R_component_3(), -2.8031, MAXERROR);
    ASSERT_NEAR(dqMul.getDual().R_component_4(), 60.5945, MAXERROR);
}

/* Test that the normalization of a DualQuaternion is computed correctly */
TEST_F(DualQuaternionTest, TestNormalize) {
    DualQuaternion<float> dqSum           = dq45 + dq30;
    DualQuaternion<float> dqSumNormalized = dqSum.normalize();

    /* Normalized dq should be:
     *     Real:
     *     (0.9203, 0.0973, 0.3663, 0.0973)
     *     Dual:
     *     (0, -12.9410, 0, 48.2963)
     */
    ASSERT_NEAR(dqSumNormalized.getReal().R_component_1(), 0.9203, MAXERROR);
    ASSERT_NEAR(dqSumNormalized.getReal().R_component_2(), 0.0973, MAXERROR);
    ASSERT_NEAR(dqSumNormalized.getReal().R_component_3(), 0.3663, MAXERROR);
    ASSERT_NEAR(dqSumNormalized.getReal().R_component_4(), 0.0973, MAXERROR);

    ASSERT_NEAR(dqSumNormalized.getDual().R_component_1(), 0.0f, MAXERROR);
    ASSERT_NEAR(dqSumNormalized.getDual().R_component_2(), -12.9410, MAXERROR);
    ASSERT_NEAR(dqSumNormalized.getDual().R_component_3(), 0.0f, MAXERROR);
    ASSERT_NEAR(dqSumNormalized.getDual().R_component_4(), 48.2963, MAXERROR);
}

/* tests that a 0 dual quaternion doesn't translate or rotate a vector */
TEST_F(DualQuaternionTest, TestDoNotTransform) {
    pcl::PointXYZ vertex(0, 0, 1);
    pcl::PointXYZ vertexTransformed = dq0.transformVertex(vertex);

    ASSERT_NEAR(vertexTransformed.x, 0, MAXERROR);
    ASSERT_NEAR(vertexTransformed.y, 0, MAXERROR);
    ASSERT_NEAR(vertexTransformed.z, 1, MAXERROR);
}

/* tests that dual quaternion rotates a vector correctly */
TEST_F(DualQuaternionTest, TestRotate) {
    pcl::PointXYZ vertex(0, 0, 1);
    pcl::PointXYZ vertexTransformed = dq90.transformVertex(vertex);

    ASSERT_NEAR(vertexTransformed.x, 1, MAXERROR);
    ASSERT_NEAR(vertexTransformed.y, 0, MAXERROR);
    ASSERT_NEAR(vertexTransformed.z, 0, MAXERROR);
}

/* tests that dual quaternion translates a vector correctly */
TEST_F(DualQuaternionTest, TestTranslate) {
    DualQuaternion<float> dq = DualQuaternion<float>(0.f, 0.f, 0.f, 1.f, 0.f, 0.f);

    pcl::PointXYZ vertex(0, 0, 1);
    pcl::PointXYZ vertexTransformed = dq.transformVertex(vertex);

    ASSERT_NEAR(vertexTransformed.x, 1, MAXERROR);
    ASSERT_NEAR(vertexTransformed.y, 0, MAXERROR);
    ASSERT_NEAR(vertexTransformed.z, 1, MAXERROR);
}

/* tests that dual quaternion simultaneously rotates and translates a vector correctly */
TEST_F(DualQuaternionTest, TestTranslateAndRotate) {
    DualQuaternion<float> dq = DualQuaternion<float>(RAD90, RAD90, RAD90, 1.f, 0.f, 0.f);

    pcl::PointXYZ vertex(0, 0, 1);
    pcl::PointXYZ vertexTransformed = dq.transformVertex(vertex);

    ASSERT_NEAR(vertexTransformed.x, 2, MAXERROR);
    ASSERT_NEAR(vertexTransformed.y, 0, MAXERROR);
    ASSERT_NEAR(vertexTransformed.z, 0, MAXERROR);
}

/* tests that roll is returned correctly */
TEST_F(DualQuaternionTest, RollTest) {
    auto dq30Rot = DualQuaternion<float>(0.f, RAD30, 0.f, 0.f, 0.f, 0.f);

    auto roll30 = dq30Rot.getRoll();
    auto roll45 = dq45.getRoll();
    auto roll90 = dq90.getRoll();

    ASSERT_NEAR(roll30, 0, MAXERROR);
    ASSERT_NEAR(roll45, RAD45, MAXERROR);
    ASSERT_NEAR(roll90, RAD90, MAXERROR);
}

/* tests that pitch is returned correctly */
TEST_F(DualQuaternionTest, PitchTest) {
    auto pitch30 = dq30.getPitch();
    auto pitch45 = dq45.getPitch();
    auto pitch90 = dq90.getPitch();

    ASSERT_NEAR(pitch30, RAD30, MAXERROR);
    ASSERT_NEAR(pitch45, RAD45, MAXERROR);
    ASSERT_NEAR(pitch90, RAD90, MAXERROR);
}

/* tests that yaw is returned correctly */
TEST_F(DualQuaternionTest, YawTest) {
    auto dq30Rot = DualQuaternion<float>(0.f, RAD30, 0.f, 0.f, 0.f, 0.f);

    auto yaw30 = dq30Rot.getYaw();
    auto yaw45 = dq45.getYaw();
    auto yaw90 = dq90.getYaw();

    ASSERT_NEAR(yaw30, 0, MAXERROR);
    ASSERT_NEAR(yaw45, RAD45, MAXERROR);
    ASSERT_NEAR(yaw90, RAD90, MAXERROR);
}

/* test dual quaternion to Euler angles conversion */
TEST_F(DualQuaternionTest, ConvertToEulerAnglesTest) {
    auto dq30Rot = DualQuaternion<float>(0.f, RAD30, 0.f, 0.f, 0.f, 0.f);

    auto eulerAngles30 = dq30Rot.getEulerAngles();
    auto eulerAngles45 = dq45.getEulerAngles();
    auto eulerAngles90 = dq90.getEulerAngles();

    /* roll */
    ASSERT_NEAR(eulerAngles30[0], 0, MAXERROR);
    ASSERT_NEAR(eulerAngles45[0], RAD45, MAXERROR);
    ASSERT_NEAR(eulerAngles90[0], RAD90, MAXERROR);

    /* pitch */
    ASSERT_NEAR(eulerAngles30[1], RAD30, MAXERROR);
    ASSERT_NEAR(eulerAngles45[1], RAD45, MAXERROR);
    ASSERT_NEAR(eulerAngles90[1], RAD90, MAXERROR);

    /* yaw */
    ASSERT_NEAR(eulerAngles30[2], 0, MAXERROR);
    ASSERT_NEAR(eulerAngles45[2], RAD45, MAXERROR);
    ASSERT_NEAR(eulerAngles90[2], RAD90, MAXERROR);
}

/* test dual quaternion to Rodrigues conversion */
TEST_F(DualQuaternionTest, ConvertToRodriguesTest) {
    auto dq30Rot = DualQuaternion<float>(0.f, RAD30, 0.f, 0.f, 0.f, 0.f);

    auto rodrigues30 = dq30Rot.getRodrigues();
    auto rodrigues45 = dq45.getRodrigues();
    auto rodrigues90 = dq90.getRodrigues();

    ASSERT_NEAR(rodrigues30[0], 0, MAXERROR);
    ASSERT_NEAR(rodrigues30[1], 0.267949192431123, MAXERROR);
    ASSERT_NEAR(rodrigues30[2], 0, MAXERROR);

    ASSERT_NEAR(rodrigues45[0], 0.226540919660986, MAXERROR);
    ASSERT_NEAR(rodrigues45[1], 0.546918160678027, MAXERROR);
    ASSERT_NEAR(rodrigues45[2], 0.226540919660986, MAXERROR);

    ASSERT_NEAR(rodrigues90[0], 0, MAXERROR);
    ASSERT_NEAR(rodrigues90[1], 1, MAXERROR);
    ASSERT_NEAR(rodrigues90[2], 0, MAXERROR);
}

TEST_F(DualQuaternionTest, TestToString) {
    std::ostringstream dqString;
    dqString << dq30;
    ASSERT_EQ("real: (0.965926,0,0.258819,0)\ndual: (0,-12.941,0,48.2963)\n", dqString.str());
}
