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
    float RAD90  = M_PI / 2;
    float RAD45  = M_PI / 4;
    float RAD30  = M_PI / 6;

    float MAXERROR = 0.0001;

    DualQuaternion<float> dq45 = DualQuaternion<float>(RAD45, RAD45, RAD45, 0.0f, 0.0f, 0.0f);

    DualQuaternion<float> dq30 = DualQuaternion<float>(0.0f, RAD30, 0.0f, 0.0f, 0.0f, 100.0f);
};

/* The following calculator has been used for the tests:
 * http://www.andre-gaschler.com/rotationconverter/
 */

/* This test checks that the Real part of Dual Quaternions is computed correctly
 * We check the results against an online calculator */
TEST_F(DualQuaternionTest, TestReal) {
    /* */
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

    float scale = 0.30;
    DualQuaternion<float> dqScale = dq30 * scale;

    /* Scaled dq should be:
     *     Real:
     *     (0.2898, 0, 0.0776, 0)
     *     Dual:
     *     (0, -3.8823, 0, 14.4889)
     */
    ASSERT_NEAR(dqScale.getReal().R_component_1(), 0.2898, MAXERROR);
    ASSERT_NEAR(dqScale.getReal().R_component_2(), 0.0f, MAXERROR);
    ASSERT_NEAR(dqScale.getReal().R_component_3(), 0.0776, MAXERROR);
    ASSERT_NEAR(dqScale.getReal().R_component_4(), 0.0f, MAXERROR);

    ASSERT_NEAR(dqScale.getDual().R_component_1(), 0.0f, MAXERROR);
    ASSERT_NEAR(dqScale.getDual().R_component_2(), -3.8823, MAXERROR);
    ASSERT_NEAR(dqScale.getDual().R_component_3(), 0.0f, MAXERROR);
    ASSERT_NEAR(dqScale.getDual().R_component_4(), 14.4889, MAXERROR);
}

/* Test that the scale and assign *= between a DualQuaternion and a scalar is computed correctly */
TEST_F(DualQuaternionTest, TestScaleAssign) {

    float scale = 0.30;
    DualQuaternion<float> dqScale = DualQuaternion<float>(RAD30, RAD45, RAD30, 30, 20, 10);

    dqScale *= scale;

    /* Scaled dq should be:
     *     Real:
     *     (0.2663, 0.0406, 0.1257, 0.0406)
     *     Dual:
     *     (-2.0686, 3.7718, 2.2570, 2.8108)
     */
    ASSERT_NEAR(dqScale.getReal().R_component_1(), 0.2663, MAXERROR);
    ASSERT_NEAR(dqScale.getReal().R_component_2(), 0.0406, MAXERROR);
    ASSERT_NEAR(dqScale.getReal().R_component_3(), 0.1257, MAXERROR);
    ASSERT_NEAR(dqScale.getReal().R_component_4(), 0.0406, MAXERROR);

    ASSERT_NEAR(dqScale.getDual().R_component_1(), -2.0686, MAXERROR);
    ASSERT_NEAR(dqScale.getDual().R_component_2(), 3.7718, MAXERROR);
    ASSERT_NEAR(dqScale.getDual().R_component_3(), 2.2570, MAXERROR);
    ASSERT_NEAR(dqScale.getDual().R_component_4(), 2.8108, MAXERROR);
}



