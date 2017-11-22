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

/* */
TEST_F(DualQuaternionTest, TestTotal) {
    DualQuaternion<float> dq1(0, 1, 1, 0.0, 0.0, 0.0);
    DualQuaternion<float> dq2(0, 1, 1, 0.0, 0.0, 0.0);

    DualQuaternion<float> dqSum = dq1 + dq2;

    DualQuaternion<float> dqRes(2.0, 2.0, 2.0, 0.0, 0.0, 0.0);

    /* */
    // ASSERT_FLOAT_EQ(dqSum.getReal().R_component_1(), dqRes.getReal().R_component_1());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
