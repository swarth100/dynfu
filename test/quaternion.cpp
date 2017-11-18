#include <gtest/gtest.h>
#include <dynfu/utils/dual_quaternion.hpp>

#include <math.h>

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

    float MAXERROR = 0.000001;

    DualQuaternion<float> dq45 = DualQuaternion<float>(RAD45, RAD45, RAD45, 0.0, 0.0, 0.0);
};

/* The following calculator has been used for the tests:
 * http://www.andre-gaschler.com/rotationconverter/
 */

/* */
TEST_F(DualQuaternionTest, TestRotationReal) {
    /* */
    ASSERT_NEAR(dq45.getReal().R_component_1(), 0.8446231020115715, MAXERROR);
}

/* */
TEST_F(DualQuaternionTest, TestRotationI) {
    /* */
    ASSERT_NEAR(dq45.getReal().R_component_2(), 0.19134170284356308, MAXERROR);
}

/* */
TEST_F(DualQuaternionTest, TestRotationJ) {
    /* */
    ASSERT_NEAR(dq45.getReal().R_component_3(), 0.4619399539487806, MAXERROR);
}

/* */
TEST_F(DualQuaternionTest, TestRotationK) {
    /* */
    ASSERT_NEAR(dq45.getReal().R_component_4(), 0.19134170284356303, MAXERROR);
}

/* */
TEST_F(DualQuaternionTest, TestReal) {
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
