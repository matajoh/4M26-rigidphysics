# flake8: noqa
# test file

import numpy as np
from scipy.spatial.transform import Rotation

from rigidphysics.maths import quaternion_compose, quaternion_from_rotvec, quaternion_identity, quaternion_inverse_rotate, quaternion_rotate, quaternion_to_matrix


def test_quaternion_to_matrix():
    rotvecs = np.random.uniform(-1, 1, (100, 3))
    for rotvec in rotvecs:
        rotation = Rotation.from_rotvec(rotvec)
        desired = rotation.as_matrix()
        actual = np.empty((3, 3), np.float64)
        quaternion_to_matrix(rotation.as_quat(), actual)
        assert np.allclose(actual, desired)


def test_quaternion_rotate():
    rotvecs = np.random.uniform(-1, 1, (10, 3))
    for rotvec in rotvecs:
        rotation = Rotation.from_rotvec(rotvec)
        quaternion = rotation.as_quat()
        for _ in range(10):
            point = np.random.uniform(-1, 1, 3)
            desired = rotation.apply(point)
            actual = np.empty(3, np.float64)
            quaternion_rotate(quaternion, point, actual)
            assert np.allclose(actual, desired)

        points = np.random.uniform(-1, 1, (10, 3))
        desired = rotation.apply(points)
        actual = np.empty((10, 3), np.float64)
        quaternion_rotate(quaternion, points, actual)
        assert np.allclose(actual, desired)


def test_quaternion_inverse_rotate():
    rotvecs = np.random.uniform(-1, 1, (10, 3))
    for rotvec in rotvecs:
        rotation = Rotation.from_rotvec(rotvec)
        quaternion = rotation.as_quat()
        print(quaternion, rotation.inv().as_quat())
        for _ in range(10):
            point = np.random.uniform(-1, 1, 3)
            desired = rotation.inv().apply(point)
            actual = np.empty(3, np.float64)
            quaternion_inverse_rotate(quaternion, point, actual)
            assert np.allclose(actual, desired)

        points = np.random.uniform(-1, 1, (10, 3))
        desired = rotation.inv().apply(points)
        actual = np.empty((10, 3), np.float64)
        quaternion_inverse_rotate(quaternion, points, actual)
        assert np.allclose(actual, desired)

def test_quaternion_compose():
    rotvecs = np.random.uniform(-1, 1, (100, 2, 3))
    for rv0, rv1 in rotvecs:
        r0 = Rotation.from_rotvec(rv0)
        q0 = r0.as_quat()
        r1 = Rotation.from_rotvec(rv1)
        q1 = r1.as_quat()
        desired = (r0 * r1).as_quat()
        actual = np.empty(4, np.float64)
        quaternion_compose(q0, q1, actual)
        assert np.allclose(actual, desired)


def test_quaternion_from_rotvec():
    rotvecs = np.random.uniform(-1, 1, (100, 3))
    for rotvec in rotvecs:
        rotation = Rotation.from_rotvec(rotvec)
        quaternion = rotation.as_quat()
        actual = np.empty(4, np.float64)
        quaternion_from_rotvec(rotvec, actual)
        assert np.allclose(actual, quaternion)


def test_quaternion_identity():
    expected = Rotation.identity().as_quat()
    actual = quaternion_identity()
    assert np.allclose(actual, expected)


if __name__ == "__main__":
    test_quaternion_to_matrix()
