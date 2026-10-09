use bevy::{
    math::{vec3, vec3a, Mat3A, Vec3A},
    prelude::{Mat3, Vec3},
};

pub trait MatrixExtras
where
    Self: Sized,
{
    fn decompose_scale(&self) -> Self;
    fn decompose_scale_rotation(&self) -> (Self, Self);
    fn orthonormalize(&self) -> Self;
}

pub trait VecExtras
where
    Self: Sized,
{
    fn from_scale_mat(matrix: Mat3A) -> Self;
}

impl MatrixExtras for Mat3A {
    #[inline]
    fn decompose_scale(&self) -> Mat3A {
        let det = self.determinant();
        let scale = Vec3A::new(
            self.x_axis.length() * f32::signum(det),
            self.y_axis.length(),
            self.z_axis.length(),
        );

        Mat3A::from_cols(
            vec3a(scale.x, 0., 0.),
            vec3a(0., scale.y, 0.),
            vec3a(0., 0., scale.z),
        )
    }

    #[inline]
    fn decompose_scale_rotation(&self) -> (Mat3A, Mat3A) {
        let scale_mat = self.decompose_scale();
        let scale = Vec3A::from_scale_mat(scale_mat);
        let inv_scale = scale.recip();

        let rotation = Mat3A::from_cols(
            self.x_axis * inv_scale.x,
            self.y_axis * inv_scale.y,
            self.z_axis * inv_scale.z,
        );

        (scale_mat, rotation)
    }

    #[inline]
    fn orthonormalize(&self) -> Self {
        let x_axis = self.x_axis.normalize();

        let mut y_axis = self.y_axis;
        y_axis -= y_axis.dot(x_axis) / x_axis.dot(x_axis) * x_axis;
        y_axis = y_axis.normalize();

        let z_axis = x_axis.cross(y_axis);

        Self {
            x_axis,
            y_axis,
            z_axis,
        }
    }
}

impl MatrixExtras for Mat3 {
    #[inline]
    fn decompose_scale(&self) -> Self {
        Mat3A::from(*self).decompose_scale().into()
    }

    #[inline]
    fn decompose_scale_rotation(&self) -> (Mat3, Mat3) {
        let (scale, rotation) = Mat3A::from(*self).decompose_scale_rotation();
        (scale.into(), rotation.into())
    }

    #[inline]
    fn orthonormalize(&self) -> Self {
        Mat3A::from(*self).orthonormalize().into()
    }
}

impl VecExtras for Vec3A {
    #[inline]
    fn from_scale_mat(matrix: Mat3A) -> Self {
        vec3a(matrix.x_axis.x, matrix.y_axis.y, matrix.z_axis.z)
    }
}

impl VecExtras for Vec3 {
    #[inline]
    fn from_scale_mat(matrix: Mat3A) -> Self {
        vec3(matrix.x_axis.x, matrix.y_axis.y, matrix.z_axis.z)
    }
}
