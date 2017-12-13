pub trait ActivationPolicy {
    fn activate(&self, value: f32) -> f32;
    fn derivative(&self, value: f32) -> f32;
}

#[derive(Default)]
pub struct Relu {
}

impl ActivationPolicy for Relu {
    fn activate(&self, value: f32) -> f32 {
        if value < 0.0 {
            0.0
        } else {
            value
        }
    }

    fn derivative(&self, value: f32) -> f32 {
        if value < 0.0 {
            0.0
        } else {
            1.0
        }
    }
}
