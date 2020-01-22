use specs::prelude::*;

// use crate::DeltaTime;

//1. Just point lights
//2. Add Light Type enum

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, Component)]
pub struct Light {
    color    : glm::Vec3,
    strength : f32,
    radius   : f32,
}

impl Light {
    pub fn new() -> Light {
        Light {
            color : glm::vec3(1.0, 1.0, 1.0),
            strength : 1.0,
            radius : 10.0,
        }
    }
}