use specs::prelude::*;

// use crate::DeltaTime;

//1. Just point lights
//2. Add Light Type enum

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, Component)]
pub struct Light {
    pub position : glm::Vec3,
    pub strength : f32,
    pub color    : glm::Vec3,
    pub radius   : f32,
}

impl Light {
    pub fn new() -> Light {
        Light {
            position : glm::vec3(0.0, 0.0, 0.0),
            color : glm::vec3(1.0, 1.0, 1.0),
            strength : 1.0,
            radius : 10.0,
        }
    }
}