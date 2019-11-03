use specs::prelude::*;

use crate::DeltaTime;

#[derive(Debug, Component)]
pub struct Transform {
    pub position : glm::Vec3,
    pub rotation : glm::Quat, //TODO: need to always ensure this is a unit quaternion, as it represents an orientation
    pub scale    : glm::Vec3,
}

impl Transform {
    pub fn new() -> Transform {
        Transform {
            position : glm::vec3(0.0, 0.0, 0.0),
            rotation : glm::quat(0.0, 1.0, 0.0, 0.0),
            scale    : glm::vec3(1.0, 1.0, 1.0),
        }
    }
}

#[derive(Debug, Component, Deref, DerefMut)]
pub struct Velocity(pub glm::Vec3);

#[derive(Debug, Component, Deref, DerefMut)]
pub struct AngularVelocity(pub glm::Quat);

#[derive(Debug, Component, Deref, DerefMut)]
pub struct Acceleration(pub glm::Vec3);

#[derive(Debug, Component, Deref, DerefMut)]
pub struct Drag(pub f32);

pub struct UpdatePositionSystem;

impl<'a> System<'a> for UpdatePositionSystem {

    type SystemData = ( Read<'a, DeltaTime>, 
                        WriteStorage<'a, Transform>, 
                        ReadStorage<'a, Velocity>
                      );

    fn run(&mut self, (dt, mut transforms, velocities): Self::SystemData) {

        (&velocities, &mut transforms)
        .par_join()
        .for_each(|(vel, transform)| {
            transform.position += vel.0 * dt.0;
        });
    }
}

pub struct UpdateVelocitySystem;

//TODO: Separate systems for Accel and Drag? so we don't require both?
impl<'a> System<'a> for UpdateVelocitySystem {

    type SystemData = ( Read<'a, DeltaTime>, 
                        WriteStorage<'a, Velocity>, 
                        ReadStorage<'a, Acceleration>,
                        ReadStorage<'a, Drag>);

    fn run(&mut self, (dt, mut vel, acc, drag): Self::SystemData) {    
        for (vel, acc) in (&mut vel, &acc).join() {
            vel.0 += acc.0 * dt.0;
        }

        //drag causes velocity to approach zero from either direction (pos/neg velocity values)
        for (vel, drag) in (&mut vel, &drag).join() {
            let new_abs_vel = glm::clamp(&(vel.0.abs() - glm::vec3(drag.0, drag.0, drag.0) * dt.0), 0.0, std::f32::MAX);

            vel.x = new_abs_vel.x.copysign(vel.x);
            vel.y = new_abs_vel.y.copysign(vel.y);
            vel.z = new_abs_vel.z.copysign(vel.z);
        }
    }
}

pub struct UpdateRotationSystem;

impl<'a> System<'a> for UpdateRotationSystem {

    type SystemData = ( Read<'a, DeltaTime>, 
                        WriteStorage<'a, Transform>, 
                        ReadStorage<'a, AngularVelocity>
                      );

    fn run(&mut self, (dt, mut transforms, angular_velocities): Self::SystemData) {

        (&angular_velocities, &mut transforms)
        .par_join()
        .for_each(|(angular_velocity, transform)| {
            let mut scaled_angular_velocity = angular_velocity.0;
            scaled_angular_velocity.w *= dt.0;
            transform.rotation = glm::quat_normalize(&(transform.rotation * scaled_angular_velocity));
        });
    }
}